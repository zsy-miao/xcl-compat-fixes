//! Textures a title places in its own memory (CreatePlacedResourceX): on the console the GPU
//! samples that memory, laid out as the console's layout library says (`hle::xg`, checked
//! against the console's own). Here a PC texture holds a detiled copy: uploaded when the
//! texture is created and again before the title's next ExecuteCommandLists once the CPU wrote
//! any of its pages (unified memory and the title's graphics reservations are write-watched,
//! `gpu::uma`, `kernelx::VirtualAlloc`) – textures streamed in after creation (mips loaded
//! later, physical pages mapped in, DirectStorage) included.
//!
//! The first upload is not done when the texture is made (a title makes thousands while it
//! streams – Fortnite's render thread then spent its time in CreatePlacedResourceX and its
//! render command fences ran late): it is pending until the title's next ExecuteCommandLists,
//! where all pending and rewritten textures go in batches on one list each.
//!
//! The copy waits for the GPU on an XCL queue of the device. Memory that is not readable
//! (reserved, not mapped yet) reads as zero. GPU writes into the title's memory (copies to the
//! address) are not seen; the title's own GPU work on the texture goes to the PC texture.
//!
//! A copy starts and ends in the state the title's submitted work left the texture in: its
//! transitions of these textures are noted per PC command list as they are recorded
//! (`note_transitions`) and applied, in submission order, when the list is submitted
//! (`submitted`) – per subresource once a transition names one.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::texture::{self, Subresource};
use crate::hle::xg;

/// GetWriteWatch with atomic reset of the reported writes.
const WRITE_WATCH_FLAG_RESET: u32 = 1;
#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// (device, address its bytes are read from, layout, states) of an upload
type Upload = (usize, usize, Arc<xg::Texture>, Vec<D3D12_RESOURCE_STATES>);

struct Entry {
    device: usize,
    /// the title's address of the texture
    va: usize,
    /// where its bytes are read and its CPU writes watched: `va`, or – for a texture in a GPU
    /// view of the title's CPU memory with nothing mapped there – the same offset in the CPU
    /// view (`kernelx::mirrors`)
    src: usize,
    /// `va` lies in a GPU view: `src` is checked again when that view's mappings change
    in_gpu_view: bool,
    tex: Arc<xg::Texture>,
    /// the state the title's submitted work leaves the texture in (uploads leave it there):
    /// the creation state, then what its transitions give – one state for every subresource,
    /// or one per subresource (D3D12 order) while they differ
    states: Vec<D3D12_RESOURCE_STATES>,
    /// not uploaded yet (made since the last submit)
    pending: bool,
    /// the title's GPU work wrote it (CopyTextureRegion into it): its CPU view is stale from
    /// then on and no longer uploaded
    gpu_written: bool,
    /// backed by shared `vram::Pages` tiles (overlapping placements share video memory)
    placed: bool,
    /// tiles were unmapped after a different-range placement superseded this texture
    unmapped: bool,
    /// monotonic registration order (newest wins when uncovering shadowed textures)
    seq: u64,
}

/// textures waiting for their first upload
static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// raw COM pointers of the title's objects; uploads hold a reference while they run
unsafe impl Send for Entry {}

struct Table {
    /// Active PC textures (newest at each range) → its place in the title's memory
    by_res: BTreeMap<usize, Entry>,
    /// Active `(va, PC texture)` → end address (`va + size`); entries with different `(va, end)` are disjoint
    by_va: BTreeMap<(usize, usize), usize>,
    /// Older overlapped textures kept dormant until released or uncovered when a newer one is released
    shadowed: BTreeMap<usize, Entry>,
    /// Shadowed `(va, PC texture)` → end address (may overlap)
    shadowed_by_va: BTreeMap<(usize, usize), usize>,
    /// Largest byte span currently in `shadowed` (bounds range queries)
    max_shadowed_size: usize,
    next_seq: u64,
}

impl Table {
    const fn new() -> Self {
        Table {
            by_res: BTreeMap::new(),
            by_va: BTreeMap::new(),
            shadowed: BTreeMap::new(),
            shadowed_by_va: BTreeMap::new(),
            max_shadowed_size: 0,
            next_seq: 0,
        }
    }

    fn has_active_overlap(&self, va: usize, end: usize) -> bool {
        if let Some((_, &oend)) = self.by_va.range(..(va, 0)).next_back() {
            if oend > va {
                return true;
            }
        }
        end > va && self.by_va.range((va, 0)..(end, 0)).next().is_some()
    }

    fn active_overlaps(&self, va: usize, end: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (&(ova, ores), &oend) in self.by_va.range(..(va, 0)).rev() {
            if oend <= va {
                break;
            }
            out.push((ova, ores, oend));
        }
        if end > va {
            for (&(ova, ores), &oend) in self.by_va.range((va, 0)..(end, 0)) {
                out.push((ova, ores, oend));
            }
        }
        out
    }

    fn shadowed_overlaps(&self, va: usize, end: usize) -> Vec<(usize, usize, usize)> {
        if self.shadowed.is_empty() || end <= va {
            return Vec::new();
        }
        let lo = va.saturating_sub(self.max_shadowed_size);
        self.shadowed_by_va
            .range((lo, 0)..(end, 0))
            .filter(|&(_, &send)| send > va)
            .map(|(&(sva, sres), &send)| (sva, sres, send))
            .collect()
    }

    /// Permanently retire active and shadowed entries overlapping `[va, end)` (used when a
    /// GPU-owned render target / depth / UAV overwrites the address range). Returns
    /// `(pending_delta, retired_count, placed_res_to_unmap)`.
    fn retire_range(&mut self, va: usize, end: usize) -> (isize, usize, Vec<usize>) {
        let (mut pending_delta, mut retired, mut to_unmap) = (0isize, 0usize, Vec::new());
        for (ova, ores, _) in self.active_overlaps(va, end) {
            self.by_va.remove(&(ova, ores));
            if let Some(old) = self.by_res.remove(&ores) {
                if old.pending {
                    pending_delta -= 1;
                }
                if old.placed && !old.unmapped {
                    to_unmap.push(ores);
                }
                retired += 1;
            }
        }
        for (sva, sres, _) in self.shadowed_overlaps(va, end) {
            self.shadowed_by_va.remove(&(sva, sres));
            if let Some(old) = self.shadowed.remove(&sres) {
                if old.placed && !old.unmapped {
                    to_unmap.push(sres);
                }
                retired += 1;
            }
        }
        if self.shadowed.is_empty() {
            self.max_shadowed_size = 0;
        }
        (pending_delta, retired, to_unmap)
    }

    /// Insert `res`, shadowing older active entries whose memory `[va, va + size)` it overlaps so
    /// dead placements never re-upload over shared `vram::Pages` tiles (two committed resources
    /// with the exact same range may coexist as format aliases). Placed entries superseded by a
    /// *different* range are returned in `to_unmap` so their unshared tiles return to the pool;
    /// exact same-range entries stay mapped because they already share every tile.
    fn insert(&mut self, res: usize, mut entry: Entry) -> (isize, Vec<usize>) {
        let va = entry.va;
        let end = va.saturating_add(entry.tex.size() as usize);
        let mut pending_delta = 0isize;
        let mut to_unmap = Vec::new();
        if let Some(old) = self.by_res.remove(&res) {
            self.by_va.remove(&(old.va, res));
            if old.pending {
                pending_delta -= 1;
            }
        }
        if let Some(old) = self.shadowed.remove(&res) {
            self.shadowed_by_va.remove(&(old.va, res));
        }
        for (ova, ores, oend) in self.active_overlaps(va, end) {
            if !entry.placed && ova == va && oend == end && self.by_res.get(&ores).is_some_and(|e| !e.placed) {
                continue;
            }
            self.by_va.remove(&(ova, ores));
            if let Some(mut old) = self.by_res.remove(&ores) {
                if old.pending {
                    pending_delta -= 1;
                }
                if old.placed && !old.unmapped && (ova != va || oend != end) {
                    old.unmapped = true;
                    to_unmap.push(ores);
                }
                self.max_shadowed_size = self.max_shadowed_size.max(oend.saturating_sub(ova));
                self.shadowed_by_va.insert((ova, ores), oend);
                self.shadowed.insert(ores, old);
            }
        }
        // Also release tiles of any already-shadowed exact-range sibling now overwritten by a
        // different range (e.g. A and B shared an unaligned range, then C overwrites both).
        for (sva, sres, send) in self.shadowed_overlaps(va, end) {
            if sva != va || send != end {
                if let Some(old) = self.shadowed.get_mut(&sres) {
                    if old.placed && !old.unmapped {
                        old.unmapped = true;
                        to_unmap.push(sres);
                    }
                }
            }
        }
        self.next_seq += 1;
        entry.seq = self.next_seq;
        if entry.pending {
            pending_delta += 1;
        }
        self.by_va.insert((va, res), end);
        self.by_res.insert(res, entry);
        (pending_delta, to_unmap)
    }

    /// Remove `res` on COM release, reactivating any shadowed older textures it uncovered (newest
    /// first). Returns `(was_present, pending_delta, unmapped_res_to_remap)`.
    fn remove(&mut self, res: usize) -> (bool, isize, Vec<usize>) {
        if let Some(old) = self.shadowed.remove(&res) {
            self.shadowed_by_va.remove(&(old.va, res));
            if self.shadowed.is_empty() {
                self.max_shadowed_size = 0;
            }
            return (true, 0, Vec::new());
        }
        let Some(old) = self.by_res.remove(&res) else { return (false, 0, Vec::new()) };
        self.by_va.remove(&(old.va, res));
        let mut pending_delta = if old.pending { -1 } else { 0 };
        let va = old.va;
        let end = va.saturating_add(old.tex.size() as usize);
        let mut cands: Vec<(u64, usize, usize, usize)> = self
            .shadowed_overlaps(va, end)
            .into_iter()
            .filter_map(|(sva, sres, send)| self.shadowed.get(&sres).map(|e| (e.seq, sres, sva, send)))
            .collect();
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut to_remap = Vec::new();
        for (_, sres, sva, send) in cands {
            if self.has_active_overlap(sva, send) {
                continue;
            }
            self.shadowed_by_va.remove(&(sva, sres));
            if let Some(mut e) = self.shadowed.remove(&sres) {
                if e.unmapped {
                    e.unmapped = false;
                    to_remap.push(sres);
                    if !e.gpu_written {
                        e.pending = true;
                    }
                }
                if e.pending {
                    pending_delta += 1;
                }
                self.by_va.insert((sva, sres), send);
                self.by_res.insert(sres, e);
            }
        }
        if self.shadowed.is_empty() {
            self.max_shadowed_size = 0;
        }
        (true, pending_delta, to_remap)
    }

    fn get_mut(&mut self, res: usize) -> Option<&mut Entry> {
        let Self { by_res, shadowed, .. } = self;
        by_res.get_mut(&res).or_else(|| shadowed.get_mut(&res))
    }

    fn overlapping(&self, va: usize, len: usize) -> Vec<(usize, usize)> {
        self.active_overlaps(va, va.saturating_add(len)).into_iter().take(4).map(|(ova, ores, _)| (ores, ova)).collect()
    }
}

fn adjust_pending(delta: isize) {
    if delta > 0 {
        PENDING.fetch_add(delta as usize, Ordering::Relaxed);
    } else if delta < 0 {
        PENDING.fetch_sub((-delta) as usize, Ordering::Relaxed);
    }
}

unsafe fn clone_resources(ptrs: &[usize]) -> Vec<ID3D12Resource> {
    ptrs.iter()
        .filter_map(|&raw| {
            let p = raw as *mut c_void;
            ID3D12Resource::from_raw_borrowed(&p).map(|r| r.clone())
        })
        .collect()
}

/// PC texture → its place in the title's memory
static TEXTURES: Mutex<Table> = Mutex::new(Table::new());

#[cfg(test)]
#[path = "title_textures_mirror_tests.rs"]
mod mirror_tests;
// The dirty-page scan consumes write-watch state for every submission thread.
// Keep the scan, pending-list update and GPU uploads together.
static REFRESH: Mutex<()> = Mutex::new(());
static TEXTURES_GEN: AtomicU64 = AtomicU64::new(0);
static TEXTURES_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// cached `watch_ranges` keyed by `(sparse::mapped_gen, TEXTURES_GEN, graphics_reservations_gen)`
static WATCH_CACHE: Mutex<Option<((u64, u64, u64), Arc<Vec<(usize, usize)>>)>> = Mutex::new(None);
/// the queue uploads go to, per device
static QUEUES: Mutex<BTreeMap<usize, ID3D12CommandQueue>> = Mutex::new(BTreeMap::new());
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// Number of textures kept in title memory and uploads done so far.
pub fn counts() -> (usize, u64) {
    (TEXTURES.lock().unwrap().by_res.len(), UPLOADS.load(Ordering::Relaxed))
}

/// Whether a non-depth render target / UAV texture is small enough to be a CPU-filled glyph or UI
/// atlas (`<= 1M` pixels, up to `2048` on either axis, e.g. a retail title's `2048x128` R8 glyph cache).
fn is_small_cpu_filled_target(w: u64, h: u32, samples: u32, flags: u32) -> bool {
    let pixels = w.saturating_mul(h as u64);
    w <= 2048 && h <= 2048 && (64 * 64..=1024 * 1024).contains(&pixels) && samples <= 1 && flags & 0x2 == 0
}

/// A texture the title placed at `va` (its console desc `desc`): kept in step with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, desc: &[u8; 0x38], state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    // (a new texture object, or the address of an old one again: its first upload is not left out)
    LAST_CONTENT.lock().unwrap().remove(&(res as usize));
    STREAM_RECHECK.lock().unwrap().remove(&(res as usize));
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    // the console's 32-bit 116 / 117 floats are half floats on the PC (`pc_resource_desc`): the
    // title's bytes are not the resource's texels
    if matches!(u32::from_le_bytes(desc[0x20..0x24].try_into().unwrap()), 116 | 117) {
        return Err("a console float format the PC holds as R16G16B16A16_FLOAT".into());
    }
    // Render targets and depth buffers are the GPU's (their metadata planes, empty planes) –
    // except a small single-sample colour target with texels: a retail title makes its glyph
    // caches that way (256x256 R8 with ALLOW_RENDER_TARGET, two planes) and fills them from the
    // CPU. Left to the GPU they stayed as they were made and its menus' body text lost most of
    // its letters. Such a target is uploaded like any texture when the CPU wrote its pages; one
    // the title only renders to is not CPU-written and so not uploaded over.
    // `XCL_SYNC_SMALL_TARGETS=0` turns this off.
    let small_target = {
        let (w, h) = (u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap()), u32::from_le_bytes(desc[0x18..0x1c].try_into().unwrap()));
        let (samples, flags) = (u32::from_le_bytes(desc[0x24..0x28].try_into().unwrap()), u32::from_le_bytes(desc[0x30..0x34].try_into().unwrap()));
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("XCL_SYNC_SMALL_TARGETS").is_ok_and(|v| v.trim() == "0"));
        !off && is_small_cpu_filled_target(w, h, samples, flags)
    };
    if tex.texels().is_none() || (tex.planes.len() != 1 && !small_target) {
        return Err("a render target / depth buffer".into());
    }
    if tex.planes.len() != 1 {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a small render target at {va:#x} ({} planes) is kept in step with the title's memory – a glyph cache the CPU fills ({n} so far)", tex.planes.len());
        }
    }
    if ID3D12Resource::from_raw_borrowed(&res).is_none() {
        return Err("no resource".into());
    }
    let src = mirror_source(va);
    if src != va {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a texture at {va:#x} lies in a GPU view of the title's CPU memory: kept in step with {src:#x} ({n} so far)");
        }
    }
    let in_gpu_view = crate::hle::kernelx::mirrors::source(va).is_some();
    let e = Entry {
        device: device as usize,
        va,
        src,
        in_gpu_view,
        tex: Arc::new(tex),
        states: vec![state],
        pending: true,
        gpu_written: false,
        placed: super::vram::is_placed_texture(res as usize),
        unmapped: false,
        seq: 0,
    };
    let to_unmap = {
        let mut t = TEXTURES.lock().unwrap();
        let (delta, unmap_ptrs) = t.insert(res as usize, e);
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        clone_resources(&unmap_ptrs)
    };
    for r in &to_unmap {
        // #504: remapped/unmapped textures must not keep a content hash – their VRAM tiles
        // are gone and need a fresh upload even if the CPU bytes did not change.
        LAST_CONTENT.lock().unwrap().remove(&(r.as_raw() as usize));
        super::vram::unmap_superseded(r);
    }
    TEXTURES_GEN.fetch_add(1, Ordering::Release);
    Ok(())
}

/// Where a texture at `va` is read from: the CPU view behind a GPU view (`kernelx::mirrors`)
/// when nothing is mapped at `va` itself, else `va`. A physical page mapped at `va` counts
/// before its first use too, while it is still reserved memory (`kernelx::physical_page_at`).
fn mirror_source(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

/// Physical pages were mapped or unmapped in a GPU view since the last look: textures there
/// whose source changed (a page now mapped under one, or gone from it) read from the new one,
/// uploaded again at this submit.
fn recheck_gpu_view_sources() {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let gen = crate::hle::kernelx::gpu_view_map_gen();
    if SEEN.swap(gen, Ordering::AcqRel) == gen {
        return;
    }
    let mut changed = false;
    // (the shadowed textures too: one comes back when the texture over it goes)
    let mut g = TEXTURES.lock().unwrap();
    let t = &mut *g;
    for e in t.by_res.values_mut().chain(t.shadowed.values_mut()) {
        changed |= recheck_source(e);
    }
    drop(g);
    if changed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// `e`'s source again (a texture in a GPU view): when it changed, `e` reads the new one and is
/// uploaded at the next submit. Whether it changed.
fn recheck_source(e: &mut Entry) -> bool {
    if !e.in_gpu_view {
        return false;
    }
    let src = mirror_source(e.va);
    if src == e.src {
        return false;
    }
    e.src = src;
    if !e.pending {
        e.pending = true;
        PENDING.fetch_add(1, Ordering::Relaxed);
    }
    true
}

/// The title's GPU work writes texture `res` (a copy into it): on the console the memory then
/// holds what the GPU wrote; here the CPU's view of it is stale, so it is not uploaded again
/// (a CPU write elsewhere on its pages – a neighbour, an allocator's header – would upload the
/// stale view over the copy: Fortnite's font atlas is zeros in memory, its glyphs are copied in
/// by the GPU). The first upload, if still pending, stays: it runs before the title's submit
/// that copies.
pub fn note_gpu_write(res: usize) {
    if super::vram::level_pages() {
        // (a texture whose levels get their pages one by one: all of them before the GPU's write)
        let p = res as *mut c_void;
        unsafe {
            if let Some(r) = ID3D12Resource::from_raw_borrowed(&p) {
                super::vram::map_levels(r, None);
            }
        }
    }
    let mut marked = false;
    {
        let mut t = TEXTURES.lock().unwrap();
        if let Some(e) = t.get_mut(res) {
            if !e.gpu_written {
                e.gpu_written = true;
                GPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
                marked = true;
            }
        }
    }
    if marked {
        STREAM_RECHECK.lock().unwrap().remove(&res);
    }
}

static GPU_WRITTEN: AtomicU64 = AtomicU64::new(0);

/// The textures kept in step whose memory `[va, va + len)` overlaps (PC texture, its address).
pub fn overlapping(va: usize, len: usize) -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap().overlapping(va, len)
}

/// A GPU-owned placement (render target / depth / UAV) now occupies `[va, va + len)`: retire
/// superseded sampled textures there so CPU dirty scans do not upload over the shared tiles and
/// any unshared extra tiles return to the pool.
pub(super) unsafe fn retire_overlapping(va: usize, len: usize) {
    if !any() || len == 0 {
        return;
    }
    let src = mirror_source(va);
    let gpu_aliases: Vec<usize> = crate::hle::kernelx::mirrors::gpu_views_of(va)
        .into_iter()
        .filter(|&g| g != va && mirror_source(g) == va)
        .collect();
    let (retired, to_unmap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (mut delta, mut retired, mut unmap_ptrs) = t.retire_range(va, va.saturating_add(len));
        if src != va {
            let (d2, r2, u2) = t.retire_range(src, src.saturating_add(len));
            delta += d2;
            retired += r2;
            unmap_ptrs.extend(u2);
        }
        for g in gpu_aliases {
            let (d3, r3, u3) = t.retire_range(g, g.saturating_add(len));
            delta += d3;
            retired += r3;
            unmap_ptrs.extend(u3);
        }
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        (retired, clone_resources(&unmap_ptrs))
    };
    for r in &to_unmap {
        LAST_CONTENT.lock().unwrap().remove(&(r.as_raw() as usize));
        super::vram::unmap_superseded(r);
    }
    if retired > 0 {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    LAST_CONTENT.lock().unwrap().remove(&res);
    STREAM_RECHECK.lock().unwrap().remove(&res);
    let (removed, to_remap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (removed, delta, remap_ptrs) = t.remove(res);
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        (removed, unsafe { clone_resources(&remap_ptrs) })
    };
    for r in &to_remap {
        // #504: a texture remapped after its shadower is released must upload again.
        LAST_CONTENT.lock().unwrap().remove(&(r.as_raw() as usize));
        unsafe {
            super::vram::remap_if_unmapped(r);
            // (lazy levels: one the GPU wrote is not uploaded again, so all of it has pages)
            if super::vram::level_pages() && TEXTURES.lock().unwrap().by_res.get(&(r.as_raw() as usize)).is_some_and(|e| e.gpu_written) {
                super::vram::map_levels(r, None);
            }
        }
    }
    if removed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The title-memory address of a texture kept in step with it.
pub fn address(res: usize) -> Option<usize> {
    let t = TEXTURES.lock().unwrap();
    t.by_res.get(&res).or_else(|| t.shadowed.get(&res)).map(|e| e.va)
}

/// (texture, subresource or ALL_SUBRESOURCES, state after) of a transition the title recorded
type Transition = (usize, u32, D3D12_RESOURCE_STATES);

/// PC command list → the transitions of textures kept in step it recorded since it was made or
/// last reset, in order. They apply when the list is submitted – each time: a closed list may
/// run again.
static LIST_NOTES: Mutex<BTreeMap<usize, Vec<Transition>>> = Mutex::new(BTreeMap::new());

/// Whether any texture is kept in step (callers skip collecting transitions otherwise).
pub fn any() -> bool {
    TEXTURES_LEN.load(Ordering::Relaxed) != 0
}

/// The title recorded `transitions` on PC list `list`: those of textures kept in step are noted
/// for when the list is submitted.
pub fn note_transitions(list: usize, transitions: &[Transition]) {
    if !any() || transitions.is_empty() {
        return;
    }
    let kept: Vec<Transition> = {
        let t = TEXTURES.lock().unwrap();
        transitions.iter().filter(|(r, _, _)| t.by_res.contains_key(r) || t.shadowed.contains_key(r)).copied().collect()
    };
    if !kept.is_empty() {
        LIST_NOTES.lock().unwrap().entry(list).or_default().extend(kept);
    }
}

/// PC list `list` starts a new recording (made, reset, released): its noted transitions go.
pub fn list_reset(list: usize) {
    let mut g = LIST_NOTES.lock().unwrap();
    if !g.is_empty() {
        g.remove(&list);
    }
}

/// The PC lists `lists` were submitted, in this order: the textures kept in step are in the
/// states their transitions leave them in.
pub fn submitted(lists: &[*mut c_void]) {
    let notes: Vec<Transition> = {
        let g = LIST_NOTES.lock().unwrap();
        if g.is_empty() {
            return;
        }
        lists.iter().filter_map(|&l| g.get(&(l as usize))).flatten().copied().collect()
    };
    if notes.is_empty() {
        return;
    }
    let mut t = TEXTURES.lock().unwrap();
    for (res, sub, after) in notes {
        if let Some(e) = t.get_mut(res) {
            let n = subresource_count(&e.tex);
            apply_transition(&mut e.states, n, sub, after);
        }
    }
}

/// Subresources of a texture kept in step (as `subresources` makes them).
fn subresource_count(tex: &xg::Texture) -> usize {
    tex.texels().map_or(1, |s| s.array as usize * s.chain.levels.len())
}

/// `states` (one for all `n` subresources, or one each) after a transition of subresource `sub`
/// (or all of them) to `after`; a subresource the texture does not have is ignored.
fn apply_transition(states: &mut Vec<D3D12_RESOURCE_STATES>, n: usize, sub: u32, after: D3D12_RESOURCE_STATES) {
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || (n <= 1 && sub == 0) {
        *states = vec![after];
        return;
    }
    let k = sub as usize;
    if k >= n {
        return;
    }
    if states.len() != n {
        *states = vec![states[0]; n];
    }
    states[k] = after;
    if states.iter().all(|&s| s == after) {
        states.truncate(1);
    }
}

thread_local! {
    /// XCL's own submits on this thread (texture uploads): they read no title memory on the GPU,
    /// so they do not look for CPU writes first – that look is the title's submits' job
    static BUSY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Reusable `GetWriteWatch` output buffer (avoids 128 KiB alloc+zero+free per submit).
    static WATCH_BUF: std::cell::RefCell<Vec<*mut c_void>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Reusable dirty page list for `refresh_once`.
    static DIRTY_BUF: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn cached_watch_ranges(u: &crate::gpu::uma::Uma) -> Arc<Vec<(usize, usize)>> {
    let key = (
        super::sparse::mapped_gen(),
        TEXTURES_GEN.load(Ordering::Acquire),
        crate::hle::kernelx::graphics_reservations_gen(),
    );
    {
        let g = WATCH_CACHE.lock().unwrap();
        if let Some((k, ref ranges)) = *g {
            if k == key {
                return Arc::clone(ranges);
            }
        }
    }
    // Recompute the watched ranges (textures kept in step + mapped sparse tiles clipped to
    // write-watched regions) only when the set of textures, tiles, or reservations changed.
    let mut regions = u.watch_regions();
    regions.extend(crate::hle::kernelx::graphics_reservations());
    let mut wanted: Vec<(usize, usize)> = TEXTURES.lock().unwrap().by_res.values().map(|e| (e.src & !0xFFF, (e.src + e.tex.size() as usize + 0xFFF) & !0xFFF)).collect();
    wanted.extend(super::sparse::mapped_ranges());
    let computed = Arc::new(watch_ranges(&regions, wanted));
    *WATCH_CACHE.lock().unwrap() = Some((key, Arc::clone(&computed)));
    computed
}

/// Run `f` as XCL's own GPU work: ExecuteCommandLists inside it does not refresh.
pub(super) fn own_work<R>(f: impl FnOnce() -> R) -> R {
    let was = BUSY.with(|b| b.replace(true));
    let r = f();
    BUSY.with(|b| b.set(was));
    r
}

/// Whether the current thread is inside XCL's own internal GPU setup/upload/readback work.
pub(super) fn is_own_work() -> bool {
    BUSY.with(|b| b.get())
}

/// Before the title's GPU work: textures whose memory the CPU wrote since the last look are
/// uploaded again.
pub(super) unsafe fn refresh(queue: *mut c_void) {
    // the uploads' own ExecuteCommandLists come back here
    if BUSY.with(|b| b.get()) {
        return;
    }
    let _refresh = REFRESH.lock().unwrap();
    own_work(|| refresh_once(queue));
}

/// The pages of `[base, base + size)` written since the last look, added to `dirty` and reset
/// (`page`: their size). A full `buf` goes on after the last page it reported: starting again at
/// `base` walked the pages already reset each time (4 GiB with every page written: 105 ms a
/// look, 47 ms going on).
unsafe fn take_written(base: usize, size: usize, buf: &mut [*mut c_void], dirty: &mut Vec<usize>, page: &mut u32) {
    let end = base + size;
    let mut at = base;
    while at < end {
        let mut n = buf.len();
        if GetWriteWatch(WRITE_WATCH_FLAG_RESET, at as *const c_void, end - at, buf.as_mut_ptr(), &mut n, page) != 0 {
            break;
        }
        dirty.extend(buf[..n].iter().map(|&p| p as usize));
        if n == 0 || n < buf.len() {
            break;
        }
        at = buf[n - 1] as usize + *page as usize;
    }
}

unsafe fn refresh_once(queue: *mut c_void) {
    let Some(u) = crate::gpu::uma::get() else { return };
    let sparse = u.has_sparse();
    if !sparse && TEXTURES_LEN.load(Ordering::Relaxed) == 0 {
        return;
    }
    recheck_gpu_view_sources();
    let watch_t0 = std::time::Instant::now();
    let mut page = 0x1000u32;
    let ranges = cached_watch_ranges(u);
    let dirty_len = DIRTY_BUF.with(|dcell| {
        let mut dirty = dcell.borrow_mut();
        dirty.clear();
        WATCH_BUF.with(|wcell| {
            let mut buf = wcell.borrow_mut();
            if buf.len() < 16384 {
                buf.resize(16384, std::ptr::null_mut());
            }
            for &(base, size) in ranges.iter() {
                take_written(base, size, &mut buf, &mut dirty, &mut page);
            }
        });
        dirty.sort_unstable();
        dirty.len()
    });
    super::sparse::note_watch(watch_t0.elapsed(), dirty_len);
    if super::present::trace_frame() {
        // every 600th look: what the CPU's view of a few textures holds now (does the title's
        // data reach the address at all, and is the texture still kept in step?)
        static LOOKS: AtomicU64 = AtomicU64::new(0);
        if LOOKS.fetch_add(1, Ordering::Relaxed) % 600 == 599 {
            let t = TEXTURES.lock().unwrap();
            // 16 samples of 4 KiB over the whole texture (GFX10 keeps the small levels first,
            // level 0 last)
            let content = |e: &Entry| -> usize {
                let size = e.tex.size() as usize;
                (0..16usize)
                    .map(|k| e.src + (size.saturating_sub(4096)) * k / 15)
                    .filter(|&a| crate::gpu::shader::readable(a, 4096))
                    .map(|a| std::slice::from_raw_parts(a as *const u8, 4096).iter().filter(|&&b| b != 0).count())
                    .sum()
            };
            let (mut with, mut without) = (0, 0);
            let mut shown = 0;
            for e in t.by_res.values() {
                let c = content(e);
                if c > 0 { with += 1 } else { without += 1 }
                if c > 0 && shown < 4 {
                    shown += 1;
                    crate::xlog_info!("frame trace: title texture at {:#x} ({} bytes) has data in memory ({c} of 65536 sampled bytes); GPU-written {}", e.va, e.tex.size(), e.gpu_written);
                }
            }
            crate::xlog_info!("frame trace: title textures: {with} with data in memory, {without} all zero; {dirty_len} dirty pages this look");
        }
    }
    // the title's own memory: what the CPU wrote goes to its video memory first
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        DIRTY_BUF.with(|dcell| {
            let dirty = dcell.borrow();
            super::sparse::before_execute(q, &dirty, page as usize);
            super::vram::before_execute(q);
        });
    }
    let recheck_frame = {
        let (p, b) = super::present::counters();
        p.max(b)
    };
    recheck_streamed_unwatched(&ranges, recheck_frame);
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 {
        return;
    }
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let todo: Vec<(ID3D12Resource, Upload)> = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let mut t = TEXTURES.lock().unwrap();
        t.by_res
            .iter_mut()
            .filter(|(_, e)| {
                if e.pending {
                    return true;
                }
                if e.gpu_written {
                    return false;
                }
                let i = dirty.partition_point(|&p| p + page as usize <= e.src);
                dirty.get(i).is_some_and(|&p| p < e.src + e.tex.size() as usize)
            })
            .filter_map(|(&res, e)| {
                if e.pending {
                    e.pending = false;
                    PENDING.fetch_sub(1, Ordering::Relaxed);
                }
                let p = res as *mut c_void;
                ID3D12Resource::from_raw_borrowed(&p).map(|r| (r.clone(), (e.device, e.src, Arc::clone(&e.tex), e.states.clone())))
            })
            .collect()
    });
    seed_stream_recheck(&todo, &ranges, recheck_frame);
    upload_all(todo);
}

/// texture → `content_hash` of its memory at its last upload
static LAST_CONTENT: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());
static UNCHANGED: AtomicU64 = AtomicU64::new(0);

/// `XCL_TEXTURE_SKIP_UNCHANGED=0` uploads a texture whenever its pages were written, as before.
fn unchanged_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_SKIP_UNCHANGED").map_or(true, |v| v.trim() != "0"))
}

/// Whether texture `res` was last *successfully* uploaded from memory with hash `h` (#504).
fn last_upload_matches(res: usize, h: u64) -> bool {
    LAST_CONTENT.lock().unwrap().get(&res) == Some(&h)
}

/// Remember a successful upload of `res` from memory with hash `h`.
fn remember_upload(res: usize, h: u64) {
    LAST_CONTENT.lock().unwrap().insert(res, h);
}

/// Test helper: check-then-remember (matches the old `same_as_last_upload` contract for unit tests).
#[cfg(test)]
fn same_as_last_upload(res: usize, h: u64) -> bool {
    let mut g = LAST_CONTENT.lock().unwrap();
    let same = g.get(&res) == Some(&h);
    g.insert(res, h);
    same
}

/// A 64-bit hash of `b` for "the same bytes as before" (four lanes of multiply-rotate over
/// 8-byte words: gigabytes a second; not for anything adversarial).
fn content_hash(b: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = [K, K.rotate_left(17), K.rotate_left(31), K.rotate_left(47)];
    let mut chunks = b.chunks_exact(32);
    for c in &mut chunks {
        for (k, lane) in h.iter_mut().enumerate() {
            let v = u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
            *lane = (*lane ^ v).wrapping_mul(K).rotate_left(29);
        }
    }
    let mut tail = b.len() as u64;
    for (k, &x) in chunks.remainder().iter().enumerate() {
        tail = (tail ^ ((x as u64) << (8 * (k % 8)))).wrapping_mul(K).rotate_left(23);
    }
    h.iter().fold(tail, |a, &l| (a ^ l).wrapping_mul(K).rotate_left(27))
}

/// Frame milestones (relative to last upload or detected content change) at which unwatched
/// textures (e.g. physical-page / `MEM_MAPPED` views that `GetWriteWatch` cannot track) are
/// re-fingerprinted to catch late streaming writes that land after an initial zeroed upload.
const STREAM_RECHECK_MILESTONES: [u64; 5] = [4, 12, 40, 120, 400];
/// Per-slice sample size (start, mid, end -> at most 48 KiB hashed per due texture).
const STREAM_SAMPLE_SLICE: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreamRecheck {
    base_frame: u64,
    stage: u8,
    last_fp: u64,
}

static STREAM_RECHECK: Mutex<BTreeMap<usize, StreamRecheck>> = Mutex::new(BTreeMap::new());
static LAST_RECHECK_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
static STREAM_RECHECK_HITS: AtomicU64 = AtomicU64::new(0);

/// `XCL_TEXTURE_STREAM_RECHECK=0` disables spaced-milestone fingerprinting for unwatched textures.
fn stream_recheck_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TEXTURE_STREAM_RECHECK").map_or(true, |v| v.trim() != "0"))
}

/// Whether every 4 KiB page of `[src, src + len)` is covered by the sorted, disjoint
/// write-watched spans `watched` (`(base, size)` pairs from `cached_watch_ranges`).
fn range_is_fully_watched(src: usize, len: usize, watched: &[(usize, usize)]) -> bool {
    if len == 0 {
        return true;
    }
    let mut lo = src & !0xFFF;
    let hi = src.saturating_add(len).saturating_add(0xFFF) & !0xFFF;
    if lo >= hi {
        return true;
    }
    let mut i = watched.partition_point(|&(b, s)| b.saturating_add(s) <= lo);
    while lo < hi {
        let Some(&(b, s)) = watched.get(i) else {
            return false;
        };
        if b > lo || s == 0 {
            return false;
        }
        lo = b.saturating_add(s);
        i += 1;
    }
    true
}

/// Bounded (<= 48 KiB) start/mid/end fingerprint of a texture's backing memory, read via
/// `read_committed` so uncommitted or concurrently decommitted pages read as zero without faulting.
unsafe fn sample_fingerprint(src: usize, size: usize) -> u64 {
    if size == 0 {
        return 0;
    }
    if size <= 3 * STREAM_SAMPLE_SLICE {
        let buf = read_committed(src, size);
        return content_hash(&buf);
    }
    let mid = src.saturating_add((size - STREAM_SAMPLE_SLICE) / 2);
    let end = src.saturating_add(size - STREAM_SAMPLE_SLICE);
    let h0 = content_hash(&read_committed(src, STREAM_SAMPLE_SLICE));
    let h1 = content_hash(&read_committed(mid, STREAM_SAMPLE_SLICE));
    let h2 = content_hash(&read_committed(end, STREAM_SAMPLE_SLICE));
    let mut mix = [0u8; 32];
    mix[0..8].copy_from_slice(&h0.to_le_bytes());
    mix[8..16].copy_from_slice(&h1.to_le_bytes());
    mix[16..24].copy_from_slice(&h2.to_le_bytes());
    mix[24..32].copy_from_slice(&(size as u64).to_le_bytes());
    content_hash(&mix)
}

/// Record baseline fingerprints for unwatched textures queued into `todo` so late streaming writes
/// into mapped physical pages are caught at subsequent frame milestones.
unsafe fn seed_stream_recheck(todo: &[(ID3D12Resource, Upload)], watched: &[(usize, usize)], frame: u64) {
    if !stream_recheck_enabled() || todo.is_empty() {
        return;
    }
    let mut seeded = Vec::new();
    for (r, (_, src, tex, _)) in todo {
        let size = tex.size() as usize;
        if size == 0 || range_is_fully_watched(*src, size, watched) {
            continue;
        }
        let fp = sample_fingerprint(*src, size);
        seeded.push((r.as_raw() as usize, StreamRecheck { base_frame: frame, stage: 0, last_fp: fp }));
    }
    if !seeded.is_empty() {
        let mut g = STREAM_RECHECK.lock().unwrap();
        for (res, state) in seeded {
            g.insert(res, state);
        }
    }
}

/// At most once per presented frame, re-fingerprint due unwatched textures at spaced milestones.
unsafe fn recheck_streamed_unwatched(watched: &[(usize, usize)], frame: u64) {
    if !stream_recheck_enabled() {
        return;
    }
    if LAST_RECHECK_FRAME.swap(frame, Ordering::AcqRel) == frame {
        return;
    }
    recheck_streamed_unwatched_at(watched, frame);
}

/// Re-fingerprint due unwatched textures at `frame`, re-queuing changed entries and resetting
/// their milestone schedule so subsequent higher-resolution mip streams are also caught.
unsafe fn recheck_streamed_unwatched_at(watched: &[(usize, usize)], frame: u64) {
    let due: Vec<(usize, u64)> = {
        let g = STREAM_RECHECK.lock().unwrap();
        if g.is_empty() {
            return;
        }
        g.iter()
            .filter_map(|(&res, st)| {
                let target = *STREAM_RECHECK_MILESTONES.get(st.stage as usize)?;
                (frame.saturating_sub(st.base_frame) >= target).then_some((res, st.last_fp))
            })
            .collect()
    };
    if due.is_empty() {
        return;
    }

    let mut check = Vec::with_capacity(due.len());
    let mut prune = Vec::new();
    {
        let t = TEXTURES.lock().unwrap();
        for (res, prev_fp) in due {
            match t.by_res.get(&res) {
                Some(e) if e.gpu_written => prune.push(res),
                Some(e) if e.pending => {}
                Some(e) => {
                    let size = e.tex.size() as usize;
                    if range_is_fully_watched(e.src, size, watched) {
                        prune.push(res);
                    } else {
                        check.push((res, e.src, size, prev_fp));
                    }
                }
                None => {
                    if !t.shadowed.contains_key(&res) {
                        prune.push(res);
                    }
                }
            }
        }
    }
    if !prune.is_empty() {
        let mut g = STREAM_RECHECK.lock().unwrap();
        for res in prune {
            g.remove(&res);
        }
    }
    if check.is_empty() {
        return;
    }

    let results: Vec<(usize, u64, u64)> = check
        .into_iter()
        .map(|(res, src, size, prev_fp)| (res, prev_fp, sample_fingerprint(src, size)))
        .collect();

    let mut changed = Vec::new();
    {
        let mut g = STREAM_RECHECK.lock().unwrap();
        for (res, prev_fp, cur_fp) in results {
            let Some(st) = g.get_mut(&res) else {
                continue;
            };
            if cur_fp != prev_fp {
                *st = StreamRecheck { base_frame: frame, stage: 0, last_fp: cur_fp };
                changed.push(res);
            } else {
                st.stage = st.stage.saturating_add(1);
                if st.stage as usize >= STREAM_RECHECK_MILESTONES.len() {
                    g.remove(&res);
                }
            }
        }
    }
    if !changed.is_empty() {
        let mut queued = 0u64;
        {
            let mut t = TEXTURES.lock().unwrap();
            for res in changed {
                if let Some(e) = t.by_res.get_mut(&res) {
                    if !e.gpu_written && !e.pending {
                        e.pending = true;
                        PENDING.fetch_add(1, Ordering::Relaxed);
                        queued += 1;
                    }
                }
            }
        }
        if queued > 0 {
            let total = STREAM_RECHECK_HITS.fetch_add(queued, Ordering::Relaxed) + queued;
            if total == queued || (total.is_power_of_two() && total >= 16) {
                crate::xlog_info!("D3D12.X: title textures: streamed unwatched re-uploads triggered: {total}");
            }
        }
    }
}

/// staging bytes and texture count of one batch at most: bounds the persistent upload staging
/// buffer (`gpu::texture::upload_batch` rounds it up to a power of two and keeps the peak size)
/// and keeps each GPU copy list short enough to avoid WDDM 2 s TDR when thousands of small
/// textures stream at once.
const BATCH_BYTES: usize = 32 << 20;
const BATCH_ITEMS: usize = 256;

/// Whether another upload `(next_dev, next_bytes)` still fits in the current batch of
/// `picked_count` textures totalling `picked_bytes` on `batch_dev`. An empty batch always takes
/// its first upload (even if larger than `BATCH_BYTES`) so progress is guaranteed.
fn batch_fits(picked_count: usize, picked_bytes: usize, batch_dev: usize, next_dev: usize, next_bytes: usize) -> bool {
    picked_count == 0
        || (picked_count < BATCH_ITEMS
            && next_dev == batch_dev
            && picked_bytes.saturating_add(next_bytes) <= BATCH_BYTES)
}

/// The uploads `todo`, in batches of up to `BATCH_BYTES` and `BATCH_ITEMS` (one list and one wait each).
unsafe fn upload_all(todo: Vec<(ID3D12Resource, Upload)>) {
    static N: AtomicU64 = AtomicU64::new(0);
    let warn = |e: &str| {
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture: upload failed: {e}");
        }
    };
    let mut todo = todo.into_iter().peekable();
    while todo.peek().is_some() {
        let mut picked = Vec::new();
        let mut snapshots = Vec::new();
        let mut bytes = 0usize;
        let mut device = 0usize;
        while let Some((_, (dev, _, tex, _))) = todo.peek() {
            if !batch_fits(picked.len(), bytes, device, *dev, tex.size() as usize) {
                break;
            }
            let (r, (dev, va, tex, states)) = todo.next().unwrap();
            // Read once: hash and detile from the same snapshot (#504 TOCTOU / double-read).
            let mem = read_committed(va, tex.size() as usize);
            let h = content_hash(&mem);
            let res_key = r.as_raw() as usize;
            // The same bytes as at this texture's last *successful* upload: nothing to do.
            if unchanged_skip() && last_upload_matches(res_key, h) {
                let n = UNCHANGED.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || (n.is_power_of_two() && n >= 1024) {
                    crate::xlog_info!("title textures: {n} uploads left out so far – the texture's memory held what it held at its last upload");
                }
                continue;
            }
            device = dev;
            bytes += tex.size() as usize;
            picked.push((r, va, states, res_key, h));
            snapshots.push((mem, tex));
        }
        if picked.is_empty() {
            continue;
        }
        let subs = subresources_of_all(snapshots, bytes);
        let mut batch = Vec::with_capacity(picked.len());
        for ((r, va, states, res_key, h), subs) in picked.into_iter().zip(subs) {
            if super::present::trace_frame() {
                note_content(va, &r, &subs);
            }
            if super::vram::level_pages() {
                // (the queue waits for these mappings before the copies: `vram::before_execute` below)
                map_levels_with_texels(&r, &subs);
            }
            batch.push((r, subs, states, res_key, h));
        }
        let d = device as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { continue };
        let queue = match upload_queue(dev, device) {
            Ok(q) => q,
            Err(e) => {
                warn(&e);
                continue;
            }
        };
        let keys: Vec<_> = batch.iter().map(|(_, _, _, res, h)| (*res, *h)).collect();
        // Transfer the detiled bytes; cloning every pixel here would double the batch.
        let upload_items: Vec<_> = batch.into_iter().map(|(r, subs, states, _, _)| (r, subs, states)).collect();
        match own_work(|| {
            super::vram::before_execute(&queue);
            texture::upload_batch(dev, &queue, &upload_items)
        }) {
            Ok(failed) => {
                let failed_idx: std::collections::BTreeSet<usize> = failed.iter().map(|(i, _)| *i).collect();
                for (_, e) in &failed {
                    warn(e);
                }
                for (i, (res, h)) in keys.iter().enumerate() {
                    if !failed_idx.contains(&i) {
                        remember_upload(*res, *h);
                    }
                }
                UPLOADS.fetch_add(upload_items.len() as u64, Ordering::Relaxed);
            }
            Err(e) => warn(&e),
        }
    }
}

/// (`vram::level_pages`) Before an upload of title texture `r`: its levels with texels in `subs`
/// (its subresources as they go up) get their video memory pages if they have none - a texture
/// placed with lazy levels; any other: nothing to do. A level of zeros has no texels: a streamed
/// texture's larger levels before the title has memory behind them (read as zeros,
/// `read_committed`). It keeps no pages, reads as zeros as it would with them, and its copy goes
/// to tiles that are not mapped, which drops it (tiled resources tier 2).
unsafe fn map_levels_with_texels(r: &ID3D12Resource, subs: &[Subresource]) {
    if !super::vram::has_lazy_levels(r.as_raw() as usize) {
        return;
    }
    // (one slice: its subresources are its levels)
    let present: Vec<u32> = subs.iter().enumerate().filter(|(_, s)| !all_zero(&s.data)).map(|(i, _)| i as u32).collect();
    super::vram::map_levels(r, Some(&present));
}

/// Whether every byte of `b` is zero (eight at a time: gigabytes a second where it is).
fn all_zero(b: &[u8]) -> bool {
    let mut words = b.chunks_exact(8);
    words.all(|w| u64::from_ne_bytes(w.try_into().unwrap()) == 0) && words.remainder().iter().all(|&x| x == 0)
}

/// `XCL_TRACE_FRAME`: how much of a title texture's top level is not zero, each time it is
/// uploaded (the first 48 uploads of textures of 64x64 and more).
fn note_content(va: usize, r: &ID3D12Resource, subs: &[texture::Subresource]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let Some(top) = subs.first() else { return };
    let d = unsafe { r.GetDesc() };
    if d.Width < 64 || d.Height < 64 || N.fetch_add(1, Ordering::Relaxed) >= 48 {
        return;
    }
    let nonzero = top.data.iter().filter(|&&b| b != 0).count();
    crate::xlog_info!(
        "frame trace: title texture at {va:#x} ({}x{} format {}) uploaded: {nonzero} of {} bytes of level 0 not zero",
        d.Width,
        d.Height,
        d.Format.0,
        top.data.len()
    );
}

/// `XCL_TRACE_FRAME`: what the GPU holds in level 0 of title texture `res` (read back on XCL's
/// queue; the texture is taken to be in a shader-resource state) against the CPU's view of the
/// memory now. Each texture once, four at most.
pub unsafe fn probe_gpu_content(res: usize) {
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&res) || g.len() >= 4 {
            return;
        }
        g.push(res);
    }
    let Some((device, va, tex, gpu_written)) = TEXTURES.lock().unwrap().by_res.get(&res).map(|e| (e.device, e.src, Arc::clone(&e.tex), e.gpu_written)) else { return };
    let d = device as *mut c_void;
    let r = res as *mut c_void;
    let (Some(dev), Some(r)) = (ID3D12Device::from_raw_borrowed(&d), ID3D12Resource::from_raw_borrowed(&r)) else { return };
    let rd = r.GetDesc();
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
    dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
    let result = own_work(|| -> Result<Vec<u8>, String> {
        let queue = upload_queue(dev, device)?;
        super::vram::before_execute(&queue);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| e.to_string())?;
        let rb = rb.ok_or("no readback buffer")?;
        if super::real::dred_on() {
            super::real::name_resource(rb.as_raw(), "xcl title_tex readback");
        }
        let barrier = |a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: 0, StateBefore: a, StateAfter: b }),
            },
        };
        let srv = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        list.ResourceBarrier(&[barrier(srv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, srv)]);
        list.Close().map_err(|e| e.to_string())?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(|e| e.to_string())?)]);
        texture::wait(dev, &queue)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
        for y in 0..rows as usize {
            out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
        }
        rb.Unmap(0, None);
        Ok(out)
    });
    let cpu = subresources(va, &tex).into_iter().next().map(|s| s.data).unwrap_or_default();
    match result {
        Ok(gpu) => {
            let nonzero = gpu.iter().filter(|&&b| b != 0).count();
            let same = gpu.iter().zip(cpu.iter()).filter(|(a, b)| a == b).count();
            crate::xlog_info!(
                "frame trace: GPU content of title texture {res:#x} at {va:#x} ({}x{} format {}, {} mips, GPU-written {gpu_written}): {nonzero} of {} bytes of level 0 not zero; {same} equal to the CPU view ({} of its bytes not zero); first bytes {:02x?}",
                rd.Width,
                rd.Height,
                rd.Format.0,
                rd.MipLevels,
                gpu.len(),
                cpu.iter().filter(|&&b| b != 0).count(),
                &gpu[..gpu.len().min(16)]
            );
        }
        Err(e) => crate::xlog_warn!("frame trace: GPU content of title texture {res:#x}: {e}"),
    }
}

/// XCL's upload queues made so far.
pub(super) fn upload_queues() -> Vec<ID3D12CommandQueue> {
    QUEUES.lock().unwrap().values().cloned().collect()
}

/// XCL's queue for uploads on `dev`.
unsafe fn upload_queue(dev: &ID3D12Device, device: usize) -> Result<ID3D12CommandQueue, String> {
    let mut q = QUEUES.lock().unwrap();
    if let Some(q) = q.get(&device) {
        return Ok(q.clone());
    }
    let made: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| e.to_string())?;
    q.insert(device, made.clone());
    Ok(made)
}

/// Ranges closer than this are scanned as one. A write-watch call costs ~3.5 µs however little it
/// covers, while one call over 2 GiB of committed pages takes ~1.3 ms (measured on Windows 11): a
/// retail game's intro video had its submitting thread at 42 % in GetWriteWatch with a 1 MiB gap
/// (thousands of calls before each submit over its mapped sparse tiles). The pages of a gap are
/// reported as written too; the uploads keep to what they track.
const WATCH_GAP: usize = 64 << 20;

/// The parts of `wanted` (start, end) inside `regions` (base, size), merged (gaps under
/// `WATCH_GAP` too) and sorted, as (base, size) – each within one region (a write-watch call
/// covers one allocation).
fn watch_ranges(regions: &[(usize, usize)], mut wanted: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    wanted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in wanted {
        match merged.last_mut() {
            Some(last) if a <= last.1.saturating_add(WATCH_GAP) => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    // each region against the merged ranges it meets (sorted and disjoint: a binary search, not
    // a pass over all of them – both lists run to thousands while a game streams)
    let mut out = Vec::new();
    for &(base, size) in regions {
        let end = base + size;
        let first = merged.partition_point(|&(_, b)| b <= base);
        for &(a, b) in merged[first..].iter().take_while(|&&(a, _)| a < end) {
            let (lo, hi) = (a.max(base), b.min(end));
            if lo < hi {
                out.push((lo, hi - lo));
            }
        }
    }
    out.sort_unstable();
    out
}

/// Copy `[src, src + len)` into `dst` (pre-zeroed) without faulting if another thread unmaps or
/// decommits a page concurrently: `ReadProcessMemory` catches the fault in kernel mode and returns
/// `0`. Because Windows may return `read == 0` when a fault occurs later in the span, binary-split
/// failed multi-page ranges at 4 KiB boundaries so all still-committed pages are copied in O(log N)
/// calls and only unreadable pages stay zeroed.
unsafe fn copy_committed_region(src: usize, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const PAGE: usize = 0x1000;

    unsafe fn copy_rec(proc: *mut c_void, src: usize, dst: *mut u8, len: usize) {
        if len == 0 {
            return;
        }
        let mut read = 0usize;
        if ReadProcessMemory(proc, src as *const c_void, dst as *mut c_void, len, &mut read) != 0 {
            return;
        }
        let read = read.min(len);
        let src = src + read;
        let dst = dst.add(read);
        let len = len - read;
        let page_end = (src & !(PAGE - 1)).saturating_add(PAGE).min(src + len);
        if page_end == src + len {
            return;
        }
        let mid = ((src + len / 2) & !(PAGE - 1)).max(page_end);
        let left = mid - src;
        copy_rec(proc, src, dst, left);
        copy_rec(proc, mid, dst.add(left), len - left);
    }

    copy_rec(GetCurrentProcess(), src, dst, len);
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped).
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut out = vec![0u8; len];
    let Some(limit) = at.checked_add(len) else { return out };
    // (the coherent window is committed as a whole: no `VirtualQuery`, which sizes a region by
    // walking it)
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(at, len)) {
        std::ptr::copy_nonoverlapping(at as *const u8, out.as_mut_ptr(), len);
        return out;
    }
    let mut p = at;
    while p < limit {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.RegionSize == 0 {
            break;
        }
        let end = (m.BaseAddress as usize).saturating_add(m.RegionSize).min(limit);
        if end <= p {
            break;
        }
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
        if readable {
            copy_committed_region(p, out.as_mut_ptr().add(p - at), end - p);
        }
        p = end;
    }
    out
}

/// Batches of at least this many bytes are detiled on several threads (`subresources_of_all`)
const PARALLEL_BYTES: usize = 1 << 20;

/// `subresources` of every texture of a batch (`bytes` of them), in order: on several threads
/// when the batch is big. They were read and detiled one after another in the title's own
/// submit – most of a retail game's render thread while it streamed textures the CPU writes.
fn subresources_of_all(picked: Vec<(Vec<u8>, Arc<xg::Texture>)>, bytes: usize) -> Vec<Vec<Subresource>> {
    let threads = if bytes < PARALLEL_BYTES { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get() / 2).clamp(1, 8) };
    subresources_on(picked, threads)
}

/// Detile each owned snapshot in order, on up to `threads` workers. A worker drops its
/// snapshot after that texture, so raw inputs do not live alongside the entire finished batch.
fn subresources_on(picked: Vec<(Vec<u8>, Arc<xg::Texture>)>, threads: usize) -> Vec<Vec<Subresource>> {
    let threads = threads.min(picked.len());
    if threads <= 1 {
        return picked.into_iter().map(|(mem, tex)| subresources_from(&mem, &tex)).collect();
    }
    // Take ownership of the next input under the short queue lock. No title-memory
    // reads happen here: the same bytes were already hashed in upload_all (#504).
    let mut out: Vec<Vec<Subresource>> = (0..picked.len()).map(|_| Vec::new()).collect();
    let next = Mutex::new(picked.into_iter().enumerate());
    let done: Vec<Vec<(usize, Vec<Subresource>)>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let item = next.lock().unwrap().next();
                        let Some((k, (mem, tex))) = item else { break };
                        mine.push((k, subresources_from(&mem, &tex)));
                    }
                    mine
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap_or_default()).collect()
    });
    for (k, subs) in done.into_iter().flatten() {
        out[k] = subs;
    }
    out
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    let mem = read_committed(va, tex.size() as usize);
    subresources_from(&mem, tex)
}

/// Detile subresources from an already-read memory snapshot (one read + one hash in `upload_all`).
fn subresources_from(mem: &[u8], tex: &xg::Texture) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    if tex.depth > 0 {
        return c
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| Subresource {
                depth: l.depth,
                rows: l.height,
                row_bytes: l.width * bpe,
                data: c.detile_volume(mem, i),
            })
            .collect();
    }
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        let start = s * c.slice_bytes;
        let end = (start + c.slice_bytes).min(mem.len());
        let slice = if start < mem.len() { &mem[start..end] } else { &[] };
        for (i, l) in c.levels.iter().enumerate() {
            out.push(Subresource {
                depth: 1,
                rows: l.height,
                row_bytes: l.width * bpe,
                data: c.detile_level(slice, i, s),
            });
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    /// #504: a hash check must not record success; only `remember_upload` after a good batch.
    #[test]
    fn last_content_is_not_poisoned_by_a_hash_check_alone() {
        let res = 0x7E57_0000_3000usize;
        let h = content_hash(&[1, 2, 3, 4]);
        assert!(!last_upload_matches(res, h));
        assert!(!last_upload_matches(res, h), "check alone must not insert");
        remember_upload(res, h);
        assert!(last_upload_matches(res, h));
        assert!(!last_upload_matches(res, h ^ 1));
        forget(res);
        assert!(!last_upload_matches(res, h));
    }

    /// A texture is left out only when its memory hashes as at its own last upload, and a
    /// texture that was forgotten (released, or its address registered again) is uploaded again.
    #[test]
    fn a_texture_with_unchanged_memory_is_left_out_until_it_is_forgotten() {
        let (a, b) = (0x7E57_0000_1000usize, 0x7E57_0000_2000usize);
        let mut mem = vec![0u8; 3 << 20];
        mem[5] = 1;
        let h = content_hash(&mem);
        assert!(!same_as_last_upload(a, h), "the first upload is not left out");
        assert!(same_as_last_upload(a, h), "the same bytes again: left out");
        assert!(!same_as_last_upload(b, h), "another texture over the same bytes has its own first upload");
        // one byte anywhere, also in the tail that is not a whole 32-byte step
        for at in [0usize, 31, 32, (3 << 20) - 1] {
            let mut m = mem.clone();
            m[at] ^= 0x80;
            assert_ne!(content_hash(&m), h, "byte {at} changed");
        }
        mem[(3 << 20) - 1] = 9;
        let h2 = content_hash(&mem);
        assert!(!same_as_last_upload(a, h2), "changed memory is uploaded");
        assert!(same_as_last_upload(a, h2));
        // the length is part of the content: a prefix of zeros is not the whole
        assert_ne!(content_hash(&[0u8; 64]), content_hash(&[0u8; 96]));
        forget(a);
        assert!(!same_as_last_upload(a, h2), "forgotten: its next upload is a first one");
        forget(a);
        forget(b);
    }

    /// Real uploads, dirty-page detection and a source changing after registration:
    /// bookkeeping alone cannot show that the bytes the GPU sees follow the mirror.
    #[test]
    fn gpu_mirror_upload_follows_cpu_writes_and_later_mapping_changes() {
        use crate::hle::kernelx;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};
        use windows_sys::Win32::System::Memory as mem;

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if super::super::backend() != super::super::Backend::Real {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        // Larger than all four UMA windows together. These public allocations
        // must use the watched Windows fallback, reserving space rather than RAM.
        const SIZE: usize = 24usize << 30;
        const OFFSET: usize = 0x2_0000;
        const PAGE: usize = 0x1_0000;
        struct Reservation(usize);
        impl Drop for Reservation {
            fn drop(&mut self) {
                unsafe { kernelx::VirtualFree(self.0 as *mut c_void, 0, mem::MEM_RELEASE); }
            }
        }
        struct RegisteredTexture(usize);
        impl Drop for RegisteredTexture {
            fn drop(&mut self) { forget(self.0); }
        }
        struct PhysicalPage { id: usize, at: usize }
        impl Drop for PhysicalPage {
            fn drop(&mut self) {
                unsafe {
                    if kernelx::physical_page_at(self.at) {
                        kernelx::MapTitlePhysicalPages(self.at as *mut c_void, 1, 0, 4, std::ptr::null_mut());
                    }
                    kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, &mut self.id);
                }
            }
        }
        unsafe fn reserve(protect: u32) -> Reservation {
            let at = kernelx::VirtualAlloc(std::ptr::null(), SIZE, 0x2040_2000, protect) as usize;
            assert_ne!(at, 0, "reserve-only fallback must succeed");
            let reservation = Reservation(at);
            assert!(!crate::gpu::uma::get().unwrap().contains(at), "exercise the Windows fallback");
            assert!(kernelx::in_graphics_reservation(at), "fallback must be write-watched");
            reservation
        }
        unsafe fn read_r8(dev: &ID3D12Device, queue: &ID3D12CommandQueue, texture: &ID3D12Resource) -> Vec<u8> {
            own_work(|| {
                let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let mut total = 0;
                dev.GetCopyableFootprints(&texture.GetDesc(), 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
                let mut readback = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                    D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback,
                ).unwrap();
                let readback: ID3D12Resource = readback.unwrap();
                let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                let barrier = |before, after| D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(texture), Subresource: 0, StateBefore: before, StateAfter: after }) },
                    ..Default::default()
                };
                list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(texture), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                crate::gpu::fence::wait(&fence, 1, dev, None, 5_000).unwrap();
                let mut data = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut data)).unwrap();
                let mut out = Vec::with_capacity(64 * 64);
                for y in 0..64 {
                    out.extend_from_slice(std::slice::from_raw_parts((data as *const u8).add(footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize), 64));
                }
                readback.Unmap(0, None);
                out
            })
        }
        unsafe {
            let raw_device = super::super::real::device_raw();
            let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let cpu = reserve(mem::PAGE_READWRITE);
            let gpu = reserve(0x801);
            let (src, va) = (cpu.0 + OFFSET, gpu.0 + OFFSET);
            assert_eq!(kernelx::mirrors::source(va), Some(src));
            assert_eq!(kernelx::VirtualAlloc(src as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, src);
            let console = console_desc(3, 64, 64, 1, 1, DXGI_FORMAT_R8_UNORM.0 as u32, 1);
            assert!(xg::Texture::new(&console).unwrap().size() <= PAGE as u64);
            let mut texture = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
                D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
            ).unwrap();
            let texture: ID3D12Resource = texture.unwrap();
            let key = texture.as_raw() as usize;
            register(raw_device, texture.as_raw(), va, &console, D3D12_RESOURCE_STATE_COMMON).unwrap();
            let _registered = RegisteredTexture(key);
            for value in [0x3D, 0x9A] {
                std::ptr::write_bytes(src as *mut u8, value, PAGE);
                refresh(queue.as_raw());
                assert_eq!(read_r8(dev, &queue, &texture), vec![value; 64 * 64], "GPU contents follow CPU writes");
                assert_eq!(address(key), Some(va), "the title address stays in the GPU view");
            }
            let (mut count, mut id) = (1usize, 0usize);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, &mut id), 1);
            let mut mapped = PhysicalPage { id, at: va };
            assert_eq!(kernelx::MapTitlePhysicalPages(va as *mut c_void, 1, 0, 4, &mut mapped.id) as usize, va);
            assert!(kernelx::physical_page_at(va));
            let mut info: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(mem::VirtualQuery(va as *const c_void, &mut info, std::mem::size_of_val(&info)), 0);
            assert_eq!(info.State, mem::MEM_RESERVE, "exercise the lazy mapping before its first touch");
            assert_eq!(mirror_source(va), va, "a mapped lazy page overrides the CPU mirror");
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0; 64 * 64], "the untouched physical page supplies zeros");
            std::ptr::write_bytes(va as *mut u8, 0xC7, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0xC7; 64 * 64]);
            assert_eq!(kernelx::MapTitlePhysicalPages(va as *mut c_void, 1, 0, 4, std::ptr::null_mut()) as usize, va);
            assert!(!kernelx::physical_page_at(va));
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![0x9A; 64 * 64], "unmapping restores the earlier CPU view");
            assert_eq!(address(key), Some(va));
            dev.GetDeviceRemovedReason().unwrap();
        }
    }

    /// A physical page mapped into a GPU view but not used yet (reserved until its first
    /// access, `kernelx::lazy`) is that address's own source, not the CPU view behind it.
    /// (Test by Kalmarv, review of PR #283.)
    #[test]
    fn a_lazily_mapped_page_in_a_gpu_view_is_its_own_source() {
        use crate::hle::kernelx;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const VIEW: usize = 16 << 20;
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let cpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            let gpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            kernelx::mirrors::reserved(cpu, VIEW, 0x2040_2000, 0x04);
            kernelx::mirrors::reserved(gpu, VIEW, 0x2040_2000, 0x801);
            assert_eq!(mirror_source(gpu + 0x2_0000), cpu + 0x2_0000); // nothing mapped: the CPU view
            let (mut n, mut pages) = (1usize, [0usize; 1]);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let at = gpu + 0x2_0000;
            // a texture placed there before the page is mapped reads the CPU view …
            let mut d = [0u8; 0x38];
            d[0..4].copy_from_slice(&3u32.to_le_bytes());
            d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
            d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
            d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
            d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
            let tex = Arc::new(xg::Texture::new(&d).unwrap());
            let mut e = Entry { device: 0, va: at, src: mirror_source(at), in_gpu_view: true, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: false, placed: false, unmapped: false, seq: 0 };
            assert_eq!(e.src, cpu + 0x2_0000);
            assert!(!recheck_source(&mut e), "nothing changed yet");
            assert_eq!(kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, at);
            // … and the page mapped under it later, uploaded again
            assert!(recheck_source(&mut e));
            assert_eq!(e.src, at);
            assert!(e.pending);
            PENDING.fetch_sub(1, Ordering::Relaxed);
            let before = mirror_source(at);
            (at as *mut u32).write_volatile(7); // its first use commits it
            let after = mirror_source(at);
            let mut none = [0usize; 1];
            kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, none.as_mut_ptr());
            kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr());
            kernelx::mirrors::released(cpu);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(gpu as *mut c_void, 0, MEM_RELEASE);
            assert_eq!(after, at);
            assert_eq!(before, at, "a page mapped at the address but not used yet is its source too");
        }
    }

    /// More written pages than one call reports: each is found once, in order, and the look
    /// resets them – a buffer that fills in the middle of the range and one that fills exactly.
    #[test]
    fn written_pages_past_a_full_buffer_are_all_found_once() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 1000;
        const PAGE: usize = 0x1000;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert!(base != 0);
            let mut buf = vec![std::ptr::null_mut::<c_void>(); 64];
            let (mut dirty, mut page) = (Vec::new(), 0u32);
            take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page);
            for written in [(0..PAGES).filter(|i| i % 7 != 3).collect::<Vec<_>>(), (0..128).map(|i| i * 5).collect()] {
                let want: Vec<usize> = written.iter().map(|i| base + i * PAGE).collect();
                for &a in &want {
                    (a as *mut u8).write_volatile(1);
                }
                dirty.clear();
                take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page);
                assert_eq!(page as usize, PAGE);
                assert_eq!(dirty, want, "{} pages written", want.len());
                dirty.clear();
                take_written(base, PAGES * PAGE, &mut buf, &mut dirty, &mut page);
                assert!(dirty.is_empty(), "the look reset what it reported");
            }
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    const SRV: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);
    const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// One state while every subresource has it, one each while they differ.
    #[test]
    fn transitions_per_subresource_and_back_to_one_state() {
        let mut s = vec![D3D12_RESOURCE_STATE_COMMON];
        apply_transition(&mut s, 4, ALL, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 2, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 9, D3D12_RESOURCE_STATE_COPY_DEST); // no subresource 9
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 2, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        apply_transition(&mut s, 4, ALL, D3D12_RESOURCE_STATE_COMMON);
        assert_eq!(s, vec![D3D12_RESOURCE_STATE_COMMON]);
        // a texture of one subresource: subresource 0 is all of it
        let mut one = vec![SRV];
        apply_transition(&mut one, 1, 0, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(one, vec![D3D12_RESOURCE_STATE_COPY_SOURCE]);
    }

    /// A texture kept in step: the title's transitions count once their list is submitted, in
    /// submission order, each time it is; a Reset forgets them; other resources are not noted.
    #[test]
    fn transitions_apply_when_their_list_is_submitted() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // RGBA8 64×64, 3 mips, 2 slices (6 subresources), 4KB_S
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&2u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&3u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = Arc::new(xg::Texture::new(&d).unwrap());
        assert_eq!(subresource_count(&tex), 6);
        // fake resource / list keys, never dereferenced: GPU-written, so no refresh uploads it
        let (res, other, l1, l2) = (0x7E57_0000_1000usize, 0x7E57_0000_2000usize, 0x7E57_0000_3000usize, 0x7E57_0000_4000usize);
        TEXTURES.lock().unwrap().insert(res, Entry { device: 0, va: 0x7E57_0000_0000, src: 0x7E57_0000_0000, in_gpu_view: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: true, placed: false, unmapped: false, seq: 0 });
        TEXTURES_LEN.fetch_add(1, Ordering::Relaxed);
        let states = || TEXTURES.lock().unwrap().by_res.get(&res).unwrap().states.clone();
        let (p1, p2) = (l1 as *mut c_void, l2 as *mut c_void);

        note_transitions(l1, &[(res, ALL, SRV), (other, ALL, D3D12_RESOURCE_STATE_RENDER_TARGET), (res, 4, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        assert_eq!(LIST_NOTES.lock().unwrap().get(&l1).map(|v| v.len()), Some(2), "only the texture kept in step is noted");
        assert_eq!(states(), vec![D3D12_RESOURCE_STATE_COMMON], "recorded, not run yet");
        note_transitions(l2, &[(res, 4, SRV), (res, 0, D3D12_RESOURCE_STATE_COPY_DEST)]);
        submitted(&[p1]);
        let mut want = vec![SRV; 6];
        want[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        assert_eq!(states(), want);
        submitted(&[p2]);
        want[4] = SRV;
        want[0] = D3D12_RESOURCE_STATE_COPY_DEST;
        assert_eq!(states(), want);
        // l1 runs again (a closed list may): its transitions again, after l2's
        submitted(&[p2, p1]);
        assert_eq!(states(), {
            let mut w = vec![SRV; 6];
            w[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
            w
        });
        list_reset(l1);
        list_reset(l2);
        submitted(&[p1, p2]);
        assert_eq!(states()[4], D3D12_RESOURCE_STATE_COPY_SOURCE, "a reset list has nothing to apply");
        forget(res);
        assert!(LIST_NOTES.lock().unwrap().get(&l1).is_none());
    }

    /// Only the wanted parts of the watched regions are scanned, merged, one region each.
    #[test]
    fn watch_ranges_clip_and_merge() {
        let regions = [(0x10000, 0x10000), (0x40000, 0x10000), (0x1000_0000, 0x1000_0000)];
        let wanted = vec![(0x12000, 0x13000), (0x11000, 0x12800), (0x1F000, 0x41000), (0x60000, 0x61000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x11000, 0xF000), (0x40000, 0x10000)]);
        // ranges a gap of under WATCH_GAP apart: one scan; further apart: two
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1008_0000, 0x1008_1000), (0x1100_0000, 0x1100_1000), (0x1800_0000, 0x1800_1000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x1000_0000, 0x100_1000), (0x1800_0000, 0x1000)]);
        // a merged range over two regions: a piece in each, whatever order the regions come in
        let regions = [(0x40000, 0x10000), (0x10000, 0x10000)];
        assert_eq!(watch_ranges(&regions, vec![(0x18000, 0x48000)]), vec![(0x18000, 0x8000), (0x40000, 0x8000)]);
        // a region between two merged ranges, and one past them all
        let regions = [(0x2000_0000, 0x10_0000), (0x9000_0000, 0x1000)];
        assert_eq!(watch_ranges(&regions, vec![(0x1000_0000, 0x1000_1000), (0x2008_0000, 0x2009_0000)]), vec![(0x2008_0000, 0x1_0000)]);
    }

    /// The textures of a batch detiled on several threads come back as on one: complete and in
    /// order (2D with mips and slices, 3D, BC7, a tiny one; more textures than threads).
    #[test]
    fn a_batch_detiled_on_several_threads_matches_one_thread() {
        let descs = [
            console_desc(3, 64, 64, 2, 3, 28, 0x105),
            console_desc(3, 200, 72, 1, 5, 28, 0x105),
            console_desc(4, 40, 24, 6, 3, 28, 0x105),
            console_desc(3, 256, 256, 1, 9, 98, 0x10A),
            console_desc(3, 4, 4, 1, 1, 28, 0x105),
            console_desc(3, 128, 32, 3, 2, 28, 0x105),
            console_desc(3, 64, 64, 1, 1, 28, 0x105),
        ];
        let textures: Vec<Arc<xg::Texture>> = descs.iter().map(|d| Arc::new(xg::Texture::new(d).unwrap())).collect();
        // each texture's console memory: distinct bytes per texture
        let mems: Vec<Vec<u8>> = textures.iter().enumerate().map(|(t, x)| (0..x.size() as usize).map(|i| (i * 7 + t * 61) as u8).collect()).collect();
        let picked = || mems.iter().zip(&textures).map(|(m, x)| (m.clone(), Arc::clone(x))).collect();
        let one = subresources_on(picked(), 1);
        assert!(one.iter().all(|s| !s.is_empty()), "a texture gave no subresources");
        for threads in [2, 3, 8] {
            assert!(subresources_on(picked(), threads) == one, "{threads} threads differ from one");
        }
    }

    /// The merge with #451 must not read the title a second time after checking its hash.
    #[test]
    fn parallel_detiling_uses_the_hashed_snapshot_after_source_changes() {
        let tex = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 3, 28, 0x105)).unwrap());
        let mut source: Vec<u8> = (0..tex.size() as usize).map(|i| (i * 7 + 61) as u8).collect();
        let snapshot = source.clone();
        let hash = content_hash(&snapshot);
        let expected = subresources_from(&snapshot, &tex);
        let picked = vec![(snapshot.clone(), Arc::clone(&tex)), (snapshot, Arc::clone(&tex))];
        source.fill(0);
        assert_ne!(content_hash(&source), hash, "the live source changed after its snapshot");
        let parallel = subresources_on(picked, 2);
        assert_eq!(parallel.len(), 2);
        for subs in parallel {
            assert!(subs == expected, "parallel detiling must use the bytes that were hashed");
        }
    }

    #[test]
    fn small_cpu_filled_target_accepts_wide_glyph_atlas_and_rejects_large_or_depth() {
        assert!(is_small_cpu_filled_target(2048, 128, 1, 0x300005), "2048x128 R8 glyph atlas must sync");
        assert!(is_small_cpu_filled_target(1024, 1024, 1, 0x1));
        assert!(is_small_cpu_filled_target(64, 64, 1, 0x5));
        assert!(!is_small_cpu_filled_target(1, 1, 1, 0x1), "1x1 placeholders skipped");
        assert!(!is_small_cpu_filled_target(2560, 1440, 1, 0x1), "full-screen RT skipped");
        assert!(!is_small_cpu_filled_target(2048, 1024, 1, 0x1), ">1M pixel RT skipped");
        assert!(!is_small_cpu_filled_target(256, 256, 1, 0x2), "depth buffers skipped");
        assert!(!is_small_cpu_filled_target(256, 256, 4, 0x1), "MSAA RTs skipped");
    }

    fn console_desc(dim: u32, w: u64, h: u32, array_or_depth: u16, mips: u16, format: u32, layout: u32) -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&dim.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&w.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&array_or_depth.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
        d
    }

    /// Overlapping textures shadow earlier ones so dead placements never re-upload over shared
    /// `vram::Pages` tiles; different-range overlaps request unmapping while exact same-range
    /// overlaps stay mapped; releasing a newer placement reactivates uncovered older ones.
    #[test]
    fn overlapping_placements_retire_superseded_entries() {
        let tex_16k = Arc::new(xg::Texture::new(&console_desc(3, 64, 64, 1, 1, 28, 0x105)).unwrap());
        let tex_64k = Arc::new(xg::Texture::new(&console_desc(3, 128, 128, 1, 1, 28, 0x105)).unwrap());
        assert_eq!((tex_16k.size(), tex_64k.size()), (0x4000, 0x1_0000));
        let entry = |va: usize, tex: &Arc<xg::Texture>, pending: bool, placed: bool| Entry {
            device: 0,
            va,
            src: va,
            in_gpu_view: false,
            tex: Arc::clone(tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending,
            gpu_written: false,
            placed,
            unmapped: false,
            seq: 0,
        };
        let mut tbl = Table::new();
        // Two adjacent placed 16 KiB textures + one placed 64 KiB texture after them.
        assert_eq!(tbl.insert(1, entry(0x10_0000, &tex_16k, true, true)), (1, vec![]));
        assert_eq!(tbl.insert(2, entry(0x10_4000, &tex_16k, true, true)), (1, vec![]));
        assert_eq!(tbl.insert(3, entry(0x11_0000, &tex_64k, false, true)), (0, vec![]));
        assert_eq!(tbl.by_res.len(), 3);
        // Exact same-range sibling over 2 shadows 2 without unmapping (tiles are shared).
        assert_eq!(tbl.insert(20, entry(0x10_4000, &tex_16k, false, true)), (-1, vec![]));
        assert!(!tbl.by_res.contains_key(&2) && tbl.shadowed.contains_key(&2));
        // A 64 KiB placed texture at 0x10_0000 supersedes 1, 20, and shadowed sibling 2 (all different range).
        let (delta, mut unmap) = tbl.insert(4, entry(0x10_0000, &tex_64k, true, true));
        unmap.sort_unstable();
        assert_eq!((delta, unmap), (0, vec![1, 2, 20]));
        assert!(tbl.by_res.contains_key(&3) && tbl.by_res.contains_key(&4));
        // Releasing 4 uncovers newest non-overlapping shadowed entries (1 and 20) and requests remap.
        let (removed, delta, mut remap) = tbl.remove(4);
        remap.sort_unstable();
        assert_eq!((removed, delta, remap), (true, 1, vec![1, 20]));
        assert!(tbl.by_res.contains_key(&1) && tbl.by_res.contains_key(&20) && tbl.shadowed.contains_key(&2));
        // Releasing 20 uncovers 2; releasing middle of a 3-chain still uncovers oldest when top goes.
        let (removed, _, remap) = tbl.remove(20);
        assert_eq!((removed, remap), (true, vec![2]));
        assert!(tbl.by_res.contains_key(&2));
        assert_eq!(tbl.insert(5, entry(0x10_8000, &tex_64k, true, true)).0, 1);
        assert_eq!(tbl.overlapping(0x10_0000, 0x1_0000), vec![(1, 0x10_0000), (2, 0x10_4000), (5, 0x10_8000)]);
        // Two committed resources with the exact same range coexist; a shifted one shadows them.
        assert_eq!(tbl.insert(10, entry(0x20_0000, &tex_16k, true, false)), (1, vec![]));
        assert_eq!(tbl.insert(11, entry(0x20_0000, &tex_16k, true, false)), (1, vec![]));
        assert!(tbl.by_res.contains_key(&10) && tbl.by_res.contains_key(&11));
        assert_eq!(tbl.insert(12, entry(0x20_0000, &tex_64k, true, false)), (-1, vec![]));
        assert!(!tbl.by_res.contains_key(&10) && !tbl.by_res.contains_key(&11) && tbl.by_res.contains_key(&12));
    }

    /// Real GPU (#248): placed title textures share `vram::Pages` so streaming unreleased
    /// overlapping placements stays bounded in video memory, superseded textures never corrupt
    /// newer placements on shared tiles, 4 KiB-aligned neighbours never false-alias, and mip
    /// chains (including packed mips), cubemap arrays (`extra_pages`) and 3D volumes upload intact.
    #[test]
    fn placed_title_textures_share_vram_and_preserve_pixels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no adapter");
                return;
            };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: software adapter");
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no D3D12 device");
                return;
            };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0
            {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: tiled tier < 2");
                return;
            }
            // An adapter that reports tier 2 but refuses a reserved cube with packed mips
            // (GitHub's runner) does not run the shared-page path this test is about.
            {
                let mut cube: D3D12_RESOURCE_DESC = std::ptr::read_unaligned(super::super::device_x::pc_resource_desc(&console_desc(3, 256, 256, 6, 9, 28, 0x105)).as_ptr() as *const _);
                cube.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
                let mut r: Option<ID3D12Resource> = None;
                if dev.CreateReservedResource(&cube, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_err() || r.is_none() {
                    println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: this adapter makes no reserved cube with mips");
                    return;
                }
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let Some(uma) = crate::gpu::uma::get() else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: no UMA");
                return;
            };
            let pool_size = 16usize << 20;
            let Some(base) = uma.alloc(pool_size, 0x1_0000).map(|p| p as usize) else {
                println!("SKIPPED placed_title_textures_share_vram_and_preserve_pixels: UMA alloc failed");
                return;
            };
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local_vram = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_tex = |va: usize, desc: &[u8; 0x38]| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), va as u64, desc.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "CreatePlacedResourceX failed: {hr:#x}");
                assert!(super::super::vram::is_placed_texture(p as usize), "expected shared vram::Pages texture at {va:#x}");
                ID3D12Resource::from_raw(p)
            };
            let write_tiled = |va: usize, desc: &[u8; 0x38], seed: u8| {
                let tex = xg::Texture::new(desc).unwrap();
                let s = tex.texels().unwrap();
                let bytes = std::slice::from_raw_parts_mut(va as *mut u8, tex.size() as usize);
                bytes.fill(0);
                if s.depth > 0 {
                    for (m, l) in s.chain.levels.iter().enumerate() {
                        let pat: Vec<u8> = (0..l.width * l.height * l.depth * s.bpe as usize).map(|k| seed.wrapping_add(m as u8).wrapping_add((k * 13) as u8)).collect();
                        s.chain.tile_volume(bytes, m, &pat);
                    }
                } else {
                    for slice in 0..s.array as usize {
                        let dst = &mut bytes[slice * s.chain.slice_bytes..(slice + 1) * s.chain.slice_bytes];
                        for (m, l) in s.chain.levels.iter().enumerate() {
                            let pat: Vec<u8> = (0..l.width * l.height * s.bpe as usize).map(|k| seed.wrapping_add((slice * 31 + m * 7) as u8).wrapping_add((k * 13) as u8)).collect();
                            s.chain.tile_level(dst, m, slice, &pat);
                        }
                    }
                }
            };
            let read_sub = |r: &ID3D12Resource, sub: u32| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, sub, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut rb,
                )
                .unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                own_work(|| {
                    super::super::vram::before_execute(&queue);
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let slice_pitch = fp.Footprint.RowPitch as usize * rows as usize;
                let mut out = Vec::with_capacity(fp.Footprint.Depth as usize * rows as usize * row_bytes as usize);
                for z in 0..fp.Footprint.Depth as usize {
                    for y in 0..rows as usize {
                        out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(z * slice_pitch + y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                    }
                }
                rb.Unmap(0, None);
                out
            };

            // 1. Streaming 512 unreleased 1 MiB textures (512 MiB placed) across 16 MiB stays bounded
            // with zero additional page growth after the 16 MiB working set is covered.
            let dev_key = dev.as_raw() as usize;
            let dev_pages = || super::super::vram::page_bytes_for(dev_key).0;
            let stream_desc = console_desc(3, 512, 512, 1, 1, 28, 0x109); // 64KB_S, 1 MiB
            let (pages0, vram0, kept0) = (dev_pages(), local_vram(), counts().0);
            let mut leaked_refs = Vec::with_capacity(512);
            for k in 0..16usize {
                leaked_refs.push(place_tex(base + k * (1 << 20), &stream_desc));
            }
            let (pages16, vram16) = (dev_pages(), local_vram());
            for k in 16..512usize {
                leaked_refs.push(place_tex(base + (k % 16) * (1 << 20), &stream_desc));
            }
            let (pages_delta, vram_delta, vram_after_fill, kept_delta) = (
                dev_pages() - pages0,
                local_vram().saturating_sub(vram0) as usize,
                local_vram().saturating_sub(vram16) as usize,
                counts().0 - kept0,
            );
            println!("512 unreleased 1 MiB textures over 16 MiB: +{pages_delta} B pages (after fill: +{} B), +{vram_delta} B local VRAM (after fill: +{vram_after_fill} B), +{kept_delta} active kept", dev_pages() - pages16);
            assert_eq!(pages_delta, 16 << 20, "pages must equal covered 16 MiB, not 512 MiB");
            assert_eq!(dev_pages(), pages16, "no pages added after working set is covered");
            // (the process's local video memory is not asserted: other GPU tests in the process
            // allocate at the same time - on an RX 9070 it read +672 MB under --test-threads=4 and
            // stayed within the bound alone; the page count above is what this change controls)
            let _ = vram_after_fill;
            assert_eq!(kept_delta, 16, "superseded placements must be retired from active tracking");

            // 2. Overlap integrity (rows 4, 5, 6) + unmap/remap on different-range overlap:
            // (a) Exact same range: old object sees new uploads (row 4); releasing new object
            // reactivates old object for subsequent CPU rewrites (row 6).
            let same_desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB
            let same_a = place_tex(base + 0x1000, &same_desc); // 4 KiB offset too!
            write_tiled(base + 0x1000, &same_desc, 0x1A);
            refresh(queue.as_raw());
            let same_b = place_tex(base + 0x1000, &same_desc);
            write_tiled(base + 0x1000, &same_desc, 0x2B);
            refresh(queue.as_raw());
            let want_2b = subresources(base + 0x1000, &xg::Texture::new(&same_desc).unwrap()).swap_remove(0).data;
            assert_eq!(read_sub(&same_b, 0), want_2b, "row 3: new placement readback");
            assert_eq!(read_sub(&same_a, 0), want_2b, "row 4: older exact-range alias shares tiles");
            drop(same_b);
            write_tiled(base + 0x1000, &same_desc, 0x3C);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&same_a, 0), subresources(base + 0x1000, &xg::Texture::new(&same_desc).unwrap())[0].data, "row 6: releasing newer placement reactivates older one");
            drop(same_a);

            // (b) Partial / different-range overlap (row 5 + unmap dead tail + remap on release):
            let old_desc = console_desc(3, 512, 512, 1, 1, 28, 0x109); // 1 MiB
            let new_desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB
            drop(leaked_refs);
            let old_res = place_tex(base, &old_desc);
            write_tiled(base, &old_desc, 0xAA);
            refresh(queue.as_raw());
            let original_pixels = read_sub(&old_res, 0);
            assert_eq!(original_pixels, subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data);
            // Place B at base + 256 KiB (overlaps old_res tail); writing into old_res's first
            // 256 KiB must NOT re-upload old_res over B (row 5).
            let new_res = place_tex(base + 0x4_0000, &new_desc);
            write_tiled(base + 0x4_0000, &new_desc, 0x55);
            refresh(queue.as_raw());
            let want_b = subresources(base + 0x4_0000, &xg::Texture::new(&new_desc).unwrap()).swap_remove(0).data;
            *(base as *mut u8) ^= 0xFF; // dirty page outside B, inside old_res
            refresh(queue.as_raw());
            assert_eq!(read_sub(&new_res, 0), want_b, "row 5: stale shadowed texture corrupted B");
            // Restore A's original CPU bytes while B still owns the overlapping GPU tiles.
            // Releasing B must repopulate A even when its CPU content matches its previous upload.
            write_tiled(base, &old_desc, 0xAA);
            drop(new_res);
            refresh(queue.as_raw());
            let restored = read_sub(&old_res, 0);
            assert_eq!(restored.len(), original_pixels.len());
            assert_eq!(restored.iter().zip(&original_pixels).position(|(a, b)| a != b), None,
                       "remapped texture must recover its last uploaded pixels");

            write_tiled(base, &old_desc, 0xD4);
            refresh(queue.as_raw());
            assert_eq!(read_sub(&old_res, 0), subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data, "reactivated unmapped texture remapped cleanly");
            drop(old_res);

            // 3. 4 KiB-aligned adjacent textures inside one 64 KiB page + straddling page boundary,
            // plus reclaiming dead extra_pages when overwritten by a larger placement without Release.
            let d16 = console_desc(3, 64, 64, 1, 1, 28, 0x105); // 4KB_S, 16 KiB
            let d64 = console_desc(3, 128, 128, 1, 1, 28, 0x105); // 4KB_S, 64 KiB
            let adj_specs = [(0x0000usize, d16, 0x11u8), (0x4000, d16, 0x22), (0x8000, d16, 0x33), (0xC000, d64, 0x44), (0x1_C000, d16, 0x55)];
            let adj: Vec<_> = adj_specs.iter().map(|&(off, d, _)| place_tex(base + off, &d)).collect();
            for &(off, d, seed) in &adj_specs {
                write_tiled(base + off, &d, seed);
            }
            refresh(queue.as_raw());
            for (r, &(off, d, _)) in adj.iter().zip(&adj_specs) {
                assert_eq!(read_sub(r, 0), subresources(base + off, &xg::Texture::new(&d).unwrap())[0].data, "4 KiB neighbour at +{off:#x} crosstalk");
            }
            let pages_before_reuse = dev_pages();
            // Binding sweeps pages idle for two seconds, so the device-wide count can fall
            // here even when this placement shares the extra tile it already has.
            let first = adj[1].as_raw() as usize;
            let before = super::super::vram::mapped_tiles_for(dev_key, first);
            let reused = place_tex(base + 0x4000, &d16);
            if super::super::vram::is_placed_texture(first) {
                let before = before.expect("the first unaligned placement is mapped");
                assert!(!before.1.is_empty(), "a 16 KiB texture in a 64 KiB page uses an extra tile");
                assert_eq!(
                    super::super::vram::mapped_tiles_for(dev_key, first).as_ref(),
                    Some(&before),
                    "the first placement keeps its tiles when the va is placed again",
                );
                assert_eq!(
                    super::super::vram::mapped_tiles_for(dev_key, reused.as_raw() as usize).as_ref().map(|t| &t.1),
                    Some(&before.1),
                    "reusing unaligned va must share extra_pages tile",
                );
            } else {
                assert_eq!(dev_pages(), pages_before_reuse, "reusing unaligned va must share extra_pages tile");
            }
            // Overwrite the first 64 KiB (covering +0x0000, +0x4000 (both adj[1] & reused), +0x8000, +0xC000)
            // WITHOUT releasing adj/reused: dead extra_pages (+0x4000, +0x8000) must drop immediately!
            let cover = place_tex(base, &d64);
            let pages_after_cover = dev_pages();
            assert!(pages_after_cover < pages_before_reuse, "superseded unaligned extra_pages must be reclaimed immediately ({pages_before_reuse} -> {pages_after_cover})");
            drop((cover, reused, adj));

            // 4. Full mip chain (standard + packed mips, RGBA8 & BC7), tiny all-packed mips,
            // 6-slice cubemap (exercises extra_pages reuse), and 3D volume texture (Tier 3).
            let shapes = [
                ("2D 512x512 RGBA8 10 mips", console_desc(3, 512, 512, 1, 10, 28, 0x109)),
                ("2D 1024x1024 BC7 11 mips", console_desc(3, 1024, 1024, 1, 11, 98, 0x109)),
                ("2D 64x64 RGBA8 all-packed 7 mips", console_desc(3, 64, 64, 1, 7, 28, 0x105)),
                ("2D 256x256x6 cubemap 9 mips", console_desc(3, 256, 256, 6, 9, 28, 0x105)),
                ("3D 64x64x64 4 mips", console_desc(4, 64, 64, 64, 4, 28, 0x105)),
            ];
            for (label, d) in shapes {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), base as u64, d.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "{label}: hr {hr:#x}");
                let res = ID3D12Resource::from_raw(p);
                let is_3d = u32::from_le_bytes(d[0..4].try_into().unwrap()) == 4;
                let array = u16::from_le_bytes(d[0x1C..0x1E].try_into().unwrap());
                let placed = super::super::vram::is_placed_texture(res.as_raw() as usize);
                if array > 1 && !is_3d {
                    // arrays and cubes with packed mips: a driver may refuse them reserved (GitHub's
                    // runner did with this cube); the texture is then committed – what the pixels
                    // below check either way
                    if !placed {
                        println!("{label}: not reserved on this adapter (committed instead)");
                    }
                } else if (!is_3d || opts.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_3.0)
                    && !(is_3d
                        && xg::Texture::new(&d)
                            .unwrap()
                            .texels()
                            .is_some_and(|s| s.first_tail < s.chain.levels.len()))
                {
                    assert!(placed, "{label} should use shared vram::Pages");
                } else if is_3d
                    && xg::Texture::new(&d)
                        .unwrap()
                        .texels()
                        .is_some_and(|s| s.first_tail < s.chain.levels.len())
                {
                    assert!(
                        !placed,
                        "{label} has an XG packed mip tail; use the committed detile/upload path"
                    );
                }
                write_tiled(base, &d, 0x71);
                refresh(queue.as_raw());
                let want = subresources(base, &xg::Texture::new(&d).unwrap());
                for (sub, exp) in want.iter().enumerate() {
                    assert_eq!(read_sub(&res, sub as u32), exp.data, "{label}: subresource {sub} mismatch");
                }
                // Placing a second cubemap at the same va must reuse every tile including extra_pages.
                if label.contains("cubemap") {
                    // Binding sweeps unrelated targets idle for two seconds, so a device-wide
                    // byte count can fall here even when every tile of this cube is shared.
                    let mappings = super::super::vram::tile_mappings_for(dev.as_raw() as usize, res.as_raw() as usize).expect("cube is mapped");
                    assert!(!mappings.0.is_empty() || !mappings.1.is_empty(), "cube must have mapped tiles");
                    let res2 = place_tex(base, &d);
                    assert_eq!(super::super::vram::tile_mappings_for(dev.as_raw() as usize, res2.as_raw() as usize).as_ref(), Some(&mappings),
                        "the second cube must reuse every title-page and extra-page tile");
                    assert_eq!(super::super::vram::tile_mappings_for(dev.as_raw() as usize, res.as_raw() as usize).as_ref(), Some(&mappings),
                        "the original cube must keep the same tile mappings");
                    println!("{label}: reused {} title-page tiles and {} extra-page tiles", mappings.0.len(), mappings.1.len());
                    drop(res2);
                }
                drop(res);
            }
            uma.free(base);
        }
    }

    #[test]
    fn range_is_fully_watched_handles_gaps_and_adjacent_spans() {
        let watched = [(0x10_0000usize, 0x1_0000usize), (0x11_0000, 0x2_0000), (0x20_0000, 0x1_0000)];
        assert!(range_is_fully_watched(0x10_0000, 0x8000, &watched));
        // Spanning two adjacent watched ranges is fully watched.
        assert!(range_is_fully_watched(0x10_F000, 0x4000, &watched));
        assert!(range_is_fully_watched(0x10_0000, 0x3_0000, &watched));
        // Crossing an unwatched gap or starting outside watched ranges is not fully watched.
        assert!(!range_is_fully_watched(0x12_F000, 0x2000, &watched));
        assert!(!range_is_fully_watched(0x13_0000, 0x1000, &watched));
        assert!(!range_is_fully_watched(0x0F_F000, 0x2000, &watched));
        assert!(!range_is_fully_watched(0x30_0000, 0x4000, &watched));
        assert!(!range_is_fully_watched(0x10_0000, 0x1000, &[]));
    }

    #[test]
    fn sample_fingerprint_detects_start_mid_and_end_changes_in_large_textures() {
        let mut small = vec![0u8; 32 * 1024];
        let h0 = unsafe { sample_fingerprint(small.as_ptr() as usize, small.len()) };
        small[12345] = 0x5A;
        let h1 = unsafe { sample_fingerprint(small.as_ptr() as usize, small.len()) };
        assert_ne!(h0, h1, "small texture mutation must change fingerprint");

        let mut large = vec![0u8; 256 * 1024];
        let base = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        // Start 16 KiB slice
        large[1024] = 0x11;
        let hs = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        assert_ne!(base, hs, "start-slice write must change fingerprint");
        // Middle 16 KiB slice
        large[128 * 1024] = 0x22;
        let hm = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        assert_ne!(hs, hm, "mid-slice write must change fingerprint");
        // End 16 KiB slice
        let last = large.len() - 1;
        large[last] = 0x33;
        let he = unsafe { sample_fingerprint(large.as_ptr() as usize, large.len()) };
        assert_ne!(hm, he, "end-slice write must change fingerprint");
    }

    #[test]
    fn stream_recheck_schedule_catches_late_writes_and_resets_milestones() {
        let desc = console_desc(3, 256, 256, 1, 1, 28, 0x109); // 256 KiB RGBA8
        let tex = Arc::new(xg::Texture::new(&desc).expect("layout"));
        let size = tex.size() as usize;
        let mut backing = vec![0u8; size];
        let va = backing.as_ptr() as usize;
        let res_id = 0x7410_0001usize;
        let gpu_res_id = 0x7410_0002usize;

        let e_stream = Entry {
            device: 0,
            va,
            src: va,
            in_gpu_view: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: false,
            placed: false,
            unmapped: false,
            seq: 0,
        };
        let e_gpu = Entry {
            device: 0,
            va: va + size,
            src: va,
            in_gpu_view: false,
            tex: Arc::clone(&tex),
            states: vec![D3D12_RESOURCE_STATE_COMMON],
            pending: false,
            gpu_written: true,
            placed: false,
            unmapped: false,
            seq: 0,
        };
        {
            let mut t = TEXTURES.lock().unwrap();
            let (d1, _) = t.insert(res_id, e_stream);
            let (d2, _) = t.insert(gpu_res_id, e_gpu);
            adjust_pending(d1 + d2);
        }
        let fp0 = unsafe { sample_fingerprint(va, size) };
        {
            let mut g = STREAM_RECHECK.lock().unwrap();
            g.insert(res_id, StreamRecheck { base_frame: 100, stage: 0, last_fp: fp0 });
            g.insert(gpu_res_id, StreamRecheck { base_frame: 100, stage: 0, last_fp: fp0 });
        }

        // Before first milestone (+4 frames -> frame 104), nothing is checked or re-queued.
        backing[size - 16] = 0xAB;
        unsafe { recheck_streamed_unwatched_at(&[], 103) };
        assert!(!TEXTURES.lock().unwrap().by_res.get(&res_id).unwrap().pending);

        // At milestone 0 (frame 104), late write is detected: pending becomes true, schedule resets
        // to base_frame=104, stage=0, and the gpu_written entry is pruned without re-queuing.
        unsafe { recheck_streamed_unwatched_at(&[], 104) };
        {
            let mut t = TEXTURES.lock().unwrap();
            let e = t.by_res.get_mut(&res_id).unwrap();
            assert!(e.pending, "late write to unwatched texture must re-queue upload at milestone 0");
            e.pending = false;
            PENDING.fetch_sub(1, Ordering::Relaxed);
            assert!(!t.by_res.get(&gpu_res_id).unwrap().pending, "gpu_written texture must not be re-queued");
        }
        {
            let g = STREAM_RECHECK.lock().unwrap();
            assert!(!g.contains_key(&gpu_res_id), "gpu_written texture must be pruned from STREAM_RECHECK");
            let st = g.get(&res_id).copied().expect("schedule reset after change");
            assert_eq!((st.base_frame, st.stage), (104, 0));
            assert_ne!(st.last_fp, fp0);
        }

        // Advance through all 5 milestones with no further writes: entry must not re-queue and must
        // clean itself up after milestone 4 (+400 frames -> frame 504).
        for f in [108, 116, 144, 224, 504] {
            unsafe { recheck_streamed_unwatched_at(&[], f) };
            assert!(!TEXTURES.lock().unwrap().by_res.get(&res_id).unwrap().pending, "unchanged texture must not re-queue at frame {f}");
        }
        assert!(!STREAM_RECHECK.lock().unwrap().contains_key(&res_id), "entry must retire after final 400-frame milestone");

        forget(res_id);
        forget(gpu_res_id);
    }

    /// Real GPU, `vram::level_pages` (`XCL_TEXTURE_LEVEL_PAGES=1`): a sampled texture placed on the
    /// shared pages gets the pages of a level when an upload has texels for it. Its memory has
    /// every level but the largest at first, as a title streams a texture: no page before the
    /// first upload, then pages for those levels only and their texels right, zeros where there
    /// are none. With memory under the largest level too, a texture placed there again (as a
    /// title does) takes that level's pages - the others are shared - and holds all of it. A
    /// texture the GPU writes gets the pages of all its levels.
    #[test]
    fn a_placed_texture_gets_the_pages_of_a_level_with_its_texels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const NAME: &str = "a_placed_texture_gets_the_pages_of_a_level_with_its_texels";
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return println!("SKIPPED {NAME}: no adapter") };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return println!("SKIPPED {NAME}: software adapter");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev else { return println!("SKIPPED {NAME}: no D3D12 device") };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0
            {
                return println!("SKIPPED {NAME}: tiled tier < 2");
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            // 1024x1024 RGBA8, 11 levels, 64 KiB blocks: 5.4 MiB on the console
            let d = console_desc(3, 1024, 1024, 1, 11, 28, 0x109);
            let place_tex = |va: usize| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::device_x::create_placed_resource_x(dev.as_raw(), va as u64, d.as_ptr(), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr >= 0 && !p.is_null(), "CreatePlacedResourceX failed: {hr:#x}");
                ID3D12Resource::from_raw(p)
            };
            let read_sub = |r: &ID3D12Resource, sub: u32| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, sub, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                own_work(|| {
                    super::super::vram::before_execute(&queue);
                    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    texture::wait(&dev, &queue).unwrap();
                });
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
                for y in 0..rows as usize {
                    out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                }
                rb.Unmap(0, None);
                out
            };
            let tex = xg::Texture::new(&d).unwrap();
            let surface = tex.texels().unwrap();
            let (size, chain) = (tex.size() as usize, &surface.chain);
            // the whole texture as the title will have it (no byte of a level zero), and what its
            // levels then hold
            let mut whole = vec![0u8; size];
            for (m, l) in chain.levels.iter().enumerate() {
                let pat: Vec<u8> = (0..l.width * l.height * surface.bpe as usize).map(|k| 1 + ((m * 7 + k * 13) % 251) as u8).collect();
                chain.tile_level(&mut whole[..chain.slice_bytes], m, 0, &pat);
            }
            let want = subresources(whole.as_ptr() as usize, &tex);
            assert_eq!(want.len(), 11);
            // its place in the title's memory: everything below mip 0 is there (the levels follow
            // each other from the smallest to mip 0, `tiling::Chain`)
            let at = VirtualAlloc(std::ptr::null(), size.next_multiple_of(0x1_0000), MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(at, 0);
            let top = chain.levels[0].offset;
            assert!(top > 0 && top % 0x1000 == 0 && top < size, "mip 0 is the last part of the slice: from {top:#x} of {size:#x}");
            assert!(!VirtualAlloc(at as *const c_void, top, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(whole.as_ptr(), at as *mut u8, top);

            super::super::vram::LEVEL_PAGES_FOR_TEST.store(true, Ordering::Relaxed);
            let dev_key = dev.as_raw() as usize;
            let dev_pages = || super::super::vram::page_bytes_for(dev_key).0;
            let p0 = dev_pages();
            let small = place_tex(at);
            super::super::vram::LEVEL_PAGES_FOR_TEST.store(false, Ordering::Relaxed);
            if !super::super::vram::has_lazy_levels(small.as_raw() as usize) {
                // (not reserved, or a tiling that does not add up: it has its pages as before)
                drop(small);
                VirtualFree(at as *mut c_void, 0, MEM_RELEASE);
                return println!("SKIPPED {NAME}: this adapter's texture was not placed with lazy levels");
            }
            super::super::vram::LEVEL_PAGES_FOR_TEST.store(true, Ordering::Relaxed);
            assert_eq!(dev_pages(), p0, "no pages before the first upload");
            // the tiles of mip 0 and of all of it, as this driver lays them out
            let (mut all, mut n, mut mip0) = (0u32, 1u32, D3D12_SUBRESOURCE_TILING::default());
            dev.GetResourceTiling(&small, Some(&mut all), None, None, Some(&mut n), 0, &mut mip0);
            let top_tiles = mip0.WidthInTiles * mip0.HeightInTiles as u32 * mip0.DepthInTiles as u32;
            assert!(n == 1 && top_tiles > 0 && top_tiles < all, "mip 0 has tiles of its own: {top_tiles} of {all}");
            refresh(queue.as_raw());
            assert_eq!(dev_pages() - p0, ((all - top_tiles) as usize) << 16, "pages for the levels with texels, none for mip 0");
            for sub in 1..11u32 {
                assert!(read_sub(&small, sub) == want[sub as usize].data, "level {sub} after the first upload");
            }
            assert!(read_sub(&small, 0).iter().all(|&b| b == 0), "a level without pages reads as zeros");
            // the title maps memory under mip 0 and fills it
            assert!(!VirtualAlloc((at + top) as *const c_void, size - top, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(whole.as_ptr().add(top), (at + top) as *mut u8, size - top);
            let full = place_tex(at);
            refresh(queue.as_raw());
            assert_eq!(dev_pages() - p0, (all as usize) << 16, "mip 0's pages come with its texels; the other levels' are shared");
            for sub in 0..11u32 {
                assert!(read_sub(&full, sub) == want[sub as usize].data, "level {sub} once all of it is there");
            }
            // what the GPU writes needs pages under all of the texture: the first one's mip 0 gets
            // the pages of its address, which hold what the second one's upload put there
            note_gpu_write(small.as_raw() as usize);
            assert_eq!(dev_pages() - p0, (all as usize) << 16, "the same pages");
            assert!(read_sub(&small, 0) == want[0].data, "a GPU-written texture has pages under all its levels");
            super::super::vram::LEVEL_PAGES_FOR_TEST.store(false, Ordering::Relaxed);
            drop((small, full));
            VirtualFree(at as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `read_committed` copies committed pages and leaves uncommitted gaps zeroed, including
    /// unaligned start/end offsets.
    #[test]
    fn read_committed_copies_committed_pages_and_zeroes_gaps() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            assert_ne!(VirtualAlloc((base + 2 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            std::ptr::write_bytes(base as *mut u8, 0x3A, PAGE);
            std::ptr::write_bytes((base + 2 * PAGE) as *mut u8, 0x7C, PAGE);

            let start = base + 0x0B00;
            let len = 2 * PAGE + 0x0500;
            let got = read_committed(start, len);
            assert_eq!(got.len(), len);
            assert!(got[..PAGE - 0x0B00].iter().all(|&b| b == 0x3A));
            assert!(got[PAGE - 0x0B00..2 * PAGE - 0x0B00].iter().all(|&b| b == 0));
            assert!(got[2 * PAGE - 0x0B00..].iter().all(|&b| b == 0x7C));

            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `read_committed` must not fault with 0xC0000005 when another thread concurrently
    /// decommits/recommits pages between `VirtualQuery` and the copy, and surrounding committed
    /// pages must remain intact.
    #[test]
    fn read_committed_tolerates_concurrent_decommit() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGE: usize = 0x10000;
        const PAGES: usize = 16;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_ne!(VirtualAlloc(base as *const c_void, PAGES * PAGE, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
            // Keep first and last 64 KiB pages permanently committed with known sentinels; race the middle.
            std::ptr::write_bytes(base as *mut u8, 0x5A, PAGE);
            std::ptr::write_bytes((base + (PAGES - 1) * PAGE) as *mut u8, 0xA5, PAGE);

            let stop = Arc::new(AtomicBool::new(false));
            let stop_w = Arc::clone(&stop);
            let mutator = std::thread::spawn(move || {
                let mut i = 0usize;
                while !stop_w.load(Ordering::Relaxed) {
                    let slot = 1 + (i % (PAGES - 2));
                    let p = (base + slot * PAGE) as *const c_void;
                    let _ = VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT);
                    let q = VirtualAlloc(p, PAGE, MEM_COMMIT, PAGE_READWRITE);
                    if !q.is_null() {
                        std::ptr::write_bytes(q as *mut u8, 0x3C, 4096);
                    }
                    i = i.wrapping_add(1);
                }
            });

            // Unaligned start/end across the racing region (mirrors the streaming pool layout).
            let off = 0x0B00;
            let len = (PAGES - 1) * PAGE + 0x0500;
            for _ in 0..400 {
                let out = read_committed(base + off, len);
                assert_eq!(out.len(), len);
                assert!(out[..PAGE - off].iter().all(|&b| b == 0x5A));
                assert!(out[len - (off + 0x0500)..].iter().all(|&b| b == 0xA5));
            }

            stop.store(true, Ordering::Relaxed);
            mutator.join().unwrap();
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn batch_fits_caps_items_bytes_and_device() {
        // Empty batch always accepts its first upload, even when larger than BATCH_BYTES.
        assert!(batch_fits(0, 0, 0, 1, BATCH_BYTES + 4096));
        // Once an oversized texture is in the batch, the next upload starts a new batch.
        assert!(!batch_fits(1, BATCH_BYTES + 4096, 1, 1, 4096));

        // Many tiny textures split at BATCH_ITEMS even when well below BATCH_BYTES.
        assert!(batch_fits(BATCH_ITEMS - 1, (BATCH_ITEMS - 1) * 4096, 1, 1, 4096));
        assert!(!batch_fits(BATCH_ITEMS, BATCH_ITEMS * 4096, 1, 1, 4096));

        // Byte cap accepts exact BATCH_BYTES, splits past it, and does not overflow.
        assert!(batch_fits(1, BATCH_BYTES - 4096, 1, 1, 4096));
        assert!(!batch_fits(1, BATCH_BYTES - 4096, 1, 1, 4097));
        assert!(!batch_fits(1, usize::MAX - 10, 1, 1, 100));

        // Different device always starts a new batch.
        assert!(!batch_fits(1, 4096, 1, 2, 4096));
    }
}
