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

/// A texture the title placed at `va` (its console desc `desc`): kept in step with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, desc: &[u8; 0x38], state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    // (a new texture object, or the address of an old one again: its first upload is not left out)
    LAST_CONTENT.lock().unwrap().remove(&(res as usize));
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
        // (64x64 and up: 1x1 placeholders are not worth an entry; 0x2: a depth buffer)
        !off && w <= 1024 && h <= 1024 && w * h as u64 >= 64 * 64 && samples <= 1 && flags & 0x2 == 0
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
    let mut t = TEXTURES.lock().unwrap();
    if let Some(e) = t.get_mut(res) {
        if !e.gpu_written {
            e.gpu_written = true;
            GPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
        }
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
    let (retired, to_unmap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (delta, retired, unmap_ptrs) = t.retire_range(va, va.saturating_add(len));
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        (retired, clone_resources(&unmap_ptrs))
    };
    for r in &to_unmap {
        super::vram::unmap_superseded(r);
    }
    if retired > 0 {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    LAST_CONTENT.lock().unwrap().remove(&res);
    let (removed, to_remap) = {
        let mut t = TEXTURES.lock().unwrap();
        let (removed, delta, remap_ptrs) = t.remove(res);
        adjust_pending(delta);
        TEXTURES_LEN.store(t.by_res.len(), Ordering::Relaxed);
        (removed, unsafe { clone_resources(&remap_ptrs) })
    };
    for r in &to_remap {
        unsafe {
            super::vram::remap_if_unmapped(r);
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
                loop {
                    let mut n = buf.len();
                    if GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, size, buf.as_mut_ptr(), &mut n, &mut page) != 0 {
                        break;
                    }
                    dirty.extend(buf[..n].iter().map(|&p| p as usize));
                    if n < buf.len() {
                        break;
                    }
                }
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

/// Whether texture `res` was last uploaded from memory with hash `h`; `h` is remembered as its
/// content from here on (the caller uploads it when this is false).
fn same_as_last_upload(res: usize, h: u64) -> bool {
    LAST_CONTENT.lock().unwrap().insert(res, h) == Some(h)
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

/// staging bytes of one batch at most
const BATCH_BYTES: usize = 256 << 20;

/// The uploads `todo`, in batches of up to `BATCH_BYTES` (one list and one wait each).
unsafe fn upload_all(todo: Vec<(ID3D12Resource, Upload)>) {
    static N: AtomicU64 = AtomicU64::new(0);
    let warn = |e: &str| {
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture: upload failed: {e}");
        }
    };
    let mut todo = todo.into_iter().peekable();
    while todo.peek().is_some() {
        let mut batch = Vec::new();
        let mut bytes = 0usize;
        let mut device = 0usize;
        while let Some((_, (dev, _, tex, _))) = todo.peek() {
            if !batch.is_empty() && (bytes + tex.size() as usize > BATCH_BYTES || *dev != device) {
                break;
            }
            let (r, (dev, va, tex, states)) = todo.next().unwrap();
            // The same bytes as at this texture's last upload: nothing to do. The write watch is
            // per page and a title writes next to its textures (and rewrites what it wrote): a
            // retail title's video planes and UI targets went up again for every submit, ~25 MB
            // a presented frame where ~3 MB had changed, and the submits waited here.
            if unchanged_skip() {
                let h = content_hash(&read_committed(va, tex.size() as usize));
                if same_as_last_upload(r.as_raw() as usize, h) {
                    let n = UNCHANGED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n == 1 || (n.is_power_of_two() && n >= 1024) {
                        crate::xlog_info!("title textures: {n} uploads left out so far – the texture's memory held what it held at its last upload");
                    }
                    continue;
                }
            }
            device = dev;
            bytes += tex.size() as usize;
            let subs = subresources(va, &tex);
            if super::present::trace_frame() {
                note_content(va, &r, &subs);
            }
            batch.push((r, subs, states));
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
        match own_work(|| {
            super::vram::before_execute(&queue);
            texture::upload_batch(dev, &queue, &batch)
        }) {
            Ok(failed) => {
                for (_, e) in failed {
                    warn(&e);
                }
                UPLOADS.fetch_add(batch.len() as u64, Ordering::Relaxed);
            }
            Err(e) => warn(&e),
        }
    }
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

/// Ranges closer than this are scanned as one (a gap costs little to scan; a call per texture –
/// thousands of them – a lot)
const WATCH_GAP: usize = 1 << 20;

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
    let mut out = Vec::new();
    for (a, b) in merged {
        for &(base, size) in regions {
            let (lo, hi) = (a.max(base), b.min(base + size));
            if lo < hi {
                out.push((lo, hi - lo));
            }
        }
    }
    out.sort_unstable();
    out
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped).
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut out = vec![0u8; len];
    // (the coherent window is committed as a whole: no `VirtualQuery`, which sizes a region by
    // walking it)
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(at, len)) {
        std::ptr::copy_nonoverlapping(at as *const u8, out.as_mut_ptr(), len);
        return out;
    }
    let mut p = at;
    while p < at + len {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let end = (m.BaseAddress as usize + m.RegionSize).min(at + len);
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
        if readable {
            std::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr().add(p - at), end - p);
        }
        p = end;
    }
    out
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    if tex.depth > 0 {
        let mem = read_committed(va, tex.size() as usize);
        return c.levels.iter().enumerate().map(|(i, l)| Subresource { depth: l.depth, rows: l.height, row_bytes: l.width * bpe, data: c.detile_volume(&mem, i) }).collect();
    }
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        let slice = read_committed(va + s * c.slice_bytes, c.slice_bytes);
        for (i, l) in c.levels.iter().enumerate() {
            out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * bpe, data: c.detile_level(&slice, i, s) });
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

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

        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
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
        let _guard = TEST_LOCK.lock().unwrap();
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
        // ranges a gap of under 1 MiB apart: one scan; further apart: two
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1008_0000, 0x1008_1000), (0x1100_0000, 0x1100_1000)];
        assert_eq!(watch_ranges(&regions, wanted), vec![(0x1000_0000, 0x8_1000), (0x1100_0000, 0x1000)]);
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
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap();
        let _guard = TEST_LOCK.lock().unwrap();
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
            assert_eq!(read_sub(&old_res, 0), subresources(base, &xg::Texture::new(&old_desc).unwrap())[0].data);
            // Place B at base + 256 KiB (overlaps old_res tail); writing into old_res's first
            // 256 KiB must NOT re-upload old_res over B (row 5).
            let new_res = place_tex(base + 0x4_0000, &new_desc);
            write_tiled(base + 0x4_0000, &new_desc, 0x55);
            refresh(queue.as_raw());
            let want_b = subresources(base + 0x4_0000, &xg::Texture::new(&new_desc).unwrap()).swap_remove(0).data;
            *(base as *mut u8) ^= 0xFF; // dirty page outside B, inside old_res
            refresh(queue.as_raw());
            assert_eq!(read_sub(&new_res, 0), want_b, "row 5: stale shadowed texture corrupted B");
            // Releasing B remaps previously-unmapped old_res cleanly and restores its pixels.
            drop(new_res);
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
            let reused = place_tex(base + 0x4000, &d16);
            assert_eq!(dev_pages(), pages_before_reuse, "reusing unaligned va must share extra_pages tile");
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
                } else if !is_3d || opts.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_3.0 {
                    assert!(placed, "{label} should use shared vram::Pages");
                }
                write_tiled(base, &d, 0x71);
                refresh(queue.as_raw());
                let want = subresources(base, &xg::Texture::new(&d).unwrap());
                for (sub, exp) in want.iter().enumerate() {
                    assert_eq!(read_sub(&res, sub as u32), exp.data, "{label}: subresource {sub} mismatch");
                }
                // Placing a second cubemap at the same va must reuse every tile including extra_pages.
                if label.contains("cubemap") {
                    let p0 = dev_pages();
                    let res2 = place_tex(base, &d);
                    assert_eq!(dev_pages(), p0, "cubemap extra_pages must be reused at same va");
                    drop(res2);
                }
                drop(res);
            }
            uma.free(base);
        }
    }
}
