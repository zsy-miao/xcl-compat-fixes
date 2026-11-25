//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

#[cfg(test)]
#[path = "vram_gpu_tests.rs"]
mod gpu_regression_tests;

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    dev: usize,
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    heap: H,
    /// resources placed in it
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Return expired heaps for `dev` (preserving insertion order) to retire/drop outside `ARENAS.lock()`.
    fn sweep_dropped_for(&mut self, dev: usize, now: std::time::Instant) -> Vec<H> {
        let is_expired = |a: &Arena<H>| a.dev == dev && a.users == 0 && a.idle_since.is_some_and(|t| now.duration_since(t) >= IDLE);
        if !self.list.iter().any(is_expired) {
            return Vec::new();
        }
        let mut dropped = Vec::new();
        let mut kept = Vec::with_capacity(self.list.len());
        for a in self.list.drain(..) {
            if is_expired(&a) {
                dropped.push(a.heap);
            } else {
                kept.push(a);
            }
        }
        self.list = kept;
        dropped
    }

    /// Place `bytes` at `va` (`align`) on `dev`; returns `(placement, expired_heaps)` without dropping inside `ARENAS.lock()`.
    fn take_for(&mut self, dev: usize, va: usize, bytes: usize, align: usize, now: std::time::Instant, make: impl FnOnce(usize, usize) -> Option<H>) -> (Option<(H, usize, u64)>, Vec<H>) {
        let expired = self.sweep_dropped_for(dev, now);
        let fits = |a: &Arena<H>| a.dev == dev && a.align >= align && a.base <= va && va + bytes <= a.base + a.size && (va - a.base) % align == 0;
        let i = match self.list.iter().rposition(fits) {
            Some(i) => i,
            None => {
                let size = bytes.div_ceil(align) * align;
                let Some(heap) = make(size, align) else { return (None, expired) };
                self.next_id += 1;
                self.list.push(Arena { dev, base: va, size, align, heap, users: 0, idle_since: None, id: self.next_id });
                self.list.len() - 1
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        (Some((a.heap.clone(), va - a.base, a.id)), expired)
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
/// resource → its arena's id
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        if let Some(id) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    if super::real::dred_on() {
        if let Some(ref h) = heap {
            super::real::name_heap(h.as_raw(), &format!("xcl vram arena va={va:#x}"));
        }
    }
    heap
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

struct Target {
    va: usize,
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
    /// its levels get their pages one by one (`map_levels`); None: all of it has pages
    levels: Option<Levels>,
}

/// The tiles of a texture whose levels get video memory when an upload first has texels for
/// them (`level_pages`): a group for each level with tiles of its own, in order, then one for
/// the packed levels.
#[derive(Clone, Debug, PartialEq)]
struct Levels {
    /// (the group's first subresource, its first tile in the resource, its tiles, mapped)
    groups: Vec<(u32, u32, u32, bool)>,
    /// the texture's levels (the last group's go up to here)
    mips: u32,
}

impl Levels {
    /// From the resource's tiling: `standard` (first tile, tiles) of each level with tiles of
    /// its own, `packed` the same of the packed levels, `total` tiles in all. None: they do not
    /// add up to the resource (it then gets all its pages at once).
    fn new(standard: &[(u32, u32)], packed: Option<(u32, u32)>, total: u32, mips: u32) -> Option<Levels> {
        let mut groups: Vec<(u32, u32, u32, bool)> = standard.iter().enumerate().map(|(i, &(first, n))| (i as u32, first, n, false)).collect();
        if let Some((first, n)) = packed {
            groups.push((standard.len() as u32, first, n, false));
        }
        let mut next = 0u32;
        let mut by_tile = groups.clone();
        by_tile.sort_by_key(|g| g.1);
        for g in &by_tile {
            if g.2 == 0 || g.1 != next {
                return None;
            }
            next = g.1.checked_add(g.2)?;
        }
        (next == total && !groups.is_empty()).then_some(Levels { groups, mips })
    }

    /// The levels of group `k`.
    fn levels_of(&self, k: usize) -> std::ops::Range<u32> {
        self.groups[k].0..self.groups.get(k + 1).map_or(self.mips, |g| g.0)
    }

    /// The group level `level` is in (the packed levels share the last one).
    fn group_of(&self, level: u32) -> usize {
        self.groups.partition_point(|g| g.0 <= level).saturating_sub(1)
    }
}

/// `XCL_TEXTURE_LEVEL_PAGES=1` (off by default): a sampled 2D texture of one slice on the
/// shared pages gets the pages of a level when an upload first has texels for it. A retail game
/// places its streamed textures at full size and maps memory under their levels as it loads
/// them, the small ones first; with a page for every 64 KiB a texture covers most of the pages
/// are for levels that are not loaded.
pub(super) fn level_pages() -> bool {
    #[cfg(test)]
    if LEVEL_PAGES_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_TEXTURE_LEVEL_PAGES").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_TEXTURE_LEVEL_PAGES=1 - a sampled texture's levels get their video memory pages when an upload first has texels for them");
        }
        on
    })
}

/// `level_pages` for a test (under `VRAM_TEST_LOCK`)
#[cfg(test)]
pub(super) static LEVEL_PAGES_FOR_TEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Of the textures with lazy levels there are now: how many, the tiles they stand for, and the
/// tiles of them that have pages; and the level groups given pages so far.
static LEVEL_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// (textures with lazy levels, the tiles they stand for, the tiles of them with pages, level
/// groups given pages so far)
fn level_stats() -> (u64, u64, u64, u64) {
    let l = |k: usize| LEVEL_STATS[k].load(std::sync::atomic::Ordering::Relaxed);
    (l(0), l(1), l(2), l(3))
}

impl Levels {
    /// Its tiles that have pages.
    fn mapped_tiles(&self) -> u64 {
        self.groups.iter().filter(|g| g.3).map(|g| g.2 as u64).sum()
    }
}

/// The tiling of reserved texture `res` of `mips` levels as `Levels`, and its tiles in all.
unsafe fn level_tiling(dev: &ID3D12Device, res: &ID3D12Resource, mips: u32) -> (Option<Levels>, u32) {
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut n = mips;
    let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); mips as usize];
    dev.GetResourceTiling(res, Some(&mut total), Some(&mut packed), None, Some(&mut n), 0, tilings.as_mut_ptr());
    let standard: Vec<(u32, u32)> = tilings
        .iter()
        .take((packed.NumStandardMips as usize).min(n as usize))
        .map(|t| (t.StartTileIndexInOverallResource, t.WidthInTiles * t.HeightInTiles as u32 * t.DepthInTiles as u32))
        .collect();
    let tail = (packed.NumPackedMips > 0).then_some((packed.StartTileIndexInOverallResource, packed.NumTilesForPackedMips));
    let whole = standard.len() == packed.NumStandardMips as usize && packed.NumStandardMips as u32 + packed.NumPackedMips as u32 == mips;
    (if whole { Levels::new(&standard, tail, total, mips) } else { None }, total)
}

/// Per-heap free-tile allocator for `Pages`.
///
/// Keeping free tiles partitioned by pool heap (sorted descending so `.pop()` yields ascending
/// contiguous tile indices) prevents streaming churn across 100+ 64 MiB pools from scattering a
/// single texture across dozens of heaps — which otherwise explodes `Pages::map` into dozens of
/// `UpdateTileMappings` calls per texture.
struct TilePool {
    /// Free tile indices for each pool heap (descending when `!dirty_sort[h]`).
    free_by_heap: Vec<Vec<u32>>,
    /// Whether `free_by_heap[h]` must be sorted descending before popping.
    dirty_sort: Vec<bool>,
    /// Total free tiles currently available across all heaps.
    free_total: usize,
    /// Tiles given back, reusable after `COOL`.
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
}

impl TilePool {
    fn new() -> Self {
        TilePool {
            free_by_heap: Vec::new(),
            dirty_sort: Vec::new(),
            free_total: 0,
            cooling: std::collections::VecDeque::new(),
        }
    }

    fn add_heap_at(&mut self, slot: u32, pool_pages: u32) {
        let idx = slot as usize;
        if idx >= self.free_by_heap.len() {
            self.free_by_heap.resize_with(idx + 1, Vec::new);
            self.dirty_sort.resize(idx + 1, false);
        }
        debug_assert!(self.free_by_heap[idx].is_empty());
        self.free_by_heap[idx] = (0..pool_pages).rev().collect();
        self.dirty_sort[idx] = false;
        self.free_total += pool_pages as usize;
    }

    /// Reclaim completely empty cooled heaps (`free_by_heap[h].len() == pool_pages`), keeping up
    /// to `keep_spare` empty heaps warm so steady-state streaming never thrashes `CreateHeap`.
    /// Returns the reclaimed heap indices so the caller can drop their `ID3D12Heap`s.
    fn reclaim_empty_heaps(&mut self, keep_spare: usize, pool_pages: u32) -> Vec<u32> {
        let full = pool_pages as usize;
        let mut empty: Vec<usize> = self
            .free_by_heap
            .iter()
            .enumerate()
            .filter_map(|(h, tiles)| (tiles.len() == full).then_some(h))
            .collect();
        if empty.len() <= keep_spare {
            return Vec::new();
        }
        let mut reclaimed = Vec::with_capacity(empty.len() - keep_spare);
        for h in empty.drain(keep_spare..) {
            self.free_by_heap[h].clear();
            self.free_by_heap[h].shrink_to_fit();
            self.dirty_sort[h] = false;
            self.free_total -= full;
            reclaimed.push(h as u32);
        }
        reclaimed
    }

    fn push_cooling(&mut self, tile: Tile, now: std::time::Instant) {
        self.cooling.push_back((tile, now));
    }

    /// Move every cooled tile (`>= COOL`) into its pool heap's free list so best-fit allocation
    /// sees all ready tiles at once.
    fn drain_cooling(&mut self, now: std::time::Instant) {
        while self.cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            let Some(((h, t), _)) = self.cooling.pop_front() else { break };
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    fn return_tiles(&mut self, tiles: &[Tile]) {
        for &(h, t) in tiles {
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    /// Pick the pool heap to take up to `rem` tiles from:
    /// 1. `preferred` if it fits all `rem` tiles (keeps partial-overlap placements on 1 heap),
    /// 2. otherwise best-fit heap with `len >= rem` (fits the whole texture in 1 heap without
    ///    splitting fuller pools),
    /// 3. otherwise the heap with the most free tiles (minimizing distinct heaps touched).
    fn pick_heap(&self, rem: usize, preferred: Option<u32>) -> Option<usize> {
        if self.free_total == 0 {
            return None;
        }
        if let Some(p) = preferred.map(|h| h as usize).filter(|&h| h < self.free_by_heap.len()) {
            if self.free_by_heap[p].len() >= rem {
                return Some(p);
            }
        }
        let mut best_fit: Option<(usize, usize)> = None;
        let mut largest: Option<(usize, usize)> = None;
        for (h, tiles) in self.free_by_heap.iter().enumerate() {
            let len = tiles.len();
            if len == 0 {
                continue;
            }
            if len >= rem {
                if best_fit.is_none_or(|(_, blen)| len < blen) {
                    best_fit = Some((h, len));
                }
            } else if largest.is_none_or(|(lh, llen)| len > llen || (len == llen && preferred == Some(h as u32) && preferred != Some(lh as u32))) {
                largest = Some((h, len));
            }
        }
        best_fit.or(largest).map(|(h, _)| h)
    }

    /// Allocate `count` tiles, preferring a single heap and ascending contiguous offsets.
    /// `make_heap` allocates a new backing pool heap of `pool_pages` tiles when `free_total == 0`.
    /// On failure, any tiles already taken are returned untouched.
    fn take_batch(
        &mut self,
        count: usize,
        preferred: Option<u32>,
        now: std::time::Instant,
        pool_pages: u32,
        mut make_heap: impl FnMut() -> Option<u32>,
    ) -> Option<Vec<Tile>> {
        if count == 0 {
            return Some(Vec::new());
        }
        self.drain_cooling(now);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let rem = count - out.len();
            let h = match self.pick_heap(rem, preferred) {
                Some(h) => h,
                None => {
                    let Some(new_h) = make_heap() else {
                        self.return_tiles(&out);
                        return None;
                    };
                    self.add_heap_at(new_h, pool_pages);
                    new_h as usize
                }
            };
            if self.dirty_sort[h] {
                self.free_by_heap[h].sort_unstable_by(|a, b| b.cmp(a));
                self.dirty_sort[h] = false;
            }
            let free = &mut self.free_by_heap[h];
            let take = rem.min(free.len());
            for _ in 0..take {
                let t = free.pop().unwrap();
                out.push((h as u32, t));
            }
            self.free_total -= take;
        }
        Some(out)
    }
}

/// Allocate one 64 MiB backing heap for `Pages`.
unsafe fn alloc_pool_heap(dev: &ID3D12Device, properties: D3D12_HEAP_PROPERTIES) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
        Properties: properties,
        Alignment: 0,
        // every kind of texture (resource heap tier 2: the only flags for both)
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut heap)) {
        crate::xlog_warn!("D3D12.X: cannot allocate more title texture pages (heap type {}, memory pool {}): {e}", properties.Type.0, properties.MemoryPoolPreference.0);
        return None;
    }
    if super::real::dred_on() {
        if let Some(ref h) = heap {
            super::real::name_heap(h.as_raw(), "xcl vram tile_pool");
        }
    }
    heap
}

/// Backing 64 MiB heaps for `Pages`, with slot reuse when completely empty heaps are reclaimed.
/// `len()` reports the count of live heaps.
struct PoolHeaps {
    slots: Vec<Option<ID3D12Heap>>,
    free_slots: Vec<u32>,
    live: usize,
}

impl PoolHeaps {
    fn new() -> Self {
        Self { slots: Vec::new(), free_slots: Vec::new(), live: 0 }
    }

    fn len(&self) -> usize {
        self.live
    }

    fn push_heap(&mut self, heap: ID3D12Heap) -> u32 {
        self.live += 1;
        if let Some(idx) = self.free_slots.pop() {
            debug_assert!(self.slots[idx as usize].is_none());
            self.slots[idx as usize] = Some(heap);
            idx
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Some(heap));
            idx
        }
    }

    fn take_heap(&mut self, idx: u32) -> Option<ID3D12Heap> {
        let h = self.slots.get_mut(idx as usize)?.take()?;
        self.live -= 1;
        self.free_slots.push(idx);
        Some(h)
    }
}

impl std::ops::Index<usize> for PoolHeaps {
    type Output = ID3D12Heap;
    fn index(&self, idx: usize) -> &Self::Output {
        self.slots[idx].as_ref().expect("live VRAM pool heap")
    }
}

fn keep_spare_pools() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("XCL_VRAM_KEEP_SPARE_POOLS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
    })
}

/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    /// per-device timestamp of the last periodic reclaim check
    last_reclaim: std::time::Instant,
    pool_properties: D3D12_HEAP_PROPERTIES,
    pools: PoolHeaps,
    pool: TilePool,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
    /// cached `reservable()` answers for `(DXGI_FORMAT, SampleCount)` on this device
    reservable_cache: std::collections::HashMap<(i32, u32), bool>,
    /// reusable scratch buffers for `map()`
    scratch_heaps: Vec<u32>,
    scratch_flags: Vec<D3D12_TILE_RANGE_FLAGS>,
    scratch_offsets: Vec<u32>,
    scratch_counts: Vec<u32>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

fn texture_pool_properties(system_memory: bool) -> D3D12_HEAP_PROPERTIES {
    if system_memory {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        }
    } else {
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
    }
}

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let _ = queue.SetName(windows::core::w!("xcl vram tile-mapping queue"));
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        let system_memory = std::env::var("XCL_TEXTURE_PAGES_SYSTEM_MEMORY").is_ok_and(|value| value.trim() == "1");
        if system_memory {
            crate::xlog_info!("D3D12.X: title texture page pools use system memory (D3D12_MEMORY_POOL_L0)");
        }
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            last_reclaim: std::time::Instant::now(),
            pool_properties: texture_pool_properties(system_memory),
            pools: PoolHeaps::new(),
            pool: TilePool::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            waited: std::collections::HashMap::new(),
            reservable_cache: std::collections::HashMap::new(),
            scratch_heaps: Vec::new(),
            scratch_flags: Vec::new(),
            scratch_offsets: Vec::new(),
            scratch_counts: Vec::new(),
        })
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.pool.cooling, now);
    }

    /// Sweep expired idle pages, drain cooled tiles, and reclaim any completely empty 64 MiB
    /// pool heaps (above the warm spare count), returning their `ID3D12Heap`s to be dropped
    /// outside `PAGES.lock()`.
    fn sweep_and_reclaim(&mut self, now: std::time::Instant) -> Vec<ID3D12Heap> {
        self.sweep(now);
        self.pool.drain_cooling(now);
        let reclaimed_ids = self.pool.reclaim_empty_heaps(keep_spare_pools(), POOL_PAGES);
        let mut dropped = Vec::with_capacity(reclaimed_ids.len());
        for idx in reclaimed_ids {
            if let Some(h) = self.pools.take_heap(idx) {
                dropped.push(h);
            }
        }
        if !dropped.is_empty() {
            let mb = (dropped.len() * POOL_PAGES as usize * PAGE) >> 20;
            crate::xlog_info!(
                "D3D12.X: reclaimed {} empty 64 MiB texture pool heap(s) ({mb} MiB queued for retirement; {} pools active)",
                dropped.len(),
                self.pools.len()
            );
        }
        dropped
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap.
    /// `tiles` for the tiles of `res` from the first one of subresource `sub` on.
    unsafe fn map(&mut self, res: &ID3D12Resource, sub: u32, tiles: &[Tile]) {
        if tiles.is_empty() {
            return;
        }
        self.scratch_heaps.clear();
        self.scratch_heaps.extend(tiles.iter().map(|t| t.0));
        self.scratch_heaps.sort_unstable();
        self.scratch_heaps.dedup();
        for i in 0..self.scratch_heaps.len() {
            let h = self.scratch_heaps[i];
            let ranges = heap_ranges_trimmed(h, tiles);
            if ranges.is_empty() {
                continue;
            }
            self.scratch_flags.clear();
            self.scratch_offsets.clear();
            self.scratch_counts.clear();
            let mut num_tiles = 0u32;
            for &(mapped, off, count) in &ranges {
                self.scratch_flags.push(if mapped { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP });
                self.scratch_offsets.push(off);
                self.scratch_counts.push(count);
                num_tiles += count;
            }
            let coord = D3D12_TILED_RESOURCE_COORDINATE { Subresource: sub, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: num_tiles, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(self.scratch_flags.as_ptr()),
                Some(self.scratch_offsets.as_ptr()),
                Some(self.scratch_counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        if total == 0 {
            return;
        }
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    fn release_tiles(&mut self, used: Vec<usize>, used_extra: Vec<(usize, usize)>, now: std::time::Instant, immediate_cool: bool) {
        for a in used {
            if let Some(p) = self.pages.get_mut(&a) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.pages.remove(&a) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Own(a), now));
                    }
                }
            }
        }
        for k in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&k) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.extra_pages.remove(&k) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(k.0, k.1), now));
                    }
                }
            }
        }
    }

    /// Acquire/share all `total` tiles for `res` at `va` and map them on the `Pages` queue.
    unsafe fn bind(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize, console_bytes: Option<u64>, total: u32) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        self.bind_range(dev, res, va, console_bytes, total, 0, 0..total as usize)
    }

    /// `bind` for the tiles `range` of the resource's `total`, which start at subresource `sub`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn bind_range(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize, console_bytes: Option<u64>, total: u32, sub: u32, range: std::ops::Range<usize>) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        let (first, own) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten()?;
        let now = std::time::Instant::now();
        self.sweep(now);

        let mut missing = 0usize;
        let mut preferred: Option<u32> = None;
        for i in range.clone() {
            let existing = if i < own {
                self.pages.get(&(first + i * PAGE)).map(|p| p.tile.0)
            } else {
                self.extra_pages.get(&(va, i)).map(|p| p.tile.0)
            };
            match existing {
                Some(h) if preferred.is_none() => preferred = Some(h),
                Some(_) => {}
                None => missing += 1,
            }
        }

        let properties = self.pool_properties;
        let pools = &mut self.pools;
        let pool = &mut self.pool;
        let fresh = pool.take_batch(missing, preferred, now, POOL_PAGES, || {
            let heap = alloc_pool_heap(dev, properties)?;
            Some(pools.push_heap(heap))
        })?;
        let mut fresh_iter = fresh.into_iter();

        let mut tiles = Vec::with_capacity(range.len());
        let mut used = Vec::with_capacity(own.min(range.len()));
        let mut used_extra = Vec::new();
        for i in range {
            let tile = if i < own {
                let a = first + i * PAGE;
                let t = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used.push(a);
                t
            } else {
                let k = (va, i);
                let t = match self.extra_pages.get_mut(&k) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.extra_pages.insert(k, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used_extra.push(k);
                t
            };
            tiles.push(tile);
        }
        self.map(res, sub, &tiles);
        Some((used, used_extra))
    }
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// Like `heap_ranges`, with any trailing `SKIP` range trimmed so `UpdateTileMappings` only walks
/// up to the last tile belonging to `heap`.
fn heap_ranges_trimmed(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out = heap_ranges(heap, tiles);
    while out.last().is_some_and(|r| !r.0) {
        out.pop();
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

/// A reserved texture needs TILED in the format's support (`support2`) and, multisampled, a
/// tiled quality level at its sample count.
fn can_reserve(support2: i32, samples: u32, tiled_msaa_levels: u32) -> bool {
    support2 & D3D12_FORMAT_SUPPORT2_TILED.0 != 0 && (samples <= 1 || tiled_msaa_levels > 0)
}

/// Whether the device can make `desc` a reserved texture, asked before CreateReservedResource:
/// refused, it is an error in the debug layer (#940: R32G8X24_TYPELESS depth, 1 and 4 samples).
/// A query that fails says yes, and creation decides.
unsafe fn reservable(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut f as *mut _ as *mut c_void, std::mem::size_of_val(&f) as u32).is_err() {
        return true;
    }
    // a format without TILED is not reservable whatever the sample count's query says
    if !can_reserve(f.Support2.0, 1, 0) {
        return false;
    }
    let samples = desc.SampleDesc.Count;
    let mut m = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: desc.Format, SampleCount: samples, Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE, NumQualityLevels: 0 };
    if samples > 1 && dev.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &mut m as *mut _ as *mut c_void, std::mem::size_of_val(&m) as u32).is_err() {
        return true;
    }
    can_reserve(f.Support2.0, samples, m.NumQualityLevels)
}

/// Canonical physical-backing address for `va`: when `va` lies inside an unmapped GPU mirror view
/// of title CPU memory (`kernelx::mirrors::source`), map both views to the same backing address so
/// shared 64 KiB VRAM pages and arenas alias instead of duplicating video memory. If distinct
/// physical or committed pages are mapped directly at `va`, keep `va` unaliased (matching
/// `title_textures::mirror_source`).
fn canonical_va(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    lazy: bool,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let va = canonical_va(va);
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let log = |why: &str| {
        if require_aligned && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({why}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
    };
    let dev_key = dev.as_raw() as usize;
    let is_reservable = {
        let mut g = PAGES.lock().unwrap();
        let pages = g.entry(dev_key).or_insert_with(|| Pages::new(dev)).as_mut()?;
        *pages.reservable_cache.entry((tiled.Format.0, tiled.SampleDesc.Count)).or_insert_with(|| reservable(dev, &tiled))
    };
    if !is_reservable {
        log("no tiled support for its format or sample count");
        return None;
    }
    // CreateReservedResource and GetResourceTiling do not touch `PAGES`; keep them outside the
    // lock so concurrent streaming threads, `before_execute`, and `page_bytes` do not serialize.
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        log(&format!("{hr:#x}"));
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    let levels = if lazy {
        let (levels, n) = level_tiling(dev, res, desc.MipLevels as u32);
        total = n;
        levels.filter(|_| total > 0 && owned_pages(va, console_bytes, total as usize).is_some())
    } else {
        dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
        None
    };
    let bound = {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            if levels.is_some() {
                // nothing mapped yet: a level gets its pages with its first texels (`map_levels`)
                LEVEL_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                LEVEL_STATS[1].fetch_add(total as u64, std::sync::atomic::Ordering::Relaxed);
                pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used: Vec::new(), used_extra: Vec::new(), levels });
                true
            } else if let Some((used, used_extra)) = pages.bind(dev, res, va, console_bytes, total) {
                pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used, used_extra, levels: None });
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    if !bound {
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// Unmap a placed texture superseded by a placement with a different address range, returning its
/// unshared tiles (`extra_pages` and unshared tail pages) to the cooling pool immediately.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.get_mut(&key) else { continue };
        if !t.mapped {
            return;
        }
        let total = t.total;
        let used = std::mem::take(&mut t.used);
        let used_extra = std::mem::take(&mut t.used_extra);
        t.mapped = false;
        if let Some(l) = t.levels.as_mut() {
            LEVEL_STATS[2].fetch_sub(l.mapped_tiles(), std::sync::atomic::Ordering::Relaxed);
            l.groups.iter_mut().for_each(|g| g.3 = false);
        }
        pages.unmap(res, total);
        pages.release_tiles(used, used_extra, now, true);
        return;
    }
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released.
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> bool {
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return false;
    }
    let Some(dev) = dev else { return false };
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return false };
    let Some((va, console_bytes, total, mapped)) = pages.targets.get(&key).map(|t| (t.va, t.console_bytes, t.total, t.mapped)) else {
        return false;
    };
    if mapped {
        return true;
    }
    // (lazy levels: they get their pages again with their next texels)
    if let Some(t) = pages.targets.get_mut(&key).filter(|t| t.levels.is_some()) {
        t.mapped = true;
        return true;
    }
    let Some((used, used_extra)) = pages.bind(&dev, res, va, console_bytes, total) else {
        return false;
    };
    if let Some(t) = pages.targets.get_mut(&key) {
        t.used = used;
        t.used_extra = used_extra;
        t.mapped = true;
    }
    true
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.remove(&resource) else { continue };
        if let Some(l) = &t.levels {
            LEVEL_STATS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            LEVEL_STATS[1].fetch_sub(t.total as u64, std::sync::atomic::Ordering::Relaxed);
            LEVEL_STATS[2].fetch_sub(l.mapped_tiles(), std::sync::atomic::Ordering::Relaxed);
        }
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
        }
        return;
    }
}

/// Reclaimed heaps waiting for `unmap_mark` (if any) and every live queue's fence mark on one device.
struct RetiringBatch<H, F> { heaps: Vec<H>, unmap_mark: Option<(F, u64)>, marks: Vec<(F, u64)> }

/// Per-device multi-queue retirement state (`H` = heap, `F` = fence; pure-CPU types in tests).
struct RetireQueue<H, F> { fences: std::collections::HashMap<usize, (F, u64)>, batches: Vec<RetiringBatch<H, F>> }

impl<H, F: Clone> RetireQueue<H, F> {
    fn new() -> Self { Self { fences: std::collections::HashMap::new(), batches: Vec::new() } }

    /// Seal `heaps` behind `unmap_mark` and one fence signal per queue (signaled BEFORE queueing).
    /// Returns `heaps` immediately when zero marks were signaled and `unmap_mark` is `None`.
    fn retire_with<Q>(&mut self, heaps: Vec<H>, unmap_mark: Option<(F, u64)>, queues: &[(usize, Q)], mut make_fence: impl FnMut() -> Option<F>, mut signal: impl FnMut(&Q, &F, u64) -> bool) -> Vec<H> {
        if heaps.is_empty() { return Vec::new(); }
        let mut marks = Vec::with_capacity(queues.len());
        for (k, q) in queues {
            let (f, v) = match self.fences.entry(*k) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => { let Some(f) = make_fence() else { continue }; e.insert((f, 0)) }
            };
            if signal(q, f, *v + 1) { *v += 1; marks.push((f.clone(), *v)); }
        }
        if marks.is_empty() && unmap_mark.is_none() { return heaps; }
        self.batches.push(RetiringBatch { heaps, unmap_mark, marks });
        Vec::new()
    }

    /// Reap completed batches and prune dead-queue fences once completed; caller drops outside locks.
    fn sweep_with(&mut self, live_queues: &HashSet<usize>, mut completed: impl FnMut(&F) -> u64) -> (Vec<RetiringBatch<H, F>>, Vec<F>) {
        let (mut done, mut kept) = (Vec::new(), Vec::new());
        for b in std::mem::take(&mut self.batches) {
            if b.unmap_mark.as_ref().is_none_or(|(f, v)| completed(f) >= *v) && b.marks.iter().all(|(f, v)| completed(f) >= *v) { done.push(b); } else { kept.push(b); }
        }
        self.batches = kept;
        let stale: Vec<usize> = self.fences.iter().filter(|(k, (f, v))| !live_queues.contains(k) && completed(f) >= *v).map(|(&k, _)| k).collect();
        let pruned = stale.into_iter().filter_map(|k| self.fences.remove(&k).map(|(f, _)| f)).collect();
        (done, pruned)
    }
}

unsafe impl Send for RetireQueue<ID3D12Heap, ID3D12Fence> {}

static RETIRE: Mutex<Option<std::collections::HashMap<usize, RetireQueue<ID3D12Heap, ID3D12Fence>>>> = Mutex::new(None);
static RETIRING_BATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Collect all known live command queues belonging to `dev_key` (deduplicated by raw ptr).
unsafe fn collect_device_queues(dev_key: usize, extra: Option<&ID3D12CommandQueue>) -> Vec<(usize, ID3D12CommandQueue)> {
    let (mut seen, mut out) = (HashSet::new(), Vec::new());
    let mut add = |q: ID3D12CommandQueue| {
        let key = q.as_raw() as usize;
        let mut q_dev: Option<ID3D12Device> = None;
        if seen.insert(key) && q.GetDevice(&mut q_dev).is_ok() && q_dev.is_some_and(|d| d.as_raw() as usize == dev_key) {
            out.push((key, q));
        }
    };
    for raw in super::present::known_queues() {
        let raw = raw as *mut c_void;
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) { add(q.clone()); }
    }
    for q in extra.into_iter().cloned().chain(super::title_textures::upload_queues()).chain(super::sparse::queue()) {
        add(q);
    }
    out
}

/// Retire `newly_reclaimed` across all live queues on `dev_key`, sweep completed batches, and prune
/// dead-queue fences; drops every finished `ID3D12Heap` and `ID3D12Fence` outside all locks.
unsafe fn retire_heaps_and_sweep(dev: &ID3D12Device, dev_key: usize, newly_reclaimed: Vec<ID3D12Heap>, unmap_mark: Option<(ID3D12Fence, u64)>, extra_queue: Option<&ID3D12CommandQueue>) {
    let queues = collect_device_queues(dev_key, extra_queue);
    let live_keys: HashSet<usize> = queues.iter().map(|&(k, _)| k).collect();
    let _drop_outside_locks = {
        let mut guard = RETIRE.lock().unwrap();
        let map = guard.get_or_insert_with(std::collections::HashMap::new);
        if newly_reclaimed.is_empty() && !map.contains_key(&dev_key) { return; }
        let rq = map.entry(dev_key).or_insert_with(RetireQueue::new);
        let before = rq.batches.len();
        let immediate = rq.retire_with(newly_reclaimed, unmap_mark, &queues, || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok(), |q, f, v| q.Signal(f, v).is_ok());
        let (done, pruned) = rq.sweep_with(&live_keys, |f| f.GetCompletedValue());
        RETIRING_BATCHES.fetch_add(rq.batches.len().wrapping_sub(before), std::sync::atomic::Ordering::AcqRel);
        (immediate, done, pruned)
    };
}

/// Before GPU work on `queue`: have it wait for every tile mapping made so far on its device,
/// release any previously retired heaps whose GPU fences have completed, and periodically (once
/// per second per device) retire completely empty 64 MiB VRAM pool heaps and idle arenas behind
/// per-queue GPU fence signals across all live queues on the device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    if queue.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let dev_key = dev.as_raw() as usize;
    let now = std::time::Instant::now();
    let mut newly_reclaimed = Vec::new();
    let mut do_reclaim = false;
    let mut unmap_mark: Option<(ID3D12Fence, u64)> = None;
    {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            let key = queue.as_raw() as usize;
            if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
                let _ = queue.Wait(&pages.fence, pages.value);
                super::list_x::note_internal_wait(key, "texture page mappings", &pages.fence, pages.value);
                pages.waited.insert(key, pages.value);
            }
            if now.duration_since(pages.last_reclaim) >= std::time::Duration::from_secs(1) {
                pages.last_reclaim = now;
                do_reclaim = true;
                newly_reclaimed = pages.sweep_and_reclaim(now);
                if !newly_reclaimed.is_empty() {
                    unmap_mark = Some((pages.fence.clone(), pages.value));
                }
            }
        }
    }
    if do_reclaim {
        newly_reclaimed.extend(ARENAS.lock().unwrap().sweep_dropped_for(dev_key, now));
    }
    if do_reclaim || !newly_reclaimed.is_empty() || RETIRING_BATCHES.load(std::sync::atomic::Ordering::Acquire) > 0 {
        retire_heaps_and_sweep(&dev, dev_key, newly_reclaimed, unmap_mark, Some(queue));
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Tile identities of one mapped resource: title pages, then extra pages.
/// A device-wide byte count also falls when unrelated idle pages are swept.
#[cfg(test)]
pub(super) fn mapped_tiles_for(device: usize, resource: usize) -> Option<(Vec<(u32, u32)>, Vec<(u32, u32)>)> {
    let g = PAGES.lock().unwrap();
    let p = g.get(&device)?.as_ref()?;
    let target = p.targets.get(&resource)?;
    if !target.mapped {
        return None;
    }
    let used = target.used.iter().map(|key| p.pages.get(key).map(|page| page.tile)).collect::<Option<Vec<_>>>()?;
    let extra = target.used_extra.iter().map(|key| p.extra_pages.get(key).map(|page| page.tile)).collect::<Option<Vec<_>>>()?;
    (used.len() + extra.len() == target.total as usize).then_some((used, extra))
}

#[cfg(test)]
pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    ((p.pages.len() + p.extra_pages.len()) * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// Pool/tile identities used by one mapped resource: title pages, then extra pages.
/// Unlike device-wide byte counts, these cannot shrink when unrelated idle targets expire.
#[cfg(test)]
pub(super) fn tile_mappings_for(device: usize, resource: usize) -> Option<(Vec<(u32, u32)>, Vec<(u32, u32)>)> {
    let g = PAGES.lock().unwrap();
    let p = g.get(&device)?.as_ref()?;
    let target = p.targets.get(&resource)?;
    if !target.mapped { return None; }
    let used = target.used.iter().map(|key| p.pages.get(key).map(|page| page.tile)).collect::<Option<Vec<_>>>()?;
    let extra = target.used_extra.iter().map(|key| p.extra_pages.get(key).map(|page| page.tile)).collect::<Option<Vec<_>>>()?;
    (used.len() + extra.len() == target.total as usize).then_some((used, extra))
}

/// (bytes of pages in use by tiled textures, bytes of pool heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((0, 0), |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A sampled texture of the title at `va` (`desc`: the PC description, `console_bytes` its size on
/// the console) backed by shared 64 KiB video-memory pages; None when reserved placement is not
/// available (the caller falls back to a committed resource).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) || std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        return None;
    }
    if console_has_packed_3d_mips(desc) {
        return None;
    }
    // (one slice: the tiles of an array's packed levels are not asked about here)
    let lazy = level_pages() && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.DepthOrArraySize == 1 && desc.SampleDesc.Count == 1 && desc.Flags == D3D12_RESOURCE_FLAG_NONE;
    place_on_pages(dev, this, va, desc, state, clear, console_bytes, false, lazy, riid, ppv)
}

/// The levels `levels` of `res` (None: all of them) get their pages if they have none yet: a
/// texture placed with lazy levels (any other: nothing to do). Returns the levels that got pages
/// by this call - the queue the texture is used on next has to wait for that (`before_execute`),
/// and what the pages held before is another texture's.
pub(super) unsafe fn map_levels(res: &ID3D12Resource, levels: Option<&[u32]>) -> Vec<u32> {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    for (&dev_key, pages) in g.iter_mut() {
        let Some(pages) = pages.as_mut() else { continue };
        let Some(t) = pages.targets.get(&key) else { continue };
        let Some(lv) = t.levels.as_ref().filter(|_| t.mapped) else { return Vec::new() };
        let mut wanted: Vec<usize> = match levels {
            Some(l) => l.iter().map(|&x| lv.group_of(x)).collect(),
            None => (0..lv.groups.len()).collect(),
        };
        wanted.sort_unstable();
        wanted.dedup();
        let todo: Vec<(usize, u32, u32, u32, std::ops::Range<u32>)> =
            wanted.into_iter().filter(|&k| !lv.groups[k].3).map(|k| (k, lv.groups[k].0, lv.groups[k].1, lv.groups[k].2, lv.levels_of(k))).collect();
        if todo.is_empty() {
            return Vec::new();
        }
        let (va, console_bytes, total) = (t.va, t.console_bytes, t.total);
        let d = dev_key as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { return Vec::new() };
        let mut fresh = Vec::new();
        for (k, sub, first, n, levels) in todo {
            let Some((used, used_extra)) = pages.bind_range(dev, res, va, console_bytes, total, sub, first as usize..(first + n) as usize) else { continue };
            let Some(t) = pages.targets.get_mut(&key) else { return fresh };
            t.used.extend(used);
            t.used_extra.extend(used_extra);
            if let Some(l) = t.levels.as_mut() {
                l.groups[k].3 = true;
            }
            LEVEL_STATS[2].fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
            let groups = LEVEL_STATS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if groups >= 1024 && groups.is_power_of_two() {
                let (textures, stand, mapped, _) = level_stats();
                crate::xlog_info!("D3D12.X: level pages: {textures} textures standing for {} MB have pages for {} MB ({groups} levels given pages so far)", stand >> 4, mapped >> 4);
            }
            fresh.extend(levels);
        }
        return fresh;
    }
    Vec::new()
}

/// Whether `resource` is a texture whose levels get their pages one by one (`map_levels`).
pub(super) fn has_lazy_levels(resource: usize) -> bool {
    PAGES.lock().unwrap().values().flatten().any(|p| p.targets.get(&resource).is_some_and(|t| t.levels.is_some()))
}

/// XG and the PC driver pack volume mip tails differently. Keep these textures on the
/// detile/upload path rather than mapping title pages directly onto the driver's packed tail.
fn console_has_packed_3d_mips(desc: &D3D12_RESOURCE_DESC) -> bool {
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || desc.MipLevels <= 1 {
        return false;
    }
    let mut console_desc = [0u8; 0x38];
    unsafe {
        std::ptr::copy_nonoverlapping(
            desc as *const _ as *const u8,
            console_desc.as_mut_ptr(),
            console_desc.len(),
        )
    };
    crate::hle::xg::Texture::new(&console_desc)
        .ok()
        .and_then(|t| t.texels().map(|s| s.depth > 0 && s.first_tail < s.chain.levels.len()))
        .unwrap_or(false)
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let va = canonical_va(va);
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, true, false, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let dev_key = dev.as_raw() as usize;
    let (taken, expired) = ARENAS.lock().unwrap().take_for(dev_key, va, info.SizeInBytes as usize, align, now, |size, align| arena_heap(dev, va, size, align));
    if !expired.is_empty() {
        retire_heaps_and_sweep(dev, dev_key, expired, None, None);
    }
    let (heap, offset, id) = taken?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        ARENAS.lock().unwrap().release(id, now);
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, id);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// The levels of a texture with lazy levels: a group of tiles for each level with tiles of its
    /// own and one for the packed levels; a level past the last standard one is in the packed
    /// group; a tiling that does not add up to the resource's tiles gives none.
    #[test]
    fn the_tiles_of_a_texture_are_grouped_by_its_levels() {
        // 2048x2048 BC7, 12 levels: 64, 16, 4, 1 tiles and 1 for the 8 packed levels
        let l = Levels::new(&[(0, 64), (64, 16), (80, 4), (84, 1)], Some((85, 1)), 86, 12).unwrap();
        assert_eq!(l.groups, vec![(0, 0, 64, false), (1, 64, 16, false), (2, 80, 4, false), (3, 84, 1, false), (4, 85, 1, false)]);
        assert_eq!((0..12).map(|i| l.group_of(i)).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]);
        assert_eq!((l.levels_of(0), l.levels_of(3), l.levels_of(4)), (0..1, 3..4, 4..12));
        // all of it packed: one group
        let small = Levels::new(&[], Some((0, 1)), 1, 5).unwrap();
        assert_eq!((small.group_of(0), small.group_of(4), small.levels_of(0)), (0, 0, 0..5));
        // no packed levels: the last level's group ends at the texture's levels
        let plain = Levels::new(&[(0, 4), (4, 1)], None, 5, 2).unwrap();
        assert_eq!((plain.group_of(1), plain.levels_of(1)), (1, 1..2));
        // a gap, an overlap, a tile count that is not the resource's, nothing at all
        assert_eq!(Levels::new(&[(0, 4), (5, 1)], None, 6, 2), None);
        assert_eq!(Levels::new(&[(0, 4), (3, 2)], None, 5, 2), None);
        assert_eq!(Levels::new(&[(0, 4), (4, 1)], None, 6, 2), None);
        assert_eq!(Levels::new(&[(0, 4), (4, 0)], None, 4, 2), None);
        assert_eq!(Levels::new(&[], None, 0, 1), None);
    }

    #[test]
    fn packed_console_3d_mip_tails_use_the_detile_upload_path() {
        fn desc(
            dimension: u32,
            width: u64,
            height: u32,
            depth: u16,
            mips: u16,
            format: u32,
            layout: u32,
        ) -> D3D12_RESOURCE_DESC {
            let mut bytes = [0u8; 0x38];
            bytes[0..4].copy_from_slice(&dimension.to_le_bytes());
            bytes[0x10..0x18].copy_from_slice(&width.to_le_bytes());
            bytes[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
            bytes[0x1C..0x1E].copy_from_slice(&depth.to_le_bytes());
            bytes[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
            bytes[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            bytes[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            bytes[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const D3D12_RESOURCE_DESC) }
        }

        let packed_3d = desc(4, 64, 64, 64, 4, 28, 0x105);
        assert!(console_has_packed_3d_mips(&packed_3d));
        let single_mip_3d = desc(4, 64, 64, 64, 1, 28, 0x105);
        assert!(!console_has_packed_3d_mips(&single_mip_3d));
        let packed_2d = desc(3, 64, 64, 1, 7, 28, 0x105);
        assert!(!console_has_packed_3d_mips(&packed_2d));
    }

    #[test]
    fn texture_pools_use_system_memory_only_when_requested() {
        let default = texture_pool_properties(false);
        assert_eq!(default.Type, D3D12_HEAP_TYPE_DEFAULT);
        assert_eq!(default.MemoryPoolPreference, D3D12_MEMORY_POOL_UNKNOWN);

        let system = texture_pool_properties(true);
        assert_eq!(system.Type, D3D12_HEAP_TYPE_CUSTOM);
        assert_eq!(system.CPUPageProperty, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE);
        assert_eq!(system.MemoryPoolPreference, D3D12_MEMORY_POOL_L0);
    }

    #[test]
    fn texture_page_batch_uses_the_selected_heap() {
        use crate::gpu::{adapter, agility, fence, recomp::runner::skip_test};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory6};
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};

        let _guard = VRAM_TEST_LOCK.lock().unwrap();
        unsafe {
            let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory6>() else { return skip_test("no DXGI factory") };
            let selector = adapter::Selector::from_env_or("auto");
            let Some((adapter, _)) = adapter::candidates(&factory, &selector).into_iter().next() else { return skip_test("no hardware adapter") };
            let Ok(dev) = agility::create_device(&adapter, D3D_FEATURE_LEVEL_11_0) else { return skip_test("no D3D12 device") };
            let Some(mut pages) = Pages::new(&dev) else { return skip_test("no tiled resources tier 2") };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 256, Height: 128, DepthOrArraySize: 1, MipLevels: 1,
                Format: DXGI_FORMAT_R32_UINT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut res).unwrap();
            let res = res.unwrap();
            let mut total = 0;
            dev.GetResourceTiling(&res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
            assert_eq!(total, 2);
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut options as *mut _ as *mut c_void, std::mem::size_of_val(&options) as u32).unwrap();
            let (used, extra) = pages.bind(&dev, &res, PAGE, Some(2 * PAGE as u64), total)
                .unwrap_or_else(|| panic!("bind two tiles (resource heap tier {})", options.ResourceHeapTier.0));
            fence::wait(&pages.fence, pages.value, &dev, None, 10_000).unwrap();
            assert_eq!(pages.pools.len(), 1);
            assert_eq!((pages.pages[&PAGE].tile, pages.pages[&(2 * PAGE)].tile), ((0, 0), (0, 1)));
            let properties = pages.pools[0].GetDesc().Properties;
            if std::env::var("XCL_TEXTURE_PAGES_SYSTEM_MEMORY").is_ok_and(|value| value.trim() == "1") {
                assert_eq!(properties.Type, D3D12_HEAP_TYPE_CUSTOM);
                assert_eq!(properties.MemoryPoolPreference, D3D12_MEMORY_POOL_L0);
                assert_eq!(properties.CPUPageProperty, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE);
            } else {
                assert_eq!(properties.Type, D3D12_HEAP_TYPE_DEFAULT);
            }

            // Round-trip distinct bytes through both mapped tiles before unmapping.
            let bytes = total as usize * PAGE;
            let buffer = |heap, state| {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes as u64, Height: 1,
                    DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
                };
                let mut resource: Option<ID3D12Resource> = None;
                let properties = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
                dev.CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut resource).unwrap();
                resource.unwrap()
            };
            let upload = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            let readback = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            let expected: Vec<_> = (0..bytes).map(|i| ((i / PAGE) * 37 + i % 251) as u8).collect();
            let mut ptr = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(expected.as_ptr(), ptr.cast::<u8>(), bytes);
            upload.Unmap(0, None);
            let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None::<&ID3D12PipelineState>).unwrap();
            let start = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE { NumTiles: total, ..Default::default() };
            list.CopyTiles(&res, &start, &region, &upload, 0, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&res),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            }]);
            list.CopyTiles(&res, &start, &region, &readback, 0, D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER);
            list.Close().unwrap();
            pages.queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            pages.value += 1;
            pages.queue.Signal(&pages.fence, pages.value).unwrap();
            fence::wait(&pages.fence, pages.value, &dev, None, 10_000).unwrap();
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            let matches = std::slice::from_raw_parts(ptr.cast::<u8>(), bytes) == expected;
            readback.Unmap(0, None);
            assert!(matches, "mapped texture bytes changed on the GPU round-trip");

            pages.unmap(&res, total);
            fence::wait(&pages.fence, pages.value, &dev, None, 10_000).unwrap();
            let now = Instant::now();
            pages.release_tiles(used, extra, now, true);
            pages.pool.drain_cooling(now + COOL);
            assert_eq!(pages.pool.free_total, POOL_PAGES as usize);
        }
    }

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        assert_eq!(heap_ranges_trimmed(1, &tiles), [(false, 0, 2), (true, 0, 2)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// Streaming churn across 100+ 64 MiB pool heaps: cooling tiles returned in interleaved
    /// cross-heap order are regrouped per heap and sorted into contiguous ascending runs, so new
    /// textures stay inside 1 pool heap (1 UpdateTileMappings call, 1 contiguous range) instead of
    /// scattering across dozens of heaps.
    #[test]
    fn tile_pool_keeps_churned_allocations_in_single_heap_and_contiguous() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let heaps_made = std::cell::Cell::new(0u32);
        let mut make = || {
            let h = heaps_made.get();
            heaps_made.set(h + 1);
            Some(h)
        };

        // Fill 100 pools (64 tiles each for a fast unit test, same multi-heap topology as 1024).
        const P: u32 = 64;
        let mut live: Vec<Vec<Tile>> = Vec::new();
        for _ in 0..100 {
            live.push(pool.take_batch(P as usize, None, t0, P, &mut make).unwrap());
        }
        assert_eq!(heaps_made.get(), 100);
        assert_eq!(pool.free_total, 0);

        // Free 32 tiles from every pool in interleaved round-robin order (worst-case churn across
        // all 100 pools in `cooling`).
        for tile_idx in 0..32usize {
            for h in 0..100usize {
                pool.push_cooling(live[h][tile_idx], t0);
            }
        }
        assert_eq!(pool.cooling.len(), 3200);

        // After COOL, allocating 100 textures of 32 tiles each must:
        // - allocate ZERO new heaps,
        // - place EVERY texture into exactly 1 single pool heap (1 UpdateTileMappings call),
        // - yield 1 single contiguous tile range (0..32) per texture!
        let t1 = t0 + COOL + Duration::from_millis(1);
        for _ in 0..100 {
            let alloc = pool.take_batch(32, None, t1, P, &mut make).unwrap();
            assert_eq!(alloc.len(), 32);
            let h0 = alloc[0].0;
            assert!(alloc.iter().all(|&(h, _)| h == h0), "texture scattered across multiple heaps");
            let ranges = heap_ranges_trimmed(h0, &alloc);
            assert_eq!(ranges, [(true, 0, 32)], "expected 1 contiguous ascending range");
        }
        assert_eq!(heaps_made.get(), 100, "must reuse cooled tiles without making new heaps");
        assert_eq!(pool.free_total, 0);

        // Partial overlap preference: if heap 7 has free tiles and preferred=Some(7), use heap 7.
        for t in 10..18u32 {
            pool.push_cooling((7, t), t1);
            pool.push_cooling((3, t), t1);
        }
        let t2 = t1 + COOL + Duration::from_millis(1);
        let pref = pool.take_batch(8, Some(7), t2, P, &mut make).unwrap();
        assert!(pref.iter().all(|&(h, _)| h == 7), "preferred heap should be chosen");
        assert_eq!(heap_ranges_trimmed(7, &pref), [(true, 10, 8)]);

        // Rollback on OOM: requesting more tiles than available when make_heap fails returns every
        // popped tile cleanly so free_total is preserved.
        assert_eq!(pool.free_total, 8);
        let fail = pool.take_batch(16, None, t2, P, || None);
        assert!(fail.is_none());
        assert_eq!(pool.free_total, 8, "OOM rollback must return all popped tiles");
    }

    /// Empty cooled pool heaps (above the warm spare count) are reclaimed cleanly back to VRAM and
    /// their slot indices can be reused without growing the heap table.
    #[test]
    fn tile_pool_reclaims_empty_heaps_keeping_warm_spare_and_reuses_slots() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let free_slots: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new());
        let next_slot = std::cell::Cell::new(0u32);
        let mut alloc_slot = || {
            free_slots.borrow_mut().pop().or_else(|| {
                let h = next_slot.get();
                next_slot.set(h + 1);
                Some(h)
            })
        };

        const P: u32 = 64;
        let h0 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        let h1 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        let h2 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        let h3 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        assert_eq!(next_slot.get(), 4);
        assert_eq!(pool.free_total, 0);

        // Keep 1 tile live in heap 0; free all tiles in heaps 1, 2, 3.
        for &tile in &h0[1..] {
            pool.push_cooling(tile, t0);
        }
        for batch in [&h1, &h2, &h3] {
            for &tile in batch {
                pool.push_cooling(tile, t0);
            }
        }
        let t1 = t0 + COOL + Duration::from_millis(1);
        pool.drain_cooling(t1);
        assert_eq!(pool.free_total, 63 + 64 * 3);

        // Reclaim empty heaps keeping 1 warm spare: heaps 2 and 3 reclaim, heap 1 stays warm,
        // heap 0 still has 1 live tile so it is untouched.
        let reclaimed = pool.reclaim_empty_heaps(1, P);
        assert_eq!(reclaimed, vec![2, 3]);
        assert_eq!(pool.free_total, 63 + 64);
        free_slots.borrow_mut().extend(reclaimed);

        // First 64-tile allocation reuses warm spare heap 1 (no new heap allocated).
        let reuse_spare = pool.take_batch(P as usize, None, t1, P, &mut alloc_slot).unwrap();
        assert!(reuse_spare.iter().all(|&(h, _)| h == 1));
        assert_eq!(free_slots.borrow().len(), 2);

        // Next 64-tile allocation consumes heap 0's 63 free tiles plus 1 tile from reclaimed slot 3
        // without advancing `next_slot`.
        let reuse_slot = pool.take_batch(P as usize, None, t1, P, &mut alloc_slot).unwrap();
        assert_eq!(reuse_slot.iter().filter(|&&(h, _)| h == 0).count(), 63);
        assert_eq!(reuse_slot.iter().filter(|&&(h, _)| h == 3).count(), 1);
        assert_eq!(next_slot.get(), 4);
        assert_eq!(free_slots.borrow().len(), 1);
    }

    #[test]
    fn arenas_sweep_dropped_returns_expired_heaps() {
        let t0 = Instant::now();
        let mut arenas: Arenas<u32> = Arenas::new();
        let (_, _, id1) = arenas.take_for(1, 0x10_0000, 0x1_0000, PAGE, t0, |_, _| Some(101)).0.unwrap();
        let (_, _, id2) = arenas.take_for(1, 0x20_0000, 0x1_0000, PAGE, t0, |_, _| Some(202)).0.unwrap();
        arenas.release(id1, t0);
        arenas.release(id2, t0 + Duration::from_secs(1));
        assert!(arenas.sweep_dropped_for(1, t0 + Duration::from_millis(1900)).is_empty());
        assert_eq!(arenas.sweep_dropped_for(1, t0 + IDLE + Duration::from_millis(10)), vec![101]);
        assert_eq!(arenas.bytes(), 0x1_0000);
        assert_eq!(arenas.sweep_dropped_for(1, t0 + Duration::from_secs(1) + IDLE + Duration::from_millis(10)), vec![202]);
        assert_eq!(arenas.bytes(), 0);
    }

    #[test]
    fn arenas_sweep_dropped_preserves_newest_fitting_order() {
        let t0 = Instant::now();
        let mut arenas: Arenas<u32> = Arenas::new();
        let id1 = arenas.take_for(1, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(101)).0.unwrap().2;
        let id2 = arenas.take_for(1, 0x20_0000, PAGE, PAGE, t0, |_, _| Some(202)).0.unwrap().2;
        let id3 = arenas.take_for(1, 0x20_0000, 2 * PAGE, PAGE, t0, |_, _| Some(303)).0.unwrap().2;
        arenas.release(id1, t0);
        let t1 = t0 + IDLE + Duration::from_millis(10);
        assert_eq!(arenas.sweep_dropped_for(1, t1), vec![101]);
        let (heap, off, picked_id) = arenas.take_for(1, 0x20_0000, PAGE, PAGE, t1, |_, _| panic!("must reuse existing arena")).0.unwrap();
        assert_eq!((heap, off, picked_id), (303, 0, id3), "sweep must preserve insertion order so rposition selects the newest fitting arena");
        assert_ne!(picked_id, id2);
    }

    #[test]
    fn arenas_isolate_devices_and_take_for_returns_expired_heaps() {
        use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
        struct DropFlag(Arc<AtomicUsize>);
        impl Drop for DropFlag { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
        let (t0, d1, d2) = (Instant::now(), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let mut arenas: Arenas<Arc<DropFlag>> = Arenas::new();
        let (h1, _, id1) = arenas.take_for(0x1000, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(Arc::new(DropFlag(d1.clone())))).0.unwrap();
        let (h2, _, id2) = arenas.take_for(0x2000, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(Arc::new(DropFlag(d2.clone())))).0.unwrap();
        assert_ne!(id1, id2, "distinct devices at the same VA must never alias arenas");
        drop((h1, h2));
        arenas.release(id1, t0);
        arenas.release(id2, t0);
        let t1 = t0 + IDLE + Duration::from_millis(10);
        let exp1 = arenas.sweep_dropped_for(0x1000, t1);
        assert_eq!((exp1.len(), arenas.bytes(), d1.load(Ordering::SeqCst), d2.load(Ordering::SeqCst)), (1, PAGE, 0, 0));
        let (_, exp2) = arenas.take_for(0x2000, 0x20_0000, PAGE, PAGE, t1, |_, _| Some(Arc::new(DropFlag(Arc::new(AtomicUsize::new(0))))));
        assert_eq!((exp2.len(), d2.load(Ordering::SeqCst)), (1, 0), "take_for must not drop expired heaps inside ARENAS");
        drop((exp1, exp2));
        assert_eq!((d1.load(Ordering::SeqCst), d2.load(Ordering::SeqCst)), (1, 1));
    }

    #[test]
    fn retire_queue_waits_for_all_queues_and_prunes_dead_queue_fences() {
        use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
        let (mut rq, unmap_f) = (RetireQueue::<u32, Arc<AtomicU64>>::new(), Arc::new(AtomicU64::new(0)));
        let mut qf: std::collections::HashMap<usize, Arc<AtomicU64>> = std::collections::HashMap::new();
        let mk = || Some(Arc::new(AtomicU64::new(0)));
        assert!(rq.retire_with(vec![101], Some((unmap_f.clone(), 1)), &[(1, 1usize), (2, 2), (3, 3)], mk, |q, f, _| { qf.entry(*q).or_insert_with(|| f.clone()); true }).is_empty());
        assert!(rq.retire_with(vec![202], None, &[(1, 1usize), (2, 2), (4, 4)], mk, |q, f, _| { qf.entry(*q).or_insert_with(|| f.clone()); true }).is_empty());
        qf[&2].store(2, Ordering::SeqCst);
        qf[&3].store(1, Ordering::SeqCst);
        unmap_f.store(1, Ordering::SeqCst);
        let live: HashSet<usize> = [1, 2].into_iter().collect();
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert!(done.is_empty() && pruned.len() == 1 && Arc::ptr_eq(&pruned[0], &qf[&3]) && rq.fences.contains_key(&4));
        drop(pruned);
        qf[&1].store(1, Ordering::SeqCst);
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert_eq!((done.len(), done[0].heaps.as_slice(), pruned.len()), (1, &[101][..], 0));
        assert!(Arc::strong_count(&qf[&3]) > 1, "completed batch fences must drop outside lock");
        drop(done);
        qf[&1].store(2, Ordering::SeqCst);
        qf[&4].store(1, Ordering::SeqCst);
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert_eq!((done[0].heaps.as_slice(), pruned.len(), rq.batches.is_empty()), (&[202][..], 1, true));
        assert!(Arc::ptr_eq(&pruned[0], &qf[&4]));
        assert_eq!(rq.retire_with(vec![303], None, &[] as &[(usize, usize)], mk, |_, _, _| true), vec![303]);
    }

    #[test]
    fn gpu_retire_queue_holds_heap_until_blocked_queue_passes_mark() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
        struct GuardedHeap(#[allow(dead_code)] ID3D12Heap, Arc<AtomicBool>);
        impl Drop for GuardedHeap { fn drop(&mut self) { self.1.store(true, Ordering::SeqCst); } }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let q1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let q2: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let probe: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = D3D12_HEAP_DESC { SizeInBytes: PAGE as u64, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: PAGE as u64, Flags: D3D12_HEAP_FLAG_NONE };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&desc, &mut heap).unwrap();
            let dropped = Arc::new(AtomicBool::new(false));
            q1.Wait(&gate, 1).unwrap();
            let (k1, k2) = (q1.as_raw() as usize, q2.as_raw() as usize);
            let mut rq: RetireQueue<GuardedHeap, ID3D12Fence> = RetireQueue::new();
            assert!(rq.retire_with(vec![GuardedHeap(heap.unwrap(), dropped.clone())], None, &[(k1, q1.clone()), (k2, q2.clone())], || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok(), |q, f, v| q.Signal(f, v).is_ok()).is_empty());
            q2.Signal(&probe, 1).unwrap();
            probe.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let (done, pruned) = rq.sweep_with(&[k1, k2].into_iter().collect(), |f| f.GetCompletedValue());
            assert!(done.is_empty() && pruned.is_empty() && !dropped.load(Ordering::SeqCst), "heap dropped while Q1 was still blocked");
            gate.Signal(1).unwrap();
            q1.Signal(&probe, 2).unwrap();
            probe.SetEventOnCompletion(2, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let (done, pruned) = rq.sweep_with(&[k1].into_iter().collect(), |f| f.GetCompletedValue());
            assert_eq!((done.len(), pruned.len(), dropped.load(Ordering::SeqCst)), (1, 1, false));
            drop((done, pruned));
            assert!(dropped.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn canonical_va_aliases_unmapped_gpu_view_but_preserves_committed_override() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGES_64K: u32 = 0x2040_0000;
        const GRAPHICS_NOACCESS: u32 = 0x800;
        unsafe {
            let cpu = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let gpu_unmapped = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(cpu != 0 && gpu_unmapped != 0);
            crate::hle::kernelx::mirrors::reserved(cpu, PAGE, PAGES_64K, 0x04);
            crate::hle::kernelx::mirrors::reserved(gpu_unmapped, PAGE, PAGES_64K, GRAPHICS_NOACCESS | 0x01);

            // Unmapped GPU view aliases to its CPU view, and reverse lookup finds the GPU view.
            assert_eq!(canonical_va(gpu_unmapped), cpu);
            assert!(crate::hle::kernelx::mirrors::gpu_views_of(cpu).contains(&gpu_unmapped));

            // Once distinct committed memory is mapped directly at the GPU view, canonical_va must
            // NOT alias it back to the CPU view.
            assert!(!VirtualAlloc(gpu_unmapped as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert_eq!(canonical_va(gpu_unmapped), gpu_unmapped);

            crate::hle::kernelx::mirrors::released(gpu_unmapped);
            crate::hle::kernelx::mirrors::released(cpu);
            VirtualFree(gpu_unmapped as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reserved only with TILED support, and for MSAA a tiled sample count (#940: the driver
    /// refused R32G8X24_TYPELESS reserved, at 1 and 4 samples, and the debug layer said so).
    #[test]
    fn reserved_needs_tiled_format_and_tiled_msaa() {
        let tiled = D3D12_FORMAT_SUPPORT2_TILED.0;
        assert!(can_reserve(tiled, 1, 0));
        assert!(!can_reserve(0, 1, 0), "format without TILED");
        assert!(can_reserve(tiled, 4, 1));
        assert!(!can_reserve(tiled, 4, 0), "no tiled quality level at 4 samples");
        assert!(!can_reserve(0, 4, 1));
    }

    /// The device's answer agrees with CreateReservedResource: what `reservable` turns away the
    /// driver refuses, what it lets through the driver makes (WARP or a hardware adapter).
    #[test]
    fn reservable_agrees_with_the_driver() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: windows::Win32::Graphics::Dxgi::IDXGIFactory4 = windows::Win32::Graphics::Dxgi::CreateDXGIFactory1().expect("DXGI factory");
                let Ok(warp) = factory.EnumWarpAdapter::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>() else { return };
                let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
            }
            let Some(dev) = dev else { return };
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_err() || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
                return;
            }
            for (format, samples) in [(DXGI_FORMAT_R32G8X24_TYPELESS, 1), (DXGI_FORMAT_R32G8X24_TYPELESS, 4), (DXGI_FORMAT_R32_TYPELESS, 1), (DXGI_FORMAT_R8G8B8A8_UNORM, 1)] {
                let flags = if format == DXGI_FORMAT_R8G8B8A8_UNORM { D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET } else { D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 128, Height: 128, DepthOrArraySize: 1, MipLevels: 1, Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, Flags: flags, ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                let made = dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_ok() && r.is_some();
                assert_eq!(reservable(&dev, &desc), made, "format {} x{samples}: reservable vs CreateReservedResource", format.0);
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        a.take_for(1, va, bytes, align, now, |_, _| Some(n)).0.unwrap().2
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<u64>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| take(&mut a, a0 + start + k * S, S, PAGE, now)).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for id in live.pop_front().unwrap() {
                    a.release(id, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32, &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }

    /// Reproduces the multi-pool streaming churn pattern on the active D3D12 driver (comparing
    /// scattered cross-heap `UpdateTileMappings` against single-heap contiguous
    /// `UpdateTileMappings`, both with `FLAG_NONE` and per-texture `Signal`):
    /// `cargo test -p xcl --lib -- --ignored --exact hle::d3d12x::vram::tests::churn_update_tile_mappings_timing --nocapture`
    #[test]
    #[ignore]
    fn churn_update_tile_mappings_timing() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("D3D12 device");
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            const HEAPS: usize = 64;
            const TILES_PER_TEX: usize = 32; // 2 MiB tiled texture
            const TEX_COUNT: usize = 256;
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: 64 * PAGE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_NONE,
            };
            let mut heaps = Vec::with_capacity(HEAPS);
            for _ in 0..HEAPS {
                let mut h: Option<ID3D12Heap> = None;
                dev.CreateHeap(&heap_desc, &mut h).unwrap();
                heaps.push(h.unwrap());
            }
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1024,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let make_textures = || -> Vec<ID3D12Resource> {
                (0..TEX_COUNT)
                    .map(|_| {
                        let mut r: Option<ID3D12Resource> = None;
                        dev.CreateReservedResource(&tex_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                        r.unwrap()
                    })
                    .collect()
            };

            // BEFORE pattern: flat-FIFO cooling scatters each 32-tile texture across 32 heaps,
            // calling UpdateTileMappings(FLAG_NONE) per heap + Signal per texture.
            let old_res = make_textures();
            let mut fval = 0u64;
            let t0 = Instant::now();
            let mut old_calls = 0usize;
            for (k, r) in old_res.iter().enumerate() {
                let tiles: Vec<Tile> = (0..TILES_PER_TEX).map(|i| (((k + i) % HEAPS) as u32, (i % 64) as u32)).collect();
                let mut hs: Vec<u32> = tiles.iter().map(|t| t.0).collect();
                hs.sort_unstable();
                hs.dedup();
                for h in hs {
                    let ranges = heap_ranges(h, &tiles);
                    let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|x| if x.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
                    let offsets: Vec<u32> = ranges.iter().map(|x| x.1).collect();
                    let counts: Vec<u32> = ranges.iter().map(|x| x.2).collect();
                    let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                    let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
                    queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h as usize], ranges.len() as u32, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
                    old_calls += 1;
                }
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let old_submit = t0.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let old_total = t0.elapsed();

            // AFTER pattern: per-heap contiguous TilePool -> 1 heap, 1 contiguous range,
            // D3D12_TILE_MAPPING_FLAG_NONE, per-texture Signal.
            let new_res = make_textures();
            let t1 = Instant::now();
            let mut new_calls = 0usize;
            for (k, r) in new_res.iter().enumerate() {
                let h = k % HEAPS;
                let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                let region = D3D12_TILE_REGION_SIZE { NumTiles: TILES_PER_TEX as u32, UseBox: false.into(), ..Default::default() };
                let flag = D3D12_TILE_RANGE_FLAG_NONE;
                let off = 0u32;
                let cnt = TILES_PER_TEX as u32;
                queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h], 1, Some(&flag), Some(&off), Some(&cnt), D3D12_TILE_MAPPING_FLAG_NONE);
                new_calls += 1;
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let new_submit = t1.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let new_total = t1.elapsed();

            println!(
                "BEFORE ({TEX_COUNT} x 2MiB textures, {old_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={old_submit:?} gpu_done={old_total:?}"
            );
            println!(
                "AFTER  ({TEX_COUNT} x 2MiB textures, {new_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={new_submit:?} gpu_done={new_total:?}"
            );
            assert!(new_calls < old_calls);
        }
    }
}
