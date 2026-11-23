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
//! driver does not make reserved).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// bytes of title address space an arena stands for at least (and of video memory it takes)
const ARENA: usize = 256 << 20;
/// an arena that replaces others grows to multiples of this (`plan`)
const UNION_ALIGN: usize = 256 << 20;
/// arenas start and end at multiples of this
const ARENA_ALIGN: usize = 256 << 20;

struct Arena {
    base: usize,
    size: usize,
    heap: ID3D12Heap,
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arena {}

static ARENAS: Mutex<Vec<Arena>> = Mutex::new(Vec::new());
/// bytes of every arena made (replaced ones live on while their resources do)
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().iter().map(|a| a.size).sum(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
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
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// Where a placement at `[va, va + bytes)` goes, given the arenas' ranges (base, size).
#[derive(Debug, PartialEq)]
enum Plan {
    /// into arena `index`, at `offset`
    Reuse { index: usize, offset: usize },
    /// into a new arena `[base, base + size)`, which takes the place of the arenas `replaces`
    /// (indices, ascending) in the list
    New { base: usize, size: usize, replaces: Vec<usize> },
}

/// The arenas never overlap, so an address has one arena and resources at one address share
/// memory. A new arena takes the 32 MiB blocks the range needs and, while it overlaps none,
/// grows to 128 MiB – no further than the next arena. A range that straddles arenas gets one
/// arena over the union of the range and every arena it overlaps; those arenas leave the list
/// (their resources keep their heaps, but later placements at their addresses use the new one).
fn plan(arenas: &[(usize, usize)], va: usize, bytes: usize) -> Plan {
    if let Some(index) = arenas.iter().position(|&(b, n)| b <= va && va + bytes <= b + n) {
        return Plan::Reuse { index, offset: va - arenas[index].0 };
    }
    let (mut lo, mut hi) = (va & !(ARENA_ALIGN - 1), (va + bytes).div_ceil(ARENA_ALIGN) * ARENA_ALIGN);
    let overlaps = |lo: usize, hi: usize| -> Vec<usize> { (0..arenas.len()).filter(|&i| arenas[i].0 < hi && lo < arenas[i].0 + arenas[i].1).collect() };
    let mut replaces = overlaps(lo, hi);
    if replaces.is_empty() {
        let next = arenas.iter().map(|&(b, _)| b).filter(|&b| b >= hi).min().unwrap_or(usize::MAX);
        hi = hi.max(lo.saturating_add(ARENA)).min(next);
        return Plan::New { base: lo, size: hi - lo, replaces };
    }
    // the union grows as it takes in arenas, which may overlap further ones
    loop {
        for &i in &replaces {
            lo = lo.min(arenas[i].0);
            hi = hi.max(arenas[i].0 + arenas[i].1);
        }
        let more = overlaps(lo, hi);
        if more.len() == replaces.len() {
            // the replaced arenas live on with their resources: room to grow in, so the next
            // placements around here land in this one instead of replacing it in turn (a retail
            // game: 38 replacements left 6.4 GB of arenas over 2 GB of its addresses)
            let others: Vec<(usize, usize)> = (0..arenas.len()).filter(|i| !replaces.contains(i)).map(|i| arenas[i]).collect();
            let below = others.iter().map(|&(b, n)| b + n).filter(|&e| e <= lo).max().unwrap_or(0);
            let above = others.iter().map(|&(b, _)| b).filter(|&b| b >= hi).min().unwrap_or(usize::MAX);
            let lo = (lo & !(UNION_ALIGN - 1)).max(below);
            let hi = hi.div_ceil(UNION_ALIGN).saturating_mul(UNION_ALIGN).min(above);
            return Plan::New { base: lo, size: hi - lo, replaces };
        }
        replaces = more;
    }
}

/// The arena holding `[va, va + bytes)` (None: no video memory).
unsafe fn arena_for(dev: &ID3D12Device, va: usize, bytes: usize) -> Option<(ID3D12Heap, u64)> {
    let mut g = ARENAS.lock().unwrap();
    let ranges: Vec<(usize, usize)> = g.iter().map(|a| (a.base, a.size)).collect();
    let (base, size, replaces) = match plan(&ranges, va, bytes) {
        Plan::Reuse { index, offset } => return Some((g[index].heap.clone(), offset as u64)),
        Plan::New { base, size, replaces } => (base, size, replaces),
    };
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // MSAA targets need 4 MiB placement
        Alignment: D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {base:#x}: CreateHeap: {e}");
        return None;
    }
    let heap = heap?;
    MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
    if replaces.is_empty() {
        crate::xlog_info!("D3D12.X: video-memory arena for title addresses {base:#x}..{:#x}", base + size);
    } else {
        // the placed resources hold their heaps: the old arenas live on as long as they do
        crate::xlog_info!(
            "D3D12.X: video-memory arena for title addresses {base:#x}..{:#x} replaces {} arena(s) a placement at {va:#x}..{:#x} straddled (resources already in them no longer alias later ones)",
            base + size,
            replaces.len(),
            va + bytes
        );
        for &i in replaces.iter().rev() {
            g.remove(i);
        }
    }
    g.push(Arena { base, size, heap: heap.clone() });
    Some((heap, (va - base) as u64))
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

/// The title's pages of video memory for GPU-owned textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    free: Vec<Tile>,
    /// tiles given back, reusable after `COOL`
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// target → (the title pages it uses, tiles of its own past the title's memory)
    targets: std::collections::HashMap<usize, (Vec<usize>, Vec<Tile>)>,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// None inside: no pages on this device (not tried yet: the outer None)
static PAGES: Mutex<Option<Option<Pages>>> = Mutex::new(None);

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
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: GPU-owned textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            pools: Vec::new(),
            free: Vec::new(),
            cooling: std::collections::VecDeque::new(),
            pages: std::collections::HashMap::new(),
            targets: std::collections::HashMap::new(),
            waited: std::collections::HashMap::new(),
        })
    }

    /// Pages no target used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        let old: Vec<usize> = self.pages.iter().filter(|(_, p)| p.idle_since.is_some_and(|t| now.duration_since(t) >= IDLE)).map(|(&a, _)| a).collect();
        for a in old {
            if let Some(p) = self.pages.remove(&a) {
                self.cooling.push_back((p.tile, now));
            }
        }
    }

    unsafe fn take_tile(&mut self, dev: &ID3D12Device) -> Option<Tile> {
        if let Some(t) = self.free.pop() {
            return Some(t);
        }
        if self.cooling.front().is_some_and(|(_, at)| at.elapsed() >= COOL) {
            return self.cooling.pop_front().map(|(t, _)| t);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            // every kind of texture (resource heap tier 2: the only flags for both)
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
            crate::xlog_warn!("D3D12.X: no video memory for more pages of GPU-owned textures ({e})");
            return None;
        }
        self.pools.push(heap?);
        let h = self.pools.len() as u32 - 1;
        self.free.extend((1..POOL_PAGES).rev().map(|t| (h, t)));
        Some((h, 0))
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap.
    unsafe fn map(&mut self, res: &ID3D12Resource, tiles: &[Tile]) {
        let mut heaps: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        heaps.sort_unstable();
        heaps.dedup();
        for h in heaps {
            let ranges = heap_ranges(h, tiles);
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|r| if r.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
            let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
            let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
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

/// A GPU-owned texture at `va` as a reserved resource on the title's pages there
/// (`console_bytes`: its size on the console – the pages it covers; tiles past them are its
/// own). None: it cannot be (the caller tries an arena).
unsafe fn place_on_pages(dev: &ID3D12Device, this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    if va % PAGE != 0 || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN {
        return None;
    }
    let mut g = PAGES.lock().unwrap();
    let pages = g.get_or_insert_with(|| Pages::new(dev)).as_mut()?;
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({hr:#x}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    let own = console_bytes.map_or(total as usize, |b| (b as usize).div_ceil(PAGE)).min(total as usize);
    let now = std::time::Instant::now();
    pages.sweep(now);
    let mut tiles = Vec::with_capacity(total as usize);
    let mut used = Vec::with_capacity(own);
    let mut extra = Vec::new();
    for i in 0..total as usize {
        let tile = if i < own {
            let a = va + i * PAGE;
            let tile = match pages.pages.get_mut(&a) {
                Some(p) => {
                    p.users += 1;
                    p.idle_since = None;
                    Some(p.tile)
                }
                None => pages.take_tile(dev).inspect(|&t| {
                    pages.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                }),
            };
            if tile.is_some() {
                used.push(a);
            }
            tile
        } else {
            pages.take_tile(dev).inspect(|&t| extra.push(t))
        };
        let Some(t) = tile else { break };
        tiles.push(t);
    }
    pages.targets.insert(raw as usize, (used, extra));
    if tiles.len() < total as usize {
        // no video memory for all of it: the target goes (its pages with it)
        drop(g);
        forget_pages(raw as usize);
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    }
    pages.map(res, &tiles);
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// A target released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.as_mut() else { return };
    let Some((used, extra)) = pages.targets.remove(&resource) else { return };
    let now = std::time::Instant::now();
    for a in used {
        if let Some(p) = pages.pages.get_mut(&a) {
            p.users = p.users.saturating_sub(1);
            if p.users == 0 {
                p.idle_since = Some(now);
            }
        }
    }
    pages.cooling.extend(extra.into_iter().map(|t| (t, now)));
}

/// Before a title queue's work: it waits for the pages mapped since its last work.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.as_mut() else { return };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        let _ = queue.Wait(&pages.fence, pages.value);
        pages.waited.insert(key, pages.value);
    }
}

/// (bytes of pages in use by targets, bytes of pool heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.as_ref() else { return (0, 0) };
    (p.pages.len() * PAGE + p.targets.values().map(|t| t.1.len()).sum::<usize>() * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let (heap, offset) = arena_for(dev, va, info.SizeInBytes as usize)?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset as usize, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    /// Applies placements to a list of arena ranges as `arena_for` does; returns the arena
    /// (base) each placement landed in.
    fn place_all(seq: &[(usize, usize)]) -> (Vec<(usize, usize)>, Vec<usize>) {
        let mut arenas: Vec<(usize, usize)> = Vec::new();
        let mut landed = Vec::new();
        for &(va, bytes) in seq {
            match plan(&arenas, va, bytes) {
                Plan::Reuse { index, offset } => {
                    assert_eq!(arenas[index].0 + offset, va);
                    landed.push(arenas[index].0);
                }
                Plan::New { base, size, replaces } => {
                    assert!(base <= va && va + bytes <= base + size, "{va:#x} in {base:#x}+{size:#x}");
                    assert_eq!((base % ARENA_ALIGN, size % ARENA_ALIGN), (0, 0));
                    for &i in replaces.iter().rev() {
                        arenas.remove(i);
                    }
                    arenas.push((base, size));
                    landed.push(base);
                }
            }
            for (i, a) in arenas.iter().enumerate() {
                for b in &arenas[i + 1..] {
                    assert!(a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0, "arenas {a:x?} and {b:x?} overlap");
                }
            }
        }
        (arenas, landed)
    }

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    #[test]
    fn a_straddling_placement_gets_one_arena_over_both() {
        // #104: 16 MiB at 0x1_0400_0000, then 48 MiB at 0x1_0200_0000 (ends inside the first arena)
        let (arenas, landed) = place_all(&[(0x1_0400_0000, 16 * MIB), (0x1_0200_0000, 48 * MIB), (0x1_0400_0000, 16 * MIB)]);
        assert_eq!(arenas, [(0x1_0000_0000, UNION_ALIGN)]);
        // the third placement, at the first one's address, shares the new arena with the second
        assert_eq!(landed[1], landed[2]);
    }

    #[test]
    fn a_new_arena_stops_at_the_next_one() {
        // an arena at one alignment step first; then 16 MiB just below it fits in the step below
        let (a, b) = (0x1_0000_0000 + ARENA_ALIGN, 0x1_0000_0000 + ARENA_ALIGN - 16 * MIB);
        let (arenas, landed) = place_all(&[(a, 16 * MIB), (b, 16 * MIB)]);
        assert_eq!(arenas, [(a, ARENA), (0x1_0000_0000, ARENA_ALIGN)]);
        assert_eq!(landed, [a, 0x1_0000_0000]);
    }

    #[test]
    fn a_placement_over_several_arenas_merges_them_all() {
        // two arenas with a gap, then one placement over the gap into both
        let a = 0x2_0000_0000;
        let (arenas, _) = place_all(&[(a, MIB), (a + 160 * MIB, MIB), (a + 20 * MIB, 150 * MIB), (a + 50 * MIB, MIB)]);
        assert_eq!(arenas, [(a, UNION_ALIGN)]);
    }
}

#[cfg(test)]
#[path = "vram_gpu_tests.rs"]
mod gpu_regression_tests;
