//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (64 MiB)
const POOL_TILES: u32 = 1024;
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 64 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer
const READBACK_STAGING: usize = 16 << 20;

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = 64;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, sent: std::array::from_fn(|_| None) }
    }
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut Vec<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.push((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    unmapped
}

/// Drop the runs of `to_zero` whose tile is no longer mapped – once, where the runs are taken.
/// (Pruning at every decommit that unmapped a tile walked the whole list each time: it grows
/// until the next ExecuteCommandLists, and a title mapping gigabytes of physical pages while it
/// loads – GTA VI – spent seconds there, its watchdog ended the game.)
fn prune_zero_runs(mapped: &FxHashMap<usize, TileEntry>, runs: &mut Vec<(usize, usize)>) {
    runs.retain(|&(za, _)| mapped.contains_key(&(za & !(TILE - 1))));
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// fence signalled on title queues so `queue` waits before copying GPU writes back to CPU
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// video-memory pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    free: Vec<(u32, u32)>,
    /// pools all of whose tiles came free: released once the GPU is past `value` and a moment
    /// went by (titles' queues may still read what was mapped there)
    retired: Vec<(ID3D12Heap, u64, std::time::Instant)>,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// placed buffers → the pool tiles they use
    buffers: FxHashMap<usize, Vec<(u32, u32)>>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: Vec<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// `mapped` as sorted (start, end) runs (None: to be made again)
    runs: Option<Vec<(usize, usize)>>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// Scan a 4 KiB page at `pg`, appending any changed 64-byte chunk runs directly to `out`
/// (zero per-page heap allocation; skips allocating `slot` completely when a fresh zero page
/// is still all zeros).
#[inline]
fn scan_page(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            sent[k] = h;
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        let sent = slot.insert(Box::new(ZERO_HASHES));
        sent[k] = h;
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            sent[k2] = h2;
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
pub fn pool_bytes() -> (usize, usize) {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return (0, 0) };
    let pools = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE;
    (pools, pools.saturating_sub(s.free.len() * TILE))
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early`, merged with the ranges they overlap or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` (empty: no sparse windows).
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = (|| -> windows::core::Result<State> {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
        let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
        for _ in 0..UPLOAD_SLOTS {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            list.Close()?;
            slots.push(UploadSlot { alloc, list, fence_val: 0 });
        }
        let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
        rb_list.Close()?;
        Ok(State {
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            event,
            rb_fence,
            rb_value: 0,
            windows,
            pools: Vec::new(),
            free: Vec::new(),
            retired: Vec::new(),
            mapped: FxHashMap::default(),
            pinned: FxHashMap::default(),
            buffers: FxHashMap::default(),
            to_zero: Vec::new(),
            staging: None,
            ring_write: 0,
            ring_submitted: 0,
            ring_free: 0,
            inflight: VecDeque::new(),
            zero_buf: None,
            slots,
            next_slot: 0,
            readback: None,
            rb_alloc,
            rb_list,
            rb_last_fence: 0,
            waited: FxHashMap::default(),
            runs: None,
            scratch_runs: Vec::new(),
        })
    })();
    match made {
        Ok(s) => {
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory (tiled buffers on video memory)", out.len());
            *STATE.lock().unwrap() = Some(s);
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                committed(a, b - a);
            }
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        if let Some(t) = self.free.pop() {
            return Some((t, true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        Some(((h, 0), false))
    }

    /// Pools whose every tile is free again go (64 MB of video memory each – and as much of the
    /// system's commit: a retail game's homes of physical pages left 0.75 GB of pools for
    /// 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled.
    fn retire_free_pools(&mut self) {
        let now = std::time::Instant::now();
        let done = unsafe { self.fence.GetCompletedValue() };
        self.retired.retain(|(_, v, at)| !(done >= *v && now.duration_since(*at).as_secs() >= 2));
        if self.free.len() < POOL_TILES as usize {
            return;
        }
        let mut count = vec![0u32; self.pools.len()];
        for &(h, _) in &self.free {
            count[h as usize] += 1;
        }
        let gone: Vec<usize> = (0..count.len()).filter(|&h| count[h] == POOL_TILES && self.pools[h].is_some()).collect();
        if gone.is_empty() {
            return;
        }
        self.free.retain(|&(h, _)| !gone.contains(&(h as usize)));
        for h in gone {
            if let Some(heap) = self.pools[h].take() {
                self.retired.push((heap, self.value, now));
            }
        }
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(D3D12_TILE_RANGE_FLAG_NULL);
            counts.push(run);
            i += run as usize;
        }
        self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            None,
            flags.len() as u32,
            Some(flags.as_ptr()),
            None,
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_holes` supplies
    /// the initial 16-bit hole mask for each newly mapped 64 KiB tile.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_holes: impl FnMut(usize) -> u16) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new: Vec<(usize, (u32, u32))> = Vec::new();
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && self.window_index(t).is_some() {
                let Some((tile, reused)) = self.take_tile() else { break };
                self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                self.runs = None;
                if reused {
                    self.to_zero.push((t, TILE));
                }
                new.push((t, tile));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new.is_empty() {
            return;
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        // per window
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for x in new {
            if let Some(wi) = self.window_index(x.0) {
                by_window.entry(wi).or_default().push(x);
            }
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    #[inline]
    unsafe fn wait_for(&self, target: u64) {
        if target == 0 || self.fence.GetCompletedValue() >= target {
            return;
        }
        let _ = self.fence.SetEventOnCompletion(target, self.event);
        windows::Win32::System::Threading::WaitForSingleObject(self.event, 5000);
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = self.fence.GetCompletedValue();
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// Acquire the next upload command slot, waiting only if that slot's previous submission is
    /// still executing on the GPU, then reset its allocator and list.
    unsafe fn begin_slot(&mut self) -> usize {
        let idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let fv = self.slots[idx].fence_val;
        if fv > 0 && self.fence.GetCompletedValue() < fv {
            self.wait_for(fv);
            self.reclaim_ring();
        }
        let slot = &self.slots[idx];
        let _ = slot.alloc.Reset();
        let _ = slot.list.Reset(&slot.alloc, None);
        idx
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) {
        let closed = self.slots[idx].list.Close().is_ok();
        if closed {
            if let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>() {
                super::title_textures::own_work(|| {
                    self.queue.ExecuteCommandLists(&[Some(l)]);
                });
            }
        }
        self.signal();
        self.slots[idx].fence_val = self.value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((self.value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            let mut p = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut p)).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            r.Unmap(0, None);
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Update `sent` hashes for all 64-byte chunks overlapping `[va, va + len)` so the CPU
    /// copy-back we just wrote to `va` is not re-uploaded by the next `before_execute`.
    unsafe fn sync_sent_range(&mut self, va: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pg = va & !(PAGE - 1);
        let end = va + len;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
                        let lo = va.saturating_sub(p) / CHUNK;
                        let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                        for k in lo..hi {
                            sent[k] = chunk_hash(&bytes[k * CHUNK..(k + 1) * CHUNK]);
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view and refresh
    /// `sent` hashes. Caller has already ordered `self.queue` after any GPU writes if needed.
    unsafe fn readback_ranges(&mut self, ranges: &[(usize, usize)]) {
        let Some((rb, p)) = self.readback_buf() else { return };
        let mut pending: Vec<(usize, usize, usize)> = Vec::new(); // (va, rb_off, chunk)
        let mut used = 0usize;
        let mut open = false;
        let flush_and_copy = |s: &mut State, open: &mut bool, pending: &mut Vec<(usize, usize, usize)>, used: &mut usize, p: *mut u8| {
            if *open {
                *open = false;
                if s.rb_list.Close().is_ok() {
                    if let Ok(l) = s.rb_list.cast::<ID3D12CommandList>() {
                        super::title_textures::own_work(|| {
                            s.queue.ExecuteCommandLists(&[Some(l)]);
                        });
                    }
                }
                s.signal();
                s.rb_last_fence = s.value;
                s.wait_for(s.value);
                s.reclaim_ring();
            }
            for &(dst, off, len) in pending.iter() {
                // copy only committed (non-hole) bytes
                let mut at = dst;
                let mut src_off = off;
                let end = dst + len;
                while at < end {
                    let pg = at & !(PAGE - 1);
                    let t = pg & !(TILE - 1);
                    let idx = (pg - t) / PAGE;
                    let step = (pg + PAGE).min(end) - at;
                    if s.mapped.get(&t).is_some_and(|e| e.holes & (1u16 << idx) == 0) {
                        std::ptr::copy_nonoverlapping(p.add(src_off), at as *mut u8, step);
                    }
                    at += step;
                    src_off += step;
                }
                s.sync_sent_range(dst, len);
            }
            pending.clear();
            *used = 0;
        };
        for &(a, n) in ranges {
            if n == 0 {
                continue;
            }
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let b = self.windows[wi].0;
                let to_tile_end = (tile + TILE) - cur;
                if !self.mapped.contains_key(&tile) {
                    done += (n - done).min(to_tile_end);
                    continue;
                }
                if used == READBACK_STAGING {
                    flush_and_copy(self, &mut open, &mut pending, &mut used, p);
                }
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                if !open {
                    self.wait_for(self.rb_last_fence);
                    let _ = self.rb_alloc.Reset();
                    let _ = self.rb_list.Reset(&self.rb_alloc, None);
                    open = true;
                }
                self.rb_list.CopyBufferRegion(&rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64);
                pending.push((cur, used, chunk));
                used += chunk;
                done += chunk;
            }
        }
        if open {
            flush_and_copy(self, &mut open, &mut pending, &mut used, p);
        }
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.ensure_zero_buf() {
            return;
        }
        let slot_idx = self.begin_slot();
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64);
                done += chunk;
            }
        }
        self.submit_slot(slot_idx);
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the 64 MiB ring staging buffer (zero CPU stall on the common path).
    unsafe fn upload_dirty(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() {
            return;
        }
        let Some(p) = self.staging_ptr() else { return };
        let mut open_slot: Option<usize> = None;
        let mut last_wi: Option<usize> = None;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let (aligned, chunk) = plan_ring_chunk(self.ring_write, (n - done).min(to_win_end), STAGING);
                if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                    self.reclaim_ring();
                    if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                        if let Some(slot_idx) = open_slot.take() {
                            self.submit_slot(slot_idx);
                            self.reclaim_ring();
                        }
                        while (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                            let Some(&(fv, _)) = self.inflight.front() else { break };
                            self.wait_for(fv);
                            self.reclaim_ring();
                        }
                    }
                }
                let slot_idx = match open_slot {
                    Some(idx) => idx,
                    None => {
                        let idx = self.begin_slot();
                        open_slot = Some(idx);
                        idx
                    }
                };
                let dst_off = (aligned as usize) % STAGING;
                std::ptr::copy_nonoverlapping(cur as *const u8, p.add(dst_off), chunk);
                let staging_res = &self.staging.as_ref().unwrap().0;
                self.slots[slot_idx].list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, staging_res, dst_off as u64, chunk as u64);
                self.ring_write = aligned + chunk as u64;
                done += chunk;
            }
        }
        if let Some(slot_idx) = open_slot {
            self.submit_slot(slot_idx);
        }
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    let Some(len) = u.sparse_release(p) else { return false };
    decommitted(p, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    true
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)]) {
    if ranges.is_empty() {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
    if filtered.is_empty() {
        return;
    }
    filtered.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
    for (a, n) in filtered {
        match merged.last_mut() {
            Some(last) if a <= last.0.saturating_add(last.1) => {
                let end = (last.0 + last.1).max(a + n);
                last.1 = end - last.0;
            }
            _ => merged.push((a, n)),
        }
    }
    s.rb_value += 1;
    let v = s.rb_value;
    if queue.Signal(&s.rb_fence, v).is_ok() {
        let _ = s.queue.Wait(&s.rb_fence, v);
    }
    s.readback_ranges(&merged);
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, then
/// copies the range back from VRAM into CPU memory.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    if s.window_index(va).is_none() {
        return;
    }
    let mut zeroed = std::mem::take(&mut s.to_zero);
    prune_zero_runs(&s.mapped, &mut zeroed);
    if !zeroed.is_empty() {
        for &(a, n) in &zeroed {
            s.clear_sent_range(a, n);
        }
        s.upload_zeros(&zeroed);
    }
    // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)`
    let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut pg = va & !(PAGE - 1);
    let end = va + len;
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = (t + TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                if entry.holes & (1u16 << idx) == 0 {
                    let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                    scan_page(bytes, p, &mut entry.sent[idx], &mut dirty_runs);
                }
                p += PAGE;
            }
        }
        pg = tile_end;
    }
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    s.readback_ranges(&[(va, len)]);
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.ensure_mapped(a, n, |_| 0xFFFF);
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let mut p = pg;
            while p < tile_end {
                entry.holes &= !(1u16 << ((p - t) / PAGE));
                p += PAGE;
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    let tiles = decommit_tiles(&mut s.mapped, &s.pinned, &mut s.free, &mut s.to_zero, a, n);
    if tiles.is_empty() {
        return;
    }
    s.runs = None;
    let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for &t in &tiles {
        if let Some(wi) = s.window_index(t) {
            by_window.entry(wi).or_default().push(t);
        }
    }
    for (wi, ts) in by_window {
        let (b, r) = (s.windows[wi].0, s.windows[wi].1.clone());
        s.unmap_tiles(&r, b, &ts);
    }
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    s.signal();
    s.retire_free_pools();
}

impl State {
    /// a pool tile no window maps and no buffer uses goes back to the free list
    fn release_if_unused(&mut self, tile: (u32, u32)) {
        if !self.pinned.contains_key(&tile) && !self.mapped.values().any(|e| e.pool == tile) {
            self.free.push(tile);
        }
    }
}

/// The sparse windows' tiles with video memory as sorted (start, end) runs: where a CPU write
/// has to reach the GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return Vec::new() };
    if s.runs.is_none() {
        let mut tiles: Vec<usize> = s.mapped.keys().copied().collect();
        tiles.sort_unstable();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for t in tiles {
            match runs.last_mut() {
                Some(last) if last.1 == t => last.1 = t + TILE,
                _ => runs.push((t, t + TILE)),
            }
        }
        s.runs = Some(runs);
    }
    s.runs.clone().unwrap_or_default()
}

/// time spent and pages found by the write-watch scans (`refresh_once`), for the timing line
static WATCH: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// [lock wait, chunk scan, upload] in µs, runs uploaded
static PHASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub fn note_watch(t: std::time::Duration, pages: usize) {
    WATCH[0].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
    WATCH[1].fetch_add(pages as u64, Ordering::Relaxed);
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    let t0 = std::time::Instant::now();
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let t_locked = std::time::Instant::now();
    PHASES[0].fetch_add((t_locked - t0).as_micros() as u64, Ordering::Relaxed);
    let mut zero_runs = std::mem::take(&mut s.to_zero);
    prune_zero_runs(&s.mapped, &mut zero_runs);
    for &(a, n) in &zero_runs {
        s.clear_sent_range(a, n);
    }
    let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
    dirty_runs.clear();
    let mut i = 0;
    while i < dirty.len() {
        let a = dirty[i];
        let mut n = page;
        i += 1;
        while i < dirty.len() && dirty[i] == a + n {
            n += page;
            i += 1;
        }
        // only what has video memory (committed tiles of a sparse window)
        let mut x = a;
        while x < a + n {
            let t = x & !(TILE - 1);
            let end = (t + TILE).min(a + n);
            if let Some(entry) = s.mapped.get_mut(&t) {
                let mut pg = x & !(PAGE - 1);
                while pg < end {
                    let idx = (pg - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(pg as *const u8, PAGE);
                        scan_page(bytes, pg, &mut entry.sent[idx], &mut dirty_runs);
                    }
                    pg += PAGE;
                }
            }
            x = end;
        }
    }
    let t_scanned = std::time::Instant::now();
    PHASES[1].fetch_add((t_scanned - t_locked).as_micros() as u64, Ordering::Relaxed);
    PHASES[3].fetch_add((zero_runs.len() + dirty_runs.len()) as u64, Ordering::Relaxed);
    // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
    if !zero_runs.is_empty() {
        s.upload_zeros(&zero_runs);
    }
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    PHASES[2].fetch_add(t_scanned.elapsed().as_micros() as u64, Ordering::Relaxed);
    {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static BYTES: AtomicU64 = AtomicU64::new(0);
        static MICROS: AtomicU64 = AtomicU64::new(0);
        let bytes: usize = zero_runs.iter().chain(dirty_runs.iter()).map(|r| r.1).sum();
        let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let b = BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        let us = MICROS.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed) + t0.elapsed().as_micros() as u64;
        if n.is_power_of_two() && n >= 16 {
            crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len());
            let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) / 1000;
            crate::xlog_info!(
                "sparse windows timing: write-watch {} ms ({} dirty pages), lock wait {} ms, chunk scan {} ms, upload {} ms ({} runs)",
                ms(&WATCH[0]),
                WATCH[1].load(Ordering::Relaxed),
                ms(&PHASES[0]),
                ms(&PHASES[1]),
                ms(&PHASES[2]),
                PHASES[3].load(Ordering::Relaxed)
            );
        }
    }
    dirty_runs.clear();
    s.scratch_runs = dirty_runs;
    let q = queue.as_raw() as usize;
    if s.value > 0 && s.waited.get(&q).copied().unwrap_or(0) < s.value {
        let _ = queue.Wait(&s.fence, s.value);
        s.waited.insert(q, s.value);
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let mut g = STATE.lock().unwrap();
    let s = g.as_mut()?;
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    s.window_index(start)?;
    s.window_index(va.checked_add(bytes)?.checked_sub(1)?)?;
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    let hr = super::real::dev_create_reserved_resource(this, &d as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        return None;
    }
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes
    s.ensure_mapped(start, delta + bytes, |t| query_tile_holes(t));
    let tiles: Vec<(usize, (u32, u32))> = (start..va + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|e| (t, e.pool))).collect();
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
    s.map_tiles(&res, start, &tiles);
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles.iter().map(|x| x.1).collect());
    s.signal();
    Some((hr, delta))
}

/// A placed buffer is gone: its tiles are not pinned by it any more.
pub fn forget_buffer(res: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for tile in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    /// `scan_page` skips allocating a hash box on all-zero pages, allocates on the first write,
    /// and coalesces adjacent dirty chunks across consecutive pages.
    #[test]
    fn scan_page_lazy_alloc_and_coalesce() {
        let mut page = vec![0u8; PAGE];
        let mut slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_some());
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let mut next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &mut next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` / `sync_sent_range` do not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| query_tile_holes(t));
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            s.readback_ranges(&[(base, 2 * TILE)]);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &mut entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: Vec<(usize, usize)> = Vec::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        prune_zero_runs(&mapped, &mut to_zero);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        prune_zero_runs(&mapped, &mut to_zero);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        prune_zero_runs(&mapped, &mut to_zero);
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, vec![(t4, PAGE)]);
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, vec![(t4, PAGE)], "unrelated mapped tile's to_zero must remain");
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }
}
