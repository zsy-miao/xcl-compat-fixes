//! Console unified memory (UMA). On the console CPU and GPU share memory and shaders reach
//! it through plain addresses in SGPRs. XCL keeps the title's graphics memory in regions
//! that D3D12 can open as heaps (`ID3D12Device3::OpenExistingHeapFromAddress`), so the GPU
//! works on them directly.
//!
//! A heap is smaller than 4 GiB, so the memory is up to 4 *windows*: window k is one region
//! at `base + k · 4 GiB` with `base` 16 GiB-aligned. The window of a console GPU address is
//! then `(address >> 32) & 3` and its low 32 bits are the offset in that window's heap – the
//! recompiler's `mem` (window 0), `mem1` … buffers. Allocations never straddle windows.
//!
//! Measured on Windows 11 + RX 9070 (2026-11-20): the heap must cover a fully committed
//! `VirtualAlloc` allocation (a reservation with partial commit is rejected); up to
//! 4095 MiB works (exactly 4 GiB is rejected), opening 4 GiB takes ~0.5 s.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

/// VirtualAlloc: track CPU writes (GetWriteWatch)
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// allocation granularity (as VirtualAlloc)
pub const GRAIN: usize = 0x1_0000;
/// largest region D3D12 opens as a heap
pub const MAX_SIZE: usize = (4096 << 20) - GRAIN;
/// distance between windows
pub const WINDOW: usize = 1 << 32;
pub const MAX_WINDOWS: usize = 4;

/// bytes of sparse window 0 only 32-bit descriptor heaps get (`Uma::desc_range`)
const DESC_HEADROOM: usize = 512 << 20;
/// cap on the eager committed coherent window in `for_runtime` (per-draw args, readback buffers)
const RUNTIME_COMMITTED_CAP: usize = 512 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WindowKind {
    None,
    Committed(usize),
    Sparse,
}

pub struct Uma {
    base: usize,
    kinds: [WindowKind; MAX_WINDOWS],
    /// VirtualAlloc base pointers owned by this Uma (freed on Drop)
    win_allocs: Vec<usize>,
    inner: Mutex<Inner>,
    /// free ranges of the sparse windows (address → length) and the reservations in them
    sparse_free: Mutex<(BTreeMap<usize, usize>, BTreeMap<usize, usize>)>,
    /// XCL's own pool in window 0 for `Arena` blocks: (start, end, free blocks). The title
    /// cannot take it – a title that commits all of unified memory at start (Unicorn
    /// Overlord: 2046 of 2048 MB) left no room for draw arguments, and draws were skipped.
    pool: Mutex<(usize, usize, Vec<usize>)>,
}

/// bytes of XCL's arena pool (`Uma::new`)
pub const POOL: usize = 64 << 20;

#[derive(Default)]
struct Inner {
    /// free ranges: offset → length (coalesced)
    free: BTreeMap<usize, usize>,
    /// live allocations: offset → length
    used: BTreeMap<usize, usize>,
}

/// Carve an `align`-aligned block of `size` bytes from `free` within `[lo_bound, hi_bound)`.
/// When `single_window` is true and `size <= WINDOW`, the returned block never straddles a
/// 4 GiB window boundary.
fn take_sparse_in(
    g: &mut (BTreeMap<usize, usize>, BTreeMap<usize, usize>),
    lo_bound: usize,
    hi_bound: usize,
    size: usize,
    align: usize,
    single_window: bool,
) -> Option<usize> {
    if lo_bound >= hi_bound {
        return None;
    }
    let found = g.0.iter().find_map(|(&o, &l)| {
        let lo = o.max(lo_bound);
        let hi = o.saturating_add(l).min(hi_bound);
        if lo >= hi {
            return None;
        }
        let mut at = (lo.checked_add(align - 1)?) & !(align - 1);
        while at.checked_add(size).is_some_and(|e| e <= hi) {
            if !single_window || size > WINDOW || at / WINDOW == (at + size - 1) / WINDOW {
                return Some((o, l, at));
            }
            let next_win = (at / WINDOW + 1).checked_mul(WINDOW)?;
            at = (next_win.checked_add(align - 1)?) & !(align - 1);
        }
        None
    });
    let (start, len, at) = found?;
    g.0.remove(&start);
    if at > start {
        g.0.insert(start, at - start);
    }
    if at + size < start + len {
        g.0.insert(at + size, start + len - (at + size));
    }
    g.1.insert(at, size);
    Some(at)
}

impl Uma {
    /// Commit `size` bytes (split into as few equal windows as fit, at most 4) at the first
    /// free 16 GiB-aligned address from 16 GiB up, with remaining windows reserved as sparse.
    pub fn new(size: usize) -> Option<Uma> {
        let size = ((size + GRAIN - 1) & !(GRAIN - 1)).max(GRAIN);
        let n = size.div_ceil(MAX_SIZE).clamp(1, MAX_WINDOWS);
        let mut u = Uma::with_windows(n, (size.div_ceil(n) + GRAIN - 1) & !(GRAIN - 1))?;
        u.reserve_sparse_windows();
        u.reserve_pool(POOL);
        Some(u)
    }

    /// XCL's arena pool: `bytes` at the end of the first committed window taken now, before
    /// the title can (a quarter of that window at most); the title's allocations still start
    /// at the window's start.
    pub fn reserve_pool(&self, bytes: usize) {
        let Some(win) = self.kinds.iter().position(|w| matches!(w, WindowKind::Committed(_))) else { return };
        let win_size = self.window(win).1;
        let bytes = (bytes.min(win_size / 4) / ARENA_BLOCK) * ARENA_BLOCK;
        if bytes == 0 {
            return;
        }
        let at = {
            let mut g = self.inner.lock().unwrap();
            // the free range of the committed window that reaches furthest, its last `bytes`
            let Some((start, len)) = g.free.iter().filter(|(&o, &l)| o / WINDOW == win && l >= bytes).max_by_key(|(&o, &l)| o + l).map(|(&o, &l)| (o, l)) else { return };
            let at = (start + len - bytes) & !(GRAIN - 1);
            if at < start {
                return;
            }
            g.free.remove(&start);
            if at > start {
                g.free.insert(start, at - start);
            }
            if at + bytes < start + len {
                g.free.insert(at + bytes, start + len - at - bytes);
            }
            g.used.insert(at, bytes);
            at
        };
        let p = self.base + at;
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
        *self.pool.lock().unwrap() = (p, p + bytes, (p..p + bytes).step_by(ARENA_BLOCK).rev().collect());
    }

    /// A block of `ARENA_BLOCK` bytes from XCL's pool (None: the pool is used up or absent).
    fn pool_block(&self) -> Option<usize> {
        self.pool.lock().unwrap().2.pop()
    }

    /// Runtime layout: windows 0..3 (12 GiB) reserved as write-watched sparse VRAM windows
    /// (window 0 holds 32-bit-addressed descriptor heaps + buffers + XMem, windows 1..2 hold
    /// large title reservations and overflow), and window 3 as a smaller committed coherent
    /// window (`OpenExistingHeapFromAddress`) for per-draw arguments and CPU readback buffers.
    pub fn for_runtime(size: usize) -> Option<Uma> {
        Self::for_runtime_with(size, RUNTIME_COMMITTED_CAP)
    }

    /// `for_runtime` with a coherent window of up to `coherent` bytes (`coherent_hint`).
    pub fn for_runtime_with(size: usize, coherent: usize) -> Option<Uma> {
        let committed = ((size.min(coherent.max(RUNTIME_COMMITTED_CAP)) + GRAIN - 1) & !(GRAIN - 1)).clamp(GRAIN, MAX_SIZE);
        let sparse_len = (MAX_WINDOWS - 1) * WINDOW;
        for slot in (4usize..512).step_by(MAX_WINDOWS) {
            let base = slot << 32;
            let sp = unsafe { VirtualAlloc(base as *const std::ffi::c_void, sparse_len, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) };
            if sp as usize != base {
                if !sp.is_null() {
                    unsafe { VirtualFree(sp, 0, MEM_RELEASE) };
                }
                continue;
            }
            let c_base = base + (MAX_WINDOWS - 1) * WINDOW;
            let cp = unsafe { VirtualAlloc(c_base as *const std::ffi::c_void, committed, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
            if cp as usize != c_base {
                if !cp.is_null() {
                    unsafe { VirtualFree(cp, 0, MEM_RELEASE) };
                }
                unsafe { VirtualFree(sp, 0, MEM_RELEASE) };
                continue;
            }
            let mut inner = Inner::default();
            inner.free.insert((MAX_WINDOWS - 1) * WINDOW, committed);
            let mut sparse_free: (BTreeMap<usize, usize>, BTreeMap<usize, usize>) = Default::default();
            sparse_free.0.insert(base, sparse_len);
            let u = Uma {
                base,
                kinds: [WindowKind::Sparse, WindowKind::Sparse, WindowKind::Sparse, WindowKind::Committed(committed)],
                win_allocs: vec![base, c_base],
                inner: Mutex::new(inner),
                sparse_free: Mutex::new(sparse_free),
                pool: Mutex::new((0, 0, Vec::new())),
            };
            u.reserve_pool(POOL);
            return Some(u);
        }
        Uma::new(size)
    }

    /// The windows after the committed ones as one write-watched reservation (sparse windows).
    fn reserve_sparse_windows(&mut self) {
        let committed = self.windows();
        if committed >= MAX_WINDOWS {
            return;
        }
        let start = self.base + committed * WINDOW;
        let len = (MAX_WINDOWS - committed) * WINDOW;
        let p = unsafe { VirtualAlloc(start as *const std::ffi::c_void, len, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) };
        if p as usize == start {
            for k in committed..MAX_WINDOWS {
                self.kinds[k] = WindowKind::Sparse;
            }
            self.win_allocs.push(start);
            self.sparse_free.lock().unwrap().0.insert(start, len);
        } else if !p.is_null() {
            unsafe { VirtualFree(p, 0, MEM_RELEASE) };
        }
    }

    /// The range of sparse window 0 kept for 32-bit descriptor heaps (`sparse_alloc_in(0)`):
    /// after the window's first 64 KiB, which stay unused (a console address of 0 reads as
    /// null). The title's memory never takes it, and it sits at the start of the window so the
    /// title still gets one contiguous range of 11.5 GiB: XDefiant reserves 8 GiB + 4 MiB first
    /// thing, more than windows 1..2 hold, and with the range at the top of window 0 that
    /// reservation took all of it – no descriptor heap could be made, every frame was black.
    fn desc_range(&self) -> Option<(usize, usize)> {
        self.is_sparse_window(0).then(|| (self.base + GRAIN, self.base + GRAIN + DESC_HEADROOM))
    }

    /// The first sparse address the title's memory may use (after `desc_range`).
    fn title_sparse_start(&self) -> usize {
        self.desc_range().map_or(self.base, |(_, end)| end)
    }

    /// Whether window `k` is a sparse VRAM window.
    pub fn is_sparse_window(&self, k: usize) -> bool {
        self.kinds.get(k).is_some_and(|w| matches!(w, WindowKind::Sparse))
    }

    /// Whether window `k` is a committed coherent window.
    pub fn is_committed_window(&self, k: usize) -> bool {
        self.kinds.get(k).is_some_and(|w| matches!(w, WindowKind::Committed(_)))
    }

    /// Whether this Uma has any sparse VRAM windows.
    pub fn has_sparse(&self) -> bool {
        self.kinds.iter().any(|w| matches!(w, WindowKind::Sparse))
    }

    /// Windows translated shaders address (`mem`, `mem1` …): highest active window + 1.
    pub fn shader_windows(&self) -> usize {
        self.kinds.iter().rposition(|w| !matches!(w, WindowKind::None)).map_or(0, |i| i + 1)
    }

    /// Whether `p` is in a sparse window.
    pub fn is_sparse(&self, p: usize) -> bool {
        self.window_of(p).is_some_and(|k| self.is_sparse_window(k))
    }

    /// Whether `p` is in a committed window.
    pub fn is_committed(&self, p: usize) -> bool {
        self.window_of(p).is_some_and(|k| self.is_committed_window(k))
    }

    /// Whether `[p, p + len)` lies in a committed coherent window – memory that is committed and
    /// read-write as a whole, so it needs no `VirtualQuery` (which walks a region to its end to
    /// size it: on a 4 GiB window ~40 ms a call, a retail game's DirectStorage requests crawled).
    pub fn in_committed_window(&self, p: usize, len: usize) -> bool {
        len > 0 && self.is_committed(p) && p.checked_add(len - 1).is_some_and(|e| self.window_of(e) == self.window_of(p))
    }

    /// Address space for a title reservation of `size` bytes in the sparse windows (64 KiB
    /// aligned, first fit; not committed). Prefers windows 1+ so window 0 stays uncluttered for
    /// small allocations, falling back to all of the title's sparse space (never `desc_range`).
    pub fn sparse_reserve(&self, size: usize) -> Option<usize> {
        if !self.has_sparse() {
            return None;
        }
        let size = (size.max(1) + GRAIN - 1) & !(GRAIN - 1);
        let mut g = self.sparse_free.lock().unwrap();
        let end = self.base + MAX_WINDOWS * WINDOW;
        let lo = self.title_sparse_start();
        if let Some(at) = take_sparse_in(&mut g, (self.base + WINDOW).max(lo), end, size, GRAIN, false) {
            return Some(at);
        }
        take_sparse_in(&mut g, lo, end, size, GRAIN, false)
    }

    /// Reserve an aligned range strictly inside sparse window `k` (window 0: 32-bit descriptor
    /// heaps, from `desc_range` first, then whatever of the window is free). Returns
    /// `(address, rounded_size)`.
    pub fn sparse_alloc_in(&self, k: usize, size: usize, align: usize) -> Option<(usize, usize)> {
        if !self.is_sparse_window(k) {
            return None;
        }
        let size = (size.max(1) + GRAIN - 1) & !(GRAIN - 1);
        let align = align.max(GRAIN).next_power_of_two();
        let (w_start, w_len) = self.window(k);
        let mut g = self.sparse_free.lock().unwrap();
        let desc = if k == 0 { self.desc_range() } else { None };
        if let Some((lo, hi)) = desc {
            if let Some(at) = take_sparse_in(&mut g, lo, hi, size, align, true) {
                return Some((at, size));
            }
        }
        let lo = desc.map_or(w_start, |(lo, _)| lo);
        take_sparse_in(&mut g, lo, w_start + w_len, size, align, true).map(|at| (at, size))
    }

    /// Reserve an aligned range in the sparse windows that never straddles a 4 GiB window
    /// boundary (when `size <= WINDOW`): window 0 first, then windows 1+, never `desc_range`.
    /// Returns `(address, rounded_size)`.
    pub fn sparse_alloc(&self, size: usize, align: usize) -> Option<(usize, usize)> {
        if !self.has_sparse() {
            return None;
        }
        let size = size.max(1).checked_next_multiple_of(GRAIN)?;
        let align = align.max(GRAIN).checked_next_power_of_two()?;
        let mut g = self.sparse_free.lock().unwrap();
        let end = self.base + MAX_WINDOWS * WINDOW;
        let lo = self.title_sparse_start();
        let above_0 = (self.base + WINDOW).max(lo);
        if size <= WINDOW {
            if self.is_sparse_window(0) {
                if let Some(at) = take_sparse_in(&mut g, lo, self.base + WINDOW, size, align, true) {
                    return Some((at, size));
                }
            }
            take_sparse_in(&mut g, above_0, end, size, align, true).map(|at| (at, size))
        } else {
            take_sparse_in(&mut g, above_0, end, size, align, false)
                .or_else(|| take_sparse_in(&mut g, lo, end, size, align, false))
                .map(|at| (at, size))
        }
    }

    /// The bytes of the sparse reservation starting at `p` (it is gone after this).
    pub fn sparse_release(&self, p: usize) -> Option<usize> {
        let mut g = self.sparse_free.lock().unwrap();
        let size = g.1.remove(&p)?;
        let (mut start, mut len) = (p, size);
        if let Some((&o, &l)) = g.0.range(..p).next_back() {
            if o + l == p {
                g.0.remove(&o);
                start = o;
                len += l;
            }
        }
        if let Some(l) = g.0.remove(&(p + size)) {
            len += l;
        }
        g.0.insert(start, len);
        Some(size)
    }

    /// The sparse reservation holding `p`: (start, bytes).
    pub fn sparse_reservation(&self, p: usize) -> Option<(usize, usize)> {
        let g = self.sparse_free.lock().unwrap();
        g.1.range(..=p).next_back().map(|(&a, &l)| (a, l)).filter(|&(a, l)| p < a + l)
    }

    /// Bytes currently reserved/allocated in sparse windows.
    pub fn sparse_used(&self) -> usize {
        self.sparse_free.lock().unwrap().1.values().sum()
    }

    /// `windows` committed windows of `size` bytes each.
    pub fn with_windows(windows: usize, size: usize) -> Option<Uma> {
        let size = ((size + GRAIN - 1) & !(GRAIN - 1)).clamp(GRAIN, MAX_SIZE);
        let windows = windows.clamp(1, MAX_WINDOWS);
        for slot in (4usize..512).step_by(MAX_WINDOWS) {
            let base = slot << 32;
            let mut got = Vec::new();
            for k in 0..windows {
                let want = base + k * WINDOW;
                // write-watched: textures the title keeps here are uploaded again after CPU writes
                let p = unsafe { VirtualAlloc(want as *const std::ffi::c_void, size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
                if p.is_null() {
                    break;
                }
                got.push(p);
            }
            if got.len() == windows {
                let mut inner = Inner::default();
                let mut kinds = [WindowKind::None; MAX_WINDOWS];
                let mut win_allocs = Vec::with_capacity(windows + 1);
                for k in 0..windows {
                    inner.free.insert(k * WINDOW, size);
                    kinds[k] = WindowKind::Committed(size);
                    win_allocs.push(got[k] as usize);
                }
                return Some(Uma {
                    base,
                    kinds,
                    win_allocs,
                    inner: Mutex::new(inner),
                    sparse_free: Mutex::new(Default::default()),
                    pool: Mutex::new((0, 0, Vec::new())),
                });
            }
            for p in got {
                unsafe { VirtualFree(p, 0, MEM_RELEASE) };
            }
        }
        None
    }

    pub fn base(&self) -> usize {
        self.base
    }

    /// usable bytes across all committed windows
    pub fn size(&self) -> usize {
        self.kinds
            .iter()
            .map(|w| match *w {
                WindowKind::Committed(s) => s,
                _ => 0,
            })
            .sum()
    }

    /// number of committed windows
    pub fn windows(&self) -> usize {
        self.kinds.iter().filter(|w| matches!(w, WindowKind::Committed(_))).count()
    }

    /// (address, bytes) of window `k` (a sparse window: all 4 GiB)
    pub fn window(&self, k: usize) -> (usize, usize) {
        let len = match self.kinds.get(k).copied().unwrap_or(WindowKind::None) {
            WindowKind::Committed(s) => s,
            WindowKind::Sparse => WINDOW,
            WindowKind::None => 0,
        };
        (self.base + k * WINDOW, len)
    }

    /// Underlying write-watched `VirtualAlloc` regions `(base, size)` for `GetWriteWatch`:
    /// contiguous sparse windows belong to a single reservation, while each committed window
    /// is its own allocation.
    pub fn watch_regions(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(MAX_WINDOWS);
        let mut k = 0;
        while k < MAX_WINDOWS {
            match self.kinds[k] {
                WindowKind::None => k += 1,
                WindowKind::Committed(s) => {
                    out.push((self.base + k * WINDOW, s));
                    k += 1;
                }
                WindowKind::Sparse => {
                    let start = self.base + k * WINDOW;
                    let mut len = 0usize;
                    while k < MAX_WINDOWS && matches!(self.kinds[k], WindowKind::Sparse) {
                        len += WINDOW;
                        k += 1;
                    }
                    out.push((start, len));
                }
            }
        }
        out
    }

    pub fn window_of(&self, p: usize) -> Option<usize> {
        let k = p.checked_sub(self.base)? / WINDOW;
        let (start, len) = self.window(k);
        (k < MAX_WINDOWS && len > 0 && p - start < len).then_some(k)
    }

    pub fn contains(&self, p: usize) -> bool {
        self.window_of(p).is_some()
    }

    /// Whether `p` is memory the title has here: a block handed out in a committed window, or
    /// one of its reservations in a sparse window. The windows' ranges are wide (4 GiB each) and
    /// a PC resource's GPU address can fall in one by number – that address is not the
    /// window's memory (Minecraft's dynamic vertex buffer, a PC upload buffer, was read from an
    /// empty sparse window: nothing drew).
    pub fn is_title_address(&self, p: usize) -> bool {
        if self.is_sparse(p) {
            return self.sparse_reservation(p).is_some();
        }
        if !self.is_committed(p) {
            return false;
        }
        let off = p - self.base;
        let g = self.inner.lock().unwrap();
        g.used.range(..=off).next_back().is_some_and(|(&o, &l)| off < o + l)
    }

    /// Zeroed block of `size` bytes aligned to `align` (at least 64 KiB), first fit (window 0
    /// first).
    #[track_caller]
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        self.alloc_where(size, align, None, true)
    }

    /// `alloc` for reserved address space: not zeroed (a large reservation is not touched –
    /// what the title commits or maps there it writes; never-used memory is zero anyway).
    #[track_caller]
    pub fn reserve(&self, size: usize, align: usize) -> Option<*mut u8> {
        self.alloc_where(size, align, None, false)
    }

    /// `alloc` inside window `k` (memory the console addresses with 32 bits, e.g. descriptor
    /// heaps: window 0).
    #[track_caller]
    pub fn alloc_in(&self, k: usize, size: usize, align: usize) -> Option<*mut u8> {
        self.alloc_where(size, align, Some(k), true)
    }

    /// `XCL_TRACE_MEM`: each committed-window allocation of 1 MB or more and each one that
    /// fails, with the XCL code that asked for it (what fills the window).
    #[track_caller]
    fn trace(&self, size: usize, got: Option<usize>, used: usize) {
        let at = std::panic::Location::caller();
        match got {
            Some(p) => crate::xlog_info!("UMA: {} KB at {p:#x} for {}:{} ({} MB used)", size >> 10, at.file(), at.line(), used >> 20),
            None => crate::xlog_info!("UMA: {} KB for {}:{} – no room ({} MB used)", size >> 10, at.file(), at.line(), used >> 20),
        }
    }

    fn tracing() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("XCL_TRACE_MEM").is_some())
    }

    #[track_caller]
    fn alloc_where(&self, size: usize, align: usize, window: Option<usize>, zero: bool) -> Option<*mut u8> {
        let size = size.max(1).checked_next_multiple_of(GRAIN)?;
        let align = align.max(GRAIN).checked_next_power_of_two()?;
        let mut g = self.inner.lock().unwrap();
        let found = g.free.iter().filter(|(&o, _)| window.is_none_or(|k| o / WINDOW == k)).find_map(|(&o, &l)| {
            let at = self.base.checked_add(o)?.checked_next_multiple_of(align)? - self.base;
            (at.checked_add(size).is_some_and(|end| end <= o + l)).then_some((o, l, at))
        });
        let Some((start, len, at)) = found else {
            if Self::tracing() {
                let used = g.used.values().sum();
                drop(g);
                self.trace(size, None, used);
            }
            return None;
        };
        g.free.remove(&start);
        if at > start {
            g.free.insert(start, at - start);
        }
        if at + size < start + len {
            g.free.insert(at + size, start + len - at - size);
        }
        g.used.insert(at, size);
        let used = if Self::tracing() && size >= 1 << 20 { Some(g.used.values().sum()) } else { None };
        drop(g);
        let p = (self.base + at) as *mut u8;
        if let Some(used) = used {
            self.trace(size, Some(p as usize), used);
        }
        if zero {
            unsafe { std::ptr::write_bytes(p, 0, size) };
        }
        Some(p)
    }

    /// Release a block from `alloc` (false: not one of ours).
    pub fn free(&self, p: usize) -> bool {
        if !self.is_committed(p) {
            return false;
        }
        {
            // a block of XCL's pool goes back to the pool
            let mut g = self.pool.lock().unwrap();
            if p >= g.0 && p < g.1 {
                g.2.push(p);
                return true;
            }
        }
        let mut g = self.inner.lock().unwrap();
        let off = p - self.base;
        let Some(orig) = g.used.remove(&off) else { return false };
        let (mut start, mut len) = (off, orig);
        // merge with the free neighbours
        if let Some((&o, &l)) = g.free.range(..off).next_back() {
            if o + l == off {
                g.free.remove(&o);
                start = o;
                len += l;
            }
        }
        if let Some(l) = g.free.remove(&(off + orig)) {
            len += l;
        }
        g.free.insert(start, len);
        true
    }

    /// Bytes handed out.
    pub fn used(&self) -> usize {
        self.inner.lock().unwrap().used.values().sum()
    }
}

impl Drop for Uma {
    fn drop(&mut self) {
        for &p in &self.win_allocs {
            unsafe {
                VirtualFree(p as *mut std::ffi::c_void, 0, MEM_RELEASE);
            }
        }
    }
}

/// Small pieces of committed unified memory for XCL's own per-draw data (argument tables,
/// user SGPRs): a block of `ARENA_BLOCK` bytes cut into aligned pieces, a new block when one
/// is used up. `Uma::alloc` hands out whole 64 KiB grains – three of those per draw filled
/// 2 GB within seconds in a scene of thousands of draws (Minecraft's world), and the draws
/// that found no room were skipped (the world flickered). The blocks go to `blocks`; their
/// owner frees them when the GPU is done (a command allocator's reset) and starts a fresh
/// arena.
#[derive(Default)]
pub struct Arena {
    at: usize,
    end: usize,
}

/// block size of an `Arena`
pub const ARENA_BLOCK: usize = 4 * GRAIN;

impl Arena {
    /// `n` bytes at an `align`-aligned address in committed unified memory (not zeroed), or
    /// None when committed unified memory is full.
    pub fn take(&mut self, u: &Uma, n: usize, align: usize, blocks: &mut Vec<usize>) -> Option<*mut u8> {
        let align = align.max(16).next_power_of_two();
        let n = n.max(1);
        let at = (self.at + align - 1) & !(align - 1);
        if self.at != 0 && at + n <= self.end {
            self.at = at + n;
            return Some(at as *mut u8);
        }
        if n > ARENA_BLOCK / 2 {
            // a large piece: a block of its own (the current one stays in use)
            let p = u.alloc(n, align)?;
            blocks.push(p as usize);
            return Some(p);
        }
        // XCL's pool first, then any free part of committed unified memory
        let p = match u.pool_block() {
            Some(p) => p,
            None => u.alloc(ARENA_BLOCK, GRAIN)? as usize,
        };
        blocks.push(p);
        self.at = p + n;
        self.end = p + ARENA_BLOCK;
        Some(p as *mut u8)
    }
}

static GLOBAL: OnceLock<Option<Uma>> = OnceLock::new();

/// The title's graphics memory XCL places in the coherent window (`note_coherent_need`) is
/// known only once it asks for it, and the window is committed when it is made: its size is
/// what the title asked for in its last run (`<data>/cache/uma_coherent.txt`) and XCL's own
/// 512 MB, at most a window. `XCL_COHERENT_MB` chooses it.
fn coherent_file() -> std::path::PathBuf {
    crate::state::data_dir().join("cache").join("uma_coherent.txt")
}

/// Bytes the coherent window is made with.
pub fn coherent_hint() -> usize {
    if let Some(mb) = std::env::var("XCL_COHERENT_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
        return (mb << 20).clamp(GRAIN, MAX_SIZE);
    }
    let title = std::fs::read_to_string(coherent_file()).ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0);
    // only with the commit for all of it: the title's ~13 GB, video memory for its render targets
    // and textures (WDDM charges commit for each video-memory allocation too) and the window
    // itself – a retail game took ~18 GB; with less free it runs without (its GPU heaps outside
    // unified memory: no image) rather than taking the system's commit
    if title > 0 && free_commit() < NEEDS_FOR_COHERENT_HEAPS {
        crate::xlog_warn!(
            "UMA: {} MB of commit free, under {} MB: the title's GPU heaps stay outside unified memory (a larger page file gives them the coherent window)",
            free_commit() >> 20,
            NEEDS_FOR_COHERENT_HEAPS >> 20
        );
        return RUNTIME_COMMITTED_CAP;
    }
    (title + RUNTIME_COMMITTED_CAP).min(MAX_SIZE)
}

/// Commit free before the title starts for its GPU heaps to go to the coherent window.
const NEEDS_FOR_COHERENT_HEAPS: usize = 17 << 30;

fn free_commit() -> usize {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return usize::MAX;
    }
    m.ullAvailPageFile as usize
}

/// The title has asked for `bytes` of graphics memory XCL places in the coherent window: the
/// next run makes the window that large (this one has the size it was made with).
pub fn note_coherent_need(bytes: usize) {
    static NOTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let before = std::fs::read_to_string(coherent_file()).ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0);
    if bytes > before && NOTED.fetch_max(bytes, std::sync::atomic::Ordering::Relaxed) < bytes {
        let f = coherent_file();
        if let Some(d) = f.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(&f, bytes.to_string());
        crate::xlog_info!("UMA: the title's graphics memory for the coherent window: {} MB (a window of up to {} MB from the next run)", bytes >> 20, (bytes + RUNTIME_COMMITTED_CAP).min(MAX_SIZE) >> 20);
    }
}

impl Uma {
    /// The committed coherent window (None: none).
    pub fn committed_window(&self) -> Option<usize> {
        (0..MAX_WINDOWS).find(|&k| self.is_committed_window(k))
    }
}

/// The title's graphics memory (created on first use; `XCL_UMA_MB`, default 2048).
pub fn get() -> Option<&'static Uma> {
    GLOBAL
        .get_or_init(|| {
            let mb = std::env::var("XCL_UMA_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
            let coherent = coherent_hint();
            let u = Uma::for_runtime_with((mb << 20).max(coherent), coherent);
            match &u {
                Some(u) => {
                    let sparse_n = (0..MAX_WINDOWS).filter(|&k| u.is_sparse_window(k)).count();
                    crate::xlog_info!(
                        "UMA: {} sparse VRAM window(s) ({} GiB VA) + {} MB coherent window at {:#x}",
                        sparse_n,
                        sparse_n * 4,
                        u.size() >> 20,
                        u.base()
                    );
                }
                None => crate::xlog_warn!("UMA: could not reserve {mb} MB at a 4 GiB-aligned address – plain allocations"),
            }
            u
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The retail game's black image (HANDOFF "Retail game bring-up"): ~5.7 GB of physical pages
    /// whose homes took sparse-window space, then the title's own 6 GiB reservation they are
    /// mapped into – both did not fit the windows' 11.5 GiB, and the reservation went to
    /// plain Windows memory the GPU does not see. With homes in CPU memory
    /// (`kernelx::AllocateTitlePhysicalPages`) the reservation fits.
    #[test]
    fn a_6_gib_reservation_fits_once_page_homes_take_no_window_space() {
        let Some(u) = Uma::for_runtime(64 << 20) else { return };
        let homes = 5_700_000_000usize.next_multiple_of(GRAIN);
        let (h, _) = u.sparse_alloc(homes, GRAIN).expect("the homes alone fit");
        assert!(u.sparse_reserve(6 << 30).is_none(), "homes in the windows: no room for the reservation");
        u.sparse_release(h);
        let r = u.sparse_reserve(6 << 30).expect("without the homes the reservation fits");
        assert!(u.is_sparse(r) && u.is_sparse(r + (6 << 30) - 1));
    }

    /// Arena blocks come from XCL's pool while the title holds the rest of window 0, and go
    /// back to it when freed.
    #[test]
    fn arena_blocks_come_from_the_pool_when_the_title_took_the_rest() {
        let u = Uma::with_windows(1, 8 << 20).unwrap();
        u.reserve_pool(4 * ARENA_BLOCK);
        // the pool is the window's end: the title's first allocation is still its start
        assert_eq!(u.pool.lock().unwrap().1, u.base() + (8 << 20));
        assert_eq!(u.alloc(GRAIN, GRAIN).map(|p| p as usize), Some(u.base()));
        // the title takes everything else
        let mut title = Vec::new();
        while let Some(p) = u.alloc(GRAIN, GRAIN) {
            title.push(p as usize);
        }
        assert!(u.alloc_in(0, ARENA_BLOCK, GRAIN).is_none());
        let (mut a, mut blocks) = (Arena::default(), Vec::new());
        for _ in 0..4 * (ARENA_BLOCK / 256) {
            a.take(&u, 200, 256, &mut blocks).expect("a piece from the pool");
        }
        assert_eq!(blocks.len(), 4);
        assert!(a.take(&u, 200, 256, &mut blocks).is_none(), "pool used up, the rest is the title's");
        for b in blocks.drain(..) {
            assert!(u.free(b));
        }
        let mut a = Arena::default();
        assert!(a.take(&u, 200, 256, &mut blocks).is_some(), "freed blocks are back in the pool");
    }

    /// Many small pieces share a block (aligned, disjoint); a large one gets its own.
    #[test]
    fn arena_cuts_blocks_into_aligned_pieces() {
        let u = Uma::with_windows(1, 4 << 20).unwrap();
        let (mut a, mut blocks) = (Arena::default(), Vec::new());
        let mut got = Vec::new();
        for k in 0..1000usize {
            let n = 16 + (k % 7) * 40;
            let p = a.take(&u, n, 256, &mut blocks).unwrap() as usize;
            assert_eq!(p % 256, 0);
            assert_eq!(u.window_of(p), Some(0));
            got.push((p, n));
        }
        got.sort_unstable();
        assert!(got.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0), "pieces overlap");
        // 1000 pieces of at most 256 aligned bytes: 1000 · 256 B = 4 blocks, not 1000 grains
        assert_eq!(blocks.len(), (1000 * 256usize).div_ceil(ARENA_BLOCK));
        let big = a.take(&u, ARENA_BLOCK, 256, &mut blocks).unwrap() as usize;
        assert_eq!(blocks.last(), Some(&big));
        for b in blocks {
            assert!(u.free(b));
        }
        assert_eq!(u.used(), 0);
    }

    #[test]
    fn windows_are_4_gib_apart_and_allocations_stay_inside() {
        let u = Uma::with_windows(2, 4 << 20).unwrap();
        assert_eq!(u.base() % (MAX_WINDOWS * WINDOW), 0, "16 GiB alignment");
        assert_eq!((u.windows(), u.size()), (2, 8 << 20));
        let a = u.alloc(3 << 20, 0).unwrap() as usize;
        let b = u.alloc(3 << 20, 0).unwrap() as usize; // no room left in window 0
        assert_eq!(u.window_of(a), Some(0));
        assert_eq!(u.window_of(b), Some(1));
        assert_eq!(b, u.base() + WINDOW);
        assert_eq!((b >> 32) & 3, 1, "window from the high address bits");
        assert!(u.alloc_in(0, 2 << 20, 0).is_none(), "window 0 full");
        assert!(u.alloc(2 << 20, 0).is_none());
        unsafe { *(b as *mut u32) = 7 };
        assert!(u.free(a) && u.free(b));
        assert!(!u.contains(u.base() + (4 << 20)) && !u.contains(u.base() + 2 * WINDOW));
    }

    #[test]
    fn alloc_free_coalesce() {
        // without XCL's arena pool (`Uma::new` takes it at the start of window 0)
        let u = Uma::with_windows(1, 16 << 20).unwrap();
        assert_eq!(u.base() & 0xFFFF_FFFF, 0, "4 GiB alignment");
        let a = u.alloc(100, 0).unwrap() as usize;
        let b = u.alloc(0x2_0000, 0).unwrap() as usize;
        let c = u.alloc(0x1_0000, 0x10_0000).unwrap() as usize;
        assert_eq!(a, u.base());
        assert_eq!(b, u.base() + GRAIN);
        assert_eq!(c % 0x10_0000, 0, "alignment");
        assert_eq!(u.used(), 4 * GRAIN);
        unsafe { *(b as *mut u32) = 0xDEAD };
        assert!(u.free(b) && !u.free(b) && !u.free(12345));
        assert!(u.free(a));
        // a + b coalesced: a 3-grain block fits at the base again, zeroed
        let d = u.alloc(3 * GRAIN, 0).unwrap() as usize;
        assert_eq!(d, u.base());
        assert_eq!(unsafe { *((d + GRAIN) as *const u32) }, 0);
        assert!(u.free(c) && u.free(d));
        assert_eq!(u.used(), 0);
        // everything merged back into one range
        assert!(u.alloc(16 << 20, 0).is_some());
    }

    /// Only handed-out blocks are the title's memory: an address in a window's range that is
    /// not one (a PC GPU address by number) is not.
    #[test]
    fn title_addresses_are_blocks_handed_out() {
        let u = Uma::new(16 << 20).unwrap();
        let a = u.alloc(0x2_0000, 0).unwrap() as usize;
        assert!(u.is_title_address(a) && u.is_title_address(a + 0x1_FFFF));
        assert!(!u.is_title_address(a + 0x2_0000) && !u.is_title_address(u.base() + (8 << 20)));
        assert!(!u.is_title_address(u.base() - 1) && !u.is_title_address(u.base() + (32 << 20)));
        assert!(u.free(a));
        assert!(!u.is_title_address(a));
    }

    /// XDefiant's first reservation is 8 GiB + 4 MiB, more than windows 1..2 hold: it has to go
    /// across window 0 without taking the room its 32-bit descriptor heaps need there.
    #[test]
    fn a_reservation_larger_than_windows_1_and_2_leaves_window_0_room_for_descriptor_heaps() {
        let u = Uma::for_runtime(16 << 20).unwrap();
        let big = 0x2_0041_0000;
        let r = u.sparse_reserve(big).unwrap();
        let heap = 442_368 * 32;
        let (d, ld) = u.sparse_alloc_in(0, heap, 256).expect("a descriptor heap in window 0");
        assert_eq!(u.window_of(d), Some(0));
        assert!(d + ld <= r || d >= r + big, "the heap is not inside the title's reservation");
        assert_ne!(d, u.base(), "no descriptor table at console address 0");
        let (x, _) = u.sparse_alloc(0x2_0000, 0).expect("the title's graphics memory after its reservation");
        assert!(x >= r + big || x + 0x2_0000 <= r);
        assert_eq!(u.sparse_release(x), Some(0x2_0000));
        assert_eq!(u.sparse_release(d), Some(ld));
        assert_eq!(u.sparse_release(r), Some(big));
        assert_eq!(u.sparse_used(), 0);
    }

    #[test]
    fn runtime_layout_places_sparse_in_0_to_2_and_committed_in_3() {
        let u = Uma::for_runtime(16 << 20).unwrap();
        assert_eq!(u.base() % (MAX_WINDOWS * WINDOW), 0);
        assert_eq!(u.shader_windows(), 4);
        assert_eq!(u.windows(), 1);
        assert!(u.is_sparse_window(0) && u.is_sparse_window(1) && u.is_sparse_window(2));
        assert!(u.is_committed_window(3));

        // Committed allocations land in window 3
        let c = u.alloc(0x2_0000, 0).unwrap() as usize;
        assert_eq!(u.window_of(c), Some(3));
        assert!(u.is_committed(c) && !u.is_sparse(c) && u.is_title_address(c));
        assert!(u.free(c));

        // First sparse_alloc lands in window 0, after the descriptor range
        let (a, la) = u.sparse_alloc(0x2_0000, 0).unwrap();
        assert_eq!(a, u.base() + GRAIN + DESC_HEADROOM);
        assert_eq!(la, 0x2_0000);
        assert_eq!(u.window_of(a), Some(0));
        assert!(u.is_sparse(a) && u.is_title_address(a));

        // sparse_alloc_in(0, ...) stays inside window 0
        let (d, ld) = u.sparse_alloc_in(0, 4096, 256).unwrap();
        assert_eq!(u.window_of(d), Some(0));
        assert_eq!(ld, GRAIN);

        // Large sparse_reserve prefers window 1+
        let r = u.sparse_reserve(256 << 20).unwrap();
        assert_eq!(u.window_of(r), Some(1));

        // Release and coalesce
        assert_eq!(u.sparse_release(a), Some(la));
        assert_eq!(u.sparse_release(d), Some(ld));
        assert_eq!(u.sparse_release(r), Some(256 << 20));
        assert_eq!(u.sparse_used(), 0);
        assert!(!u.is_title_address(a));

        // Allocations <= 4 GiB never straddle a window boundary, and DESC_HEADROOM is preserved
        let fill_w0 = WINDOW - DESC_HEADROOM - GRAIN;
        let (x, _) = u.sparse_alloc(fill_w0, GRAIN).unwrap();
        assert_eq!(x, u.base() + GRAIN + DESC_HEADROOM);
        let (y, _) = u.sparse_alloc(GRAIN, GRAIN).unwrap();
        assert_eq!(u.window_of(y), Some(1), "overflows to window 1 before touching window 0 descriptor headroom");
        let (desc, _) = u.sparse_alloc_in(0, GRAIN, GRAIN).unwrap();
        assert_eq!(u.window_of(desc), Some(0), "descriptor headroom in window 0 remains available");
        assert_eq!(u.sparse_release(x), Some(fill_w0));
        assert_eq!(u.sparse_release(y), Some(GRAIN));
        assert_eq!(u.sparse_release(desc), Some(GRAIN));
        assert_eq!(u.sparse_used(), 0);
        // Contiguous sparse windows 0..2 are reported as one OS watch region; window 3 as another.
        assert_eq!(u.watch_regions(), vec![(u.base(), 3 * WINDOW), (u.base() + 3 * WINDOW, 16 << 20)]);
    }
}
