//! `XCL_TRACE_ALLOC=1`: who commits big blocks of memory. `ntdll!NtAllocateVirtualMemory` is
//! detoured; every commit of `XCL_TRACE_ALLOC_MIN` bytes or more (default 1 MiB) is counted under
//! its call stack, and the stacks holding the most are logged every 15 s – the title's, XCL's
//! and the driver's allocations alike (they all end in that call).
//!
//! The detour runs inside the heap's own allocations (with the heap lock held): it takes no
//! lock and allocates nothing – the sites are a fixed table filled with atomics.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static ORIG: AtomicUsize = AtomicUsize::new(0);
static MIN: AtomicUsize = AtomicUsize::new(1 << 20);

const FRAMES: usize = 40;
const SLOTS: usize = 4096;

struct Site {
    /// stack hash (0: free)
    key: AtomicU64,
    /// frames written (by the thread that claimed the slot)
    ready: AtomicUsize,
    frames: [AtomicUsize; FRAMES],
    count: AtomicU64,
    bytes: AtomicU64,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Site = Site {
    key: AtomicU64::new(0),
    ready: AtomicUsize::new(0),
    frames: [const { AtomicUsize::new(0) }; FRAMES],
    count: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
};
static SITES: [Site; SLOTS] = [EMPTY; SLOTS];
static LOST: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn hk_alloc(process: isize, base: *mut usize, zero_bits: usize, size: *mut usize, kind: u32, protect: u32) -> i32 {
    let orig: unsafe extern "system" fn(isize, *mut usize, usize, *mut usize, u32, u32) -> i32 = std::mem::transmute(ORIG.load(Ordering::Acquire));
    let st = orig(process, base, zero_bits, size, kind, protect);
    // the current process (-1), committing, succeeded
    if st >= 0 && process == -1 && kind & 0x1000 != 0 && !size.is_null() && *size >= MIN.load(Ordering::Relaxed) {
        record(*size as u64);
    }
    st
}

unsafe fn record(n: u64) {
    let mut frames = [0usize; FRAMES];
    windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(2, FRAMES as u32, frames.as_mut_ptr() as *mut _, std::ptr::null_mut());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in frames {
        h = (h ^ f as u64).wrapping_mul(0x100_0000_01b3);
    }
    let h = h | 1;
    let mut i = h as usize % SLOTS;
    for _ in 0..SLOTS {
        let s = &SITES[i];
        let k = s.key.load(Ordering::Acquire);
        if k == h {
            s.count.fetch_add(1, Ordering::Relaxed);
            s.bytes.fetch_add(n, Ordering::Relaxed);
            return;
        }
        if k == 0 && s.key.compare_exchange(0, h, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            for (d, f) in s.frames.iter().zip(frames) {
                d.store(f, Ordering::Relaxed);
            }
            s.ready.store(1, Ordering::Release);
            s.count.fetch_add(1, Ordering::Relaxed);
            s.bytes.fetch_add(n, Ordering::Relaxed);
            return;
        }
        if s.key.load(Ordering::Acquire) == h {
            continue; // claimed for this stack just now: counted on the next look
        }
        i = (i + 1) % SLOTS;
    }
    LOST.fetch_add(n, Ordering::Relaxed);
}

fn report() {
    let mut top: Vec<(usize, u64, u64)> = SITES
        .iter()
        .enumerate()
        .filter(|(_, s)| s.ready.load(Ordering::Acquire) != 0)
        .map(|(i, s)| (i, s.count.load(Ordering::Relaxed), s.bytes.load(Ordering::Relaxed)))
        .collect();
    top.sort_by(|a, b| b.2.cmp(&a.2));
    let total: u64 = top.iter().map(|t| t.2).sum();
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    crate::xlog_info!(
        "alloc trace: {:.2} GB committed in blocks of {} KB or more, {} call stacks ({:.2} GB not placed)",
        gb(total),
        MIN.load(Ordering::Relaxed) >> 10,
        top.len(),
        gb(LOST.load(Ordering::Relaxed))
    );
    // by the first module that is not the heap / CRT / Windows' allocator
    let mut by_module: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for &(i, _, bytes) in &top {
        let owner = SITES[i]
            .frames
            .iter()
            .map(|f| f.load(Ordering::Relaxed))
            .take_while(|&f| f != 0)
            .map(crate::modules::describe_address)
            .find(|d| !["ntdll.dll", "ucrtbase.dll", "KERNELBASE.dll", "KERNEL32.DLL", "msvcrt.dll", "d3d12.dll"].iter().any(|m| d.starts_with(m)))
            .map(|d| d.split('+').next().unwrap_or("?").to_string())
            .unwrap_or_else(|| "?".into());
        *by_module.entry(owner).or_default() += bytes;
    }
    let mut mods: Vec<_> = by_module.into_iter().collect();
    mods.sort_by(|a, b| b.1.cmp(&a.1));
    crate::xlog_info!("alloc trace: by module: {}", mods.iter().take(8).map(|(m, b)| format!("{m} {:.2} GB", gb(*b))).collect::<Vec<_>>().join(", "));
    for &(i, count, bytes) in top.iter().take(8) {
        let stack: Vec<String> = SITES[i].frames.iter().map(|f| f.load(Ordering::Relaxed)).take_while(|&f| f != 0).map(crate::modules::describe_address).collect();
        let tail: Vec<&String> = stack.iter().filter(|s| !s.starts_with("amdxc64.dll") && !s.starts_with("ntdll.dll")).collect();
        crate::xlog_info!("alloc trace: {:.2} GB in {count} blocks: {} … callers: {:?}", gb(bytes), stack.iter().take(4).cloned().collect::<Vec<_>>().join(" < "), tail);
    }
    crate::log::flush();
}

pub fn install() {
    if std::env::var_os("XCL_TRACE_ALLOC").is_none() {
        return;
    }
    if let Some(m) = std::env::var("XCL_TRACE_ALLOC_MIN").ok().and_then(|v| v.parse::<usize>().ok()) {
        MIN.store(m, Ordering::Relaxed);
    }
    unsafe {
        if let Err(e) = crate::hooks::hook("ntdll.dll", "NtAllocateVirtualMemory", hk_alloc as usize, &ORIG) {
            crate::xlog_warn!("alloc trace: {e}");
            return;
        }
    }
    crate::xlog_info!("alloc trace: on (commits of {} bytes or more)", MIN.load(Ordering::Relaxed));
    let _ = std::thread::Builder::new().name("xcl-alloc-trace".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(15));
        report();
    });
}
