//! Zero physical pages mapped into CPU-only reservations, committed on first use.
//!
//! Based on upstream XCL 38b05b7's lazy physical pages. Keep PR #187's page numbering
//! and GPU-window mappings: only plain reservations defer backing. Kernel I/O must
//! call `ensure` before using a deferred buffer because it cannot invoke our handler.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{OnceLock, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RESERVE, PAGE_READWRITE};

const PAGE: usize = 0x1_0000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Reserved,
    // Retained until unmapping so a concurrent fault queued before another thread
    // committed the page can retry, rather than being mistaken for an unrelated AV.
    Committed,
}

static PAGES: RwLock<BTreeMap<usize, PageState>> = RwLock::new(BTreeMap::new());

fn install() -> bool {
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    *INSTALLED.get_or_init(|| unsafe { !AddVectoredExceptionHandler(1, Some(handler)).is_null() })
}

/// A single native region must cover the whole lazy page. A partial explicit commit
/// or protection change must not be widened by committing the original 64 KiB again.
unsafe fn query_page(page: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let end = page.checked_add(PAGE)?;
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(page as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0
        || m.BaseAddress as usize > page
        || (m.BaseAddress as usize).checked_add(m.RegionSize)? < end {
        return None;
    }
    Some(m)
}

unsafe fn writable_page(page: usize) -> bool {
    query_page(page).is_some_and(|m| m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE)
}

unsafe fn commit(page: usize) -> bool {
    let mut pages = PAGES.write().unwrap_or_else(|e| e.into_inner());
    match pages.get(&page).copied() {
        Some(PageState::Reserved) => {
            // An explicit commit/protect may have changed native state since mark().
            // Keep the write guard through querying/allocation, and never replace a
            // caller's read-only or guarded protection with read-write protection.
            let Some(m) = query_page(page) else { return false };
            match m.State {
                MEM_COMMIT if m.Protect == PAGE_READWRITE => {}
                MEM_RESERVE => {
                    if VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null() {
                        return false;
                    }
                }
                _ => return false,
            }
            pages.insert(page, PageState::Committed);
            true
        }
        Some(PageState::Committed) => writable_page(page),
        None => false,
    }
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    if info.is_null() || (*info).ExceptionRecord.is_null() { return 0; }
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2
        || r.ExceptionInformation[0] > 1 {
        return 0;
    }
    let page = r.ExceptionInformation[1] & !(PAGE - 1);
    // Committed entries are recognized too: a second thread can already be faulting
    // while the first is making the page writable.
    let tracked = PAGES.read().unwrap_or_else(|e| e.into_inner()).contains_key(&page);
    if tracked && commit(page) { -1 } else { 0 }
}

/// A reserved 64 KiB CPU page, logically containing zeros.
pub(crate) fn mark(page: usize) -> bool {
    if !page.is_multiple_of(PAGE) || !install() { return false; }
    PAGES.write().unwrap_or_else(|e| e.into_inner()).insert(page, PageState::Reserved);
    true
}

/// Remove a mapping; true if it was never touched and has no data to copy home.
pub(super) fn unmark(page: usize) -> bool {
    PAGES.write().unwrap_or_else(|e| e.into_inner()).remove(&page) == Some(PageState::Reserved)
}

/// Commit deferred pages that a kernel call or an XCL write will use.
pub fn ensure(p: usize, len: usize) {
    let Some(end) = p.checked_add(len) else { return };
    if len == 0 { return; }
    let pages: Vec<usize> = PAGES.read().unwrap_or_else(|e| e.into_inner())
        .range((p & !(PAGE - 1))..end)
        .filter_map(|(&a, &s)| (s == PageState::Reserved).then_some(a)).collect();
    for page in pages { unsafe { commit(page); } }
}

/// Commit fully reserved, tracked pages for a native read that cannot invoke the fault handler.
pub fn prepare_read(p: usize, len: usize) -> bool {
    let Some(end) = p.checked_add(len) else { return false };
    if len == 0 { return false; }
    let mut pages = PAGES.write().unwrap_or_else(|e| e.into_inner());
    let mut committed = false;
    for (&page, state) in pages.range_mut((p & !(PAGE - 1))..end) {
        if *state == PageState::Reserved && unsafe { query_page(page).is_some_and(|m| m.State == MEM_RESERVE) }
            && !unsafe { VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
            *state = PageState::Committed;
            committed = true;
        }
    }
    committed
}

/// After a successful release, forget tracked pages in the now-free region. Windows
/// can coalesce adjacent free regions; removing stale metadata there is safe too.
pub(crate) fn forget_released(at: usize) {
    if PAGES.read().unwrap_or_else(|e| e.into_inner()).is_empty() { return; }
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.State != MEM_FREE { return; }
    let start = m.BaseAddress as usize;
    let end = start.saturating_add(m.RegionSize);
    let mut pages = PAGES.write().unwrap_or_else(|e| e.into_inner());
    let gone: Vec<usize> = pages.range(start..end).map(|(&a, _)| a).collect();
    for page in gone { pages.remove(&page); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, VirtualProtect, MEM_RELEASE, PAGE_GUARD, PAGE_READONLY};

    #[test]
    fn ensure_preserves_explicit_readonly_and_guard_protection() {
        unsafe {
            for protect in [PAGE_READONLY, PAGE_READWRITE | PAGE_GUARD] {
                let r = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                assert_ne!(r, 0);
                assert!(mark(r));
                assert!(!VirtualAlloc(r as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                let mut old = 0;
                assert_ne!(VirtualProtect(r as *const c_void, PAGE, protect, &mut old), 0);
                ensure(r + 8, 64);
                let m = query_page(r).expect("the full page remains committed");
                assert_eq!(m.State, MEM_COMMIT);
                assert_eq!(m.Protect, protect, "ensure must preserve explicit protection");
                assert!(!commit(r), "a queued fault must not override explicit protection");
                assert!(unmark(r), "rejected preparation retains deferred metadata");
                assert_ne!(VirtualFree(r as *mut c_void, 0, MEM_RELEASE), 0);
            }
        }
    }

    #[test]
    fn a_lazy_page_commits_on_touch_and_ensure() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            assert!(mark(r)); assert!(mark(r + PAGE));
            assert!(!writable_page(r));
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r));
            ensure(r + PAGE + 4, 4);
            assert!(writable_page(r + PAGE));
            assert!(commit(r + PAGE), "a queued concurrent fault may retry after commit");
            assert_ne!(VirtualFree(r as *mut c_void, 0, MEM_RELEASE), 0);
            forget_released(r);
            assert!(!PAGES.read().unwrap().contains_key(&(r + PAGE)));
        }
    }

    #[test]
    fn failed_commit_keeps_the_page_deferred() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            assert_ne!(VirtualFree(r as *mut c_void, 0, MEM_RELEASE), 0);
            assert!(mark(r));
            assert!(!commit(r), "MEM_COMMIT cannot commit a released allocation");
            assert!(unmark(r), "failed allocation did not lose deferred state");
        }
    }

    #[test]
    fn concurrent_first_touches_keep_every_threads_data() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0); assert!(mark(r));
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
            let threads: Vec<_> = (0..8).map(|i| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let p = (r + i * 8) as *mut u64;
                    assert_eq!(std::ptr::read_volatile(p), 0);
                    std::ptr::write_volatile(p, (i + 1) as u64);
                })
            }).collect();
            for t in threads { t.join().unwrap(); }
            for i in 0..8 { assert_eq!(std::ptr::read_volatile((r + i * 8) as *const u64), (i + 1) as u64); }
            assert!(!unmark(r));
            assert_ne!(VirtualFree(r as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }


    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(PAGES.read().unwrap().get(&(r + PAGE)) == Some(&PageState::Committed));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
            forget_released(r);
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
            forget_released(r);
        }
    }
}
