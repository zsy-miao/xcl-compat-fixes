//! Process-wide Win32 file API hooks feeding the VFS.
//!
//! Hooks sit on KERNELBASE exports, so they also catch calls made by the CRT
//! (`fopen` → `CreateFileW`) and other system DLLs on the game's behalf – not just
//! the game's own imports. A thread-local guard makes the hooks transparent for any
//! file I/O the VFS itself performs.
//!
//! Known gap (documented in the report): direct `Nt*` file syscalls are not hooked.

use std::cell::Cell;
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::Foundation::{GetLastError, SetLastError, ERROR_NOT_SAME_DEVICE, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_DELAY_UNTIL_REBOOT, WIN32_FIND_DATAA, WIN32_FIND_DATAW};

use crate::hooks::hook;
use crate::util::{from_ansi_ptr, from_file_ansi_ptr, from_wide_ptr, wide};
use crate::vfs::{self, Decision, Target};

thread_local! {
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

struct Guard;
impl Guard {
    fn enter() -> Option<Guard> {
        IN_HOOK.with(|c| {
            if c.get() {
                None
            } else {
                c.set(true);
                Some(Guard)
            }
        })
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        let e = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        IN_HOOK.with(|c| c.set(false));
        unsafe { windows_sys::Win32::Foundation::SetLastError(e) };
    }
}

macro_rules! orig {
    ($slot:ident as $ty:ty) => {
        std::mem::transmute::<usize, $ty>($slot.load(Ordering::Acquire))
    };
}

const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;

// ---------------------------------------------------------------------------------------
// CreateFile*
// ---------------------------------------------------------------------------------------

type FnCreateFileW = unsafe extern "system" fn(*const u16, u32, u32, *const c_void, u32, u32, HANDLE) -> HANDLE;
type FnCreateFileA = unsafe extern "system" fn(*const u8, u32, u32, *const c_void, u32, u32, HANDLE) -> HANDLE;
type FnCreateFile2 = unsafe extern "system" fn(*const u16, u32, u32, u32, *const c_void) -> HANDLE;
static O_CREATE_FILE_W: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_FILE_A: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_FILE_2: AtomicUsize = AtomicUsize::new(0);

/// The first opens with their flags (how a title reads: overlapped / unbuffered I/O writes
/// memory without the CPU – write-watch does not see it).
fn log_open(api: &str, path: &str, flags: u32) {
    static N: AtomicUsize = AtomicUsize::new(0);
    // the title's own data, not drivers' caches, system files or DLLs
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".dll") || lower.ends_with(".exe") || lower.contains(r"\appdata\") || lower.contains(r"\windows\") || lower.starts_with(r"\\") {
        return;
    }
    if N.fetch_add(1, Ordering::Relaxed) < 24 {
        crate::xlog_info!(
            "VFS: {api}({path}) flags {flags:#x}{}{}",
            if flags & 0x2000_0000 != 0 { " NO_BUFFERING" } else { "" },
            if flags & 0x4000_0000 != 0 { " OVERLAPPED" } else { "" }
        );
    }
}

fn apply_open_decision(
    decision: Decision,
    disposition: u32,
    passthrough: impl FnOnce() -> HANDLE,
    redirected: impl FnOnce(*const u16, u32) -> HANDLE,
) -> HANDLE {
    match decision {
        Decision::Pass => passthrough(),
        Decision::Use(path) => redirected(wide(&path).as_ptr(), disposition),
        Decision::UseWithDisposition(path, disposition) => redirected(wide(&path).as_ptr(), disposition),
        Decision::CreateOverOriginal(path) => {
            let handle = redirected(wide(&path).as_ptr(), disposition);
            // Report that the virtual file existed before the open.
            if handle != INVALID_HANDLE_VALUE {
                unsafe { SetLastError(vfs::ERROR_ALREADY_EXISTS) };
            }
            handle
        }
        Decision::Fail(error) => {
            unsafe { SetLastError(error) };
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe extern "system" fn hk_create_file_w(n: *const u16, acc: u32, sh: u32, sa: *const c_void, disp: u32, fl: u32, t: HANDLE) -> HANDLE {
    let orig = orig!(O_CREATE_FILE_W as FnCreateFileW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, acc, sh, sa, disp, fl, t) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, acc, sh, sa, disp, fl, t) };
    log_open("CreateFileW", &path, fl);
    apply_open_decision(
        v.open_decision(&path, acc, disp, fl),
        disp,
        || orig(n, acc, sh, sa, disp, fl, t),
        |p, d| orig(p, acc, sh, sa, d, fl, t),
    )
}

unsafe extern "system" fn hk_create_file_a(n: *const u8, acc: u32, sh: u32, sa: *const c_void, disp: u32, fl: u32, t: HANDLE) -> HANDLE {
    let orig = orig!(O_CREATE_FILE_A as FnCreateFileA);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, acc, sh, sa, disp, fl, t) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, acc, sh, sa, disp, fl, t) };
    log_open("CreateFileA", &path, fl);
    apply_open_decision(
        v.open_decision(&path, acc, disp, fl),
        disp,
        || orig(n, acc, sh, sa, disp, fl, t),
        |p, d| orig!(O_CREATE_FILE_W as FnCreateFileW)(p, acc, sh, sa, d, fl, t),
    )
}

unsafe extern "system" fn hk_create_file_2(n: *const u16, acc: u32, sh: u32, disp: u32, params: *const c_void) -> HANDLE {
    let orig = orig!(O_CREATE_FILE_2 as FnCreateFile2);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, acc, sh, disp, params) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, acc, sh, disp, params) };
    // CREATEFILE2_EXTENDED_PARAMETERS { dwSize, dwFileAttributes, dwFileFlags, ... }
    let flags = if params.is_null() { 0 } else { *((params as *const u8).add(8) as *const u32) };
    log_open("CreateFile2", &path, flags);
    apply_open_decision(
        v.open_decision(&path, acc, disp, flags),
        disp,
        || orig(n, acc, sh, disp, params),
        |p, d| orig(p, acc, sh, d, params),
    )
}

// ---------------------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------------------

type FnGetAttrW = unsafe extern "system" fn(*const u16) -> u32;
type FnGetAttrExW = unsafe extern "system" fn(*const u16, i32, *mut c_void) -> i32;
type FnSetAttrW = unsafe extern "system" fn(*const u16, u32) -> i32;
static O_GET_ATTR_W: AtomicUsize = AtomicUsize::new(0);
static O_GET_ATTR_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_GET_ATTR_A: AtomicUsize = AtomicUsize::new(0);
static O_GET_ATTR_EX_A: AtomicUsize = AtomicUsize::new(0);
static O_SET_ATTR_W: AtomicUsize = AtomicUsize::new(0);

/// None = pass through untouched; Some(Ok(path)) = use path; Some(Err(code)) = fail.
fn read_target(path: &str) -> Option<Result<String, u32>> {
    let v = vfs::get()?;
    match v.classify(path) {
        Target::Pass => None,
        Target::Translated(p) => Some(Ok(p)),
        Target::Game { rel } => Some(v.read_path_rel(&rel).ok_or(vfs::ERROR_FILE_NOT_FOUND)),
    }
}

unsafe extern "system" fn hk_get_attr_w(n: *const u16) -> u32 {
    let orig = orig!(O_GET_ATTR_W as FnGetAttrW);
    let Some(_g) = Guard::enter() else { return orig(n) };
    let Some(path) = from_wide_ptr(n) else { return orig(n) };
    match read_target(&path) {
        None => orig(n),
        Some(Ok(p)) => orig(wide(&p).as_ptr()),
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_FILE_ATTRIBUTES
        }
    }
}

unsafe extern "system" fn hk_get_attr_a(n: *const u8) -> u32 {
    type FnA = unsafe extern "system" fn(*const u8) -> u32;
    let orig = orig!(O_GET_ATTR_A as FnA);
    let Some(_g) = Guard::enter() else { return orig(n) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n) };
    match read_target(&path) {
        None => orig(n),
        Some(Ok(p)) => orig!(O_GET_ATTR_W as FnGetAttrW)(wide(&p).as_ptr()),
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_FILE_ATTRIBUTES
        }
    }
}

unsafe extern "system" fn hk_get_attr_ex_w(n: *const u16, lvl: i32, info: *mut c_void) -> i32 {
    let orig = orig!(O_GET_ATTR_EX_W as FnGetAttrExW);
    let Some(_g) = Guard::enter() else { return orig(n, lvl, info) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, lvl, info) };
    match read_target(&path) {
        None => orig(n, lvl, info),
        Some(Ok(p)) => orig(wide(&p).as_ptr(), lvl, info),
        Some(Err(e)) => {
            SetLastError(e);
            0
        }
    }
}

unsafe extern "system" fn hk_get_attr_ex_a(n: *const u8, lvl: i32, info: *mut c_void) -> i32 {
    type FnA = unsafe extern "system" fn(*const u8, i32, *mut c_void) -> i32;
    let orig = orig!(O_GET_ATTR_EX_A as FnA);
    let Some(_g) = Guard::enter() else { return orig(n, lvl, info) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, lvl, info) };
    match read_target(&path) {
        None => orig(n, lvl, info),
        Some(Ok(p)) => orig!(O_GET_ATTR_EX_W as FnGetAttrExW)(wide(&p).as_ptr(), lvl, info),
        Some(Err(e)) => {
            SetLastError(e);
            0
        }
    }
}

unsafe extern "system" fn hk_set_attr_w(n: *const u16, attrs: u32) -> i32 {
    let orig = orig!(O_SET_ATTR_W as FnSetAttrW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, attrs) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, attrs) };
    match v.classify(&path) {
        Target::Pass => orig(n, attrs),
        Target::Translated(p) => orig(wide(&p).as_ptr(), attrs),
        Target::Game { rel } => {
            if !v.visible_in_overlay(&rel) && !v.visible_in_original(&rel) {
                SetLastError(vfs::ERROR_FILE_NOT_FOUND);
                return 0;
            }
            match v.prepare_write(&rel, true) {
                Ok(p) => orig(wide(&p).as_ptr(), attrs),
                Err(e) => {
                    SetLastError(e);
                    0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Free space: a console drive (T:\, G:\, …) reports the host volume its mount lives on.
// The Ex APIs additionally cap the persistent local storage mount to the title's quota.
// Untranslated, a console drive letter does not exist on the PC, the query fails and
// titles read it as a full disk (Minecraft then refuses to create a world).
// ---------------------------------------------------------------------------------------

type FnDiskFreeExW = unsafe extern "system" fn(*const u16, *mut u64, *mut u64, *mut u64) -> i32;
type FnDiskFreeExA = unsafe extern "system" fn(*const u8, *mut u64, *mut u64, *mut u64) -> i32;
type FnDiskFreeW = unsafe extern "system" fn(*const u16, *mut u32, *mut u32, *mut u32, *mut u32) -> i32;
type FnDiskFreeA = unsafe extern "system" fn(*const u8, *mut u32, *mut u32, *mut u32, *mut u32) -> i32;
static O_DISK_FREE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_DISK_FREE_EX_A: AtomicUsize = AtomicUsize::new(0);
static O_DISK_FREE_W: AtomicUsize = AtomicUsize::new(0);
static O_DISK_FREE_A: AtomicUsize = AtomicUsize::new(0);

// A NULL directory (the current directory's volume) has no path to translate and passes through.

static SPACE_LOGGED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// What a title learned about free space, once per queried directory: titles act on these
/// figures (a failed or tiny answer reads as a full disk), so the log shows what they saw.
/// Call right after the API: the title's last-error value survives the logging.
fn log_space(api: &str, path: &str, real: Option<&str>, answer: Option<(u64, u64)>) {
    let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    if let Ok(mut g) = SPACE_LOGGED.lock() {
        let seen = g.get_or_insert_with(HashSet::new);
        if seen.len() < 64 && seen.insert(format!("{api}|{}", path.to_lowercase())) {
            let host = match real {
                Some(r) if r != path => format!(" (host {r})"),
                _ => String::new(),
            };
            match answer {
                Some((avail, total)) => crate::xlog_info!("{api}({path}){host}: {} MiB available of {} MiB", avail >> 20, total >> 20),
                None => crate::xlog_warn!("{api}({path}){host} failed (error {err})"),
            }
        }
    }
    unsafe { SetLastError(err) };
}

/// `GetDiskFreeSpaceW` figures in bytes: (free clusters, total clusters) × bytes per cluster.
unsafe fn clusters_to_bytes(spc: *const u32, bps: *const u32, nfree: *const u32, ntotal: *const u32) -> Option<(u64, u64)> {
    if spc.is_null() || bps.is_null() || nfree.is_null() || ntotal.is_null() {
        return None;
    }
    let cluster = *spc as u64 * *bps as u64;
    Some((*nfree as u64 * cluster, *ntotal as u64 * cluster))
}

unsafe fn ex_answer(ok: i32, avail: *const u64, total: *const u64) -> Option<(u64, u64)> {
    (ok != 0).then(|| (avail.as_ref().copied().unwrap_or(0), total.as_ref().copied().unwrap_or(0)))
}

unsafe fn pls_disk_free_ex(
    orig: FnDiskFreeExW,
    path: *const u16,
    root: &std::path::Path,
    quota: u64,
    [avail, total, free]: [*mut u64; 3],
) -> i32 {
    if avail.is_null() && total.is_null() && free.is_null() {
        return orig(path, avail, total, free);
    }
    let (mut host_avail, mut host_total, mut host_free) = (0, 0, 0);
    let ok = orig(path, &mut host_avail, &mut host_total, &mut host_free);
    if ok == 0 {
        return ok;
    }
    let err = windows_sys::Win32::Foundation::GetLastError();
    let capacity = quota.min(host_total);
    // Host free space already reflects host usage. Subtract PLS usage from its quota.
    let remaining = if avail.is_null() && free.is_null() {
        quota
    } else {
        quota.saturating_sub(crate::hle::xgameruntime::pls_used_bytes(root))
    };
    // These pointers are optional and may alias. Write in the native API's order.
    if !avail.is_null() {
        avail.write(host_avail.min(remaining));
    }
    if !total.is_null() {
        total.write(capacity);
    }
    if !free.is_null() {
        free.write(host_free.min(remaining));
    }
    SetLastError(err);
    ok
}

unsafe fn disk_free_ex(
    api: &str,
    path: &str,
    err: u32,
    outputs: [*mut u64; 3],
    pass: impl FnOnce() -> i32,
) -> i32 {
    let [avail, total, free] = outputs;
    let (target, pls) = match vfs::get().map(|v| (v, v.classify(path))) {
        Some((v, Target::Translated(p))) => {
            let pls = v.pls_root_for(&p);
            (Some(Ok(p)), pls)
        }
        Some((v, Target::Game { rel })) => (
            Some(v.read_path_rel(&rel).ok_or(vfs::ERROR_FILE_NOT_FOUND)),
            None,
        ),
        _ => (None, None),
    };
    let (r, real) = match target {
        None => {
            SetLastError(err);
            (pass(), None)
        }
        Some(Ok(p)) => {
            let orig = orig!(O_DISK_FREE_EX_W as FnDiskFreeExW);
            let w = wide(&p);
            let r = if let Some(root) = pls {
                let quota = crate::state::config().pls_quota_bytes();
                SetLastError(err);
                pls_disk_free_ex(orig, w.as_ptr(), root, quota, outputs)
            } else {
                SetLastError(err);
                orig(w.as_ptr(), avail, total, free)
            };
            (r, Some(p))
        }
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    log_space(api, path, real.as_deref(), ex_answer(r, avail, total));
    r
}

unsafe extern "system" fn hk_disk_free_ex_w(
    n: *const u16,
    avail: *mut u64,
    total: *mut u64,
    free: *mut u64,
) -> i32 {
    let orig = orig!(O_DISK_FREE_EX_W as FnDiskFreeExW);
    let err = windows_sys::Win32::Foundation::GetLastError();
    let Some(_g) = Guard::enter() else {
        return orig(n, avail, total, free);
    };
    let Some(path) = from_wide_ptr(n) else {
        return orig(n, avail, total, free);
    };
    disk_free_ex(
        "GetDiskFreeSpaceExW",
        &path,
        err,
        [avail, total, free],
        || orig(n, avail, total, free),
    )
}

unsafe extern "system" fn hk_disk_free_ex_a(
    n: *const u8,
    avail: *mut u64,
    total: *mut u64,
    free: *mut u64,
) -> i32 {
    let orig = orig!(O_DISK_FREE_EX_A as FnDiskFreeExA);
    let err = windows_sys::Win32::Foundation::GetLastError();
    let Some(_g) = Guard::enter() else { return orig(n, avail, total, free) };
    let Some(path) = from_file_ansi_ptr(n) else {
        return orig(n, avail, total, free);
    };
    disk_free_ex(
        "GetDiskFreeSpaceExA",
        &path,
        err,
        [avail, total, free],
        || orig(n, avail, total, free),
    )
}

unsafe extern "system" fn hk_disk_free_w(n: *const u16, spc: *mut u32, bps: *mut u32, nfree: *mut u32, ntotal: *mut u32) -> i32 {
    let orig = orig!(O_DISK_FREE_W as FnDiskFreeW);
    let Some(_g) = Guard::enter() else { return orig(n, spc, bps, nfree, ntotal) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, spc, bps, nfree, ntotal) };
    let (r, real) = match read_target(&path) {
        None => (orig(n, spc, bps, nfree, ntotal), None),
        Some(Ok(p)) => (orig(wide(&p).as_ptr(), spc, bps, nfree, ntotal), Some(p)),
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    let answer = if r != 0 { clusters_to_bytes(spc, bps, nfree, ntotal) } else { None };
    log_space("GetDiskFreeSpaceW", &path, real.as_deref(), answer);
    r
}

unsafe extern "system" fn hk_disk_free_a(n: *const u8, spc: *mut u32, bps: *mut u32, nfree: *mut u32, ntotal: *mut u32) -> i32 {
    let orig = orig!(O_DISK_FREE_A as FnDiskFreeA);
    let Some(_g) = Guard::enter() else { return orig(n, spc, bps, nfree, ntotal) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, spc, bps, nfree, ntotal) };
    let (r, real) = match read_target(&path) {
        None => (orig(n, spc, bps, nfree, ntotal), None),
        Some(Ok(p)) => (orig!(O_DISK_FREE_W as FnDiskFreeW)(wide(&p).as_ptr(), spc, bps, nfree, ntotal), Some(p)),
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    let answer = if r != 0 { clusters_to_bytes(spc, bps, nfree, ntotal) } else { None };
    log_space("GetDiskFreeSpaceA", &path, real.as_deref(), answer);
    r
}

// ---------------------------------------------------------------------------------------
// Directory / delete / move / copy
// ---------------------------------------------------------------------------------------

type FnCreateDirW = unsafe extern "system" fn(*const u16, *const c_void) -> i32;
type FnPathW = unsafe extern "system" fn(*const u16) -> i32;
type FnMoveFileExW = unsafe extern "system" fn(*const u16, *const u16, u32) -> i32;
type FnMoveFileWithProgressW = unsafe extern "system" fn(*const u16, *const u16, *const c_void, *const c_void, u32) -> i32;
type FnMoveFileWithProgressA = unsafe extern "system" fn(*const u8, *const u8, *const c_void, *const c_void, u32) -> i32;
type FnCopyFileExW = unsafe extern "system" fn(*const u16, *const u16, *const c_void, *const c_void, *mut i32, u32) -> i32;
type FnCopyFileW = unsafe extern "system" fn(*const u16, *const u16, i32) -> i32;
type FnCopyFile2 = unsafe extern "system" fn(*const u16, *const u16, *const c_void) -> i32;
type FnReplaceFileW = unsafe extern "system" fn(*const u16, *const u16, *const u16, u32, *const c_void, *const c_void) -> i32;
static O_CREATE_DIR_W: AtomicUsize = AtomicUsize::new(0);
static O_REMOVE_DIR_W: AtomicUsize = AtomicUsize::new(0);
static O_DELETE_FILE_W: AtomicUsize = AtomicUsize::new(0);
static O_READ_FILE: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_PROGRESS_W: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_A: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_EX_A: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_PROGRESS_A: AtomicUsize = AtomicUsize::new(0);
static O_COPY_FILE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_COPY_FILE_W: AtomicUsize = AtomicUsize::new(0);
static O_COPY_FILE_2: AtomicUsize = AtomicUsize::new(0);
static O_REPLACE_FILE_W: AtomicUsize = AtomicUsize::new(0);

unsafe fn bool_result(r: Result<(), u32>) -> i32 {
    match r {
        Ok(()) => 1,
        Err(e) => {
            SetLastError(e);
            0
        }
    }
}

unsafe extern "system" fn hk_create_dir_w(n: *const u16, sa: *const c_void) -> i32 {
    let orig = orig!(O_CREATE_DIR_W as FnCreateDirW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, sa) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, sa) };
    match v.classify(&path) {
        Target::Pass => orig(n, sa),
        Target::Translated(p) => orig(wide(&p).as_ptr(), sa),
        Target::Game { rel } => bool_result(v.create_dir(&rel)),
    }
}

unsafe extern "system" fn hk_remove_dir_w(n: *const u16) -> i32 {
    let orig = orig!(O_REMOVE_DIR_W as FnPathW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n) };
    let Some(path) = from_wide_ptr(n) else { return orig(n) };
    match v.classify(&path) {
        Target::Pass => orig(n),
        Target::Translated(p) => orig(wide(&p).as_ptr()),
        Target::Game { rel } => bool_result(v.remove_dir(&rel)),
    }
}

/// ReadFile: a buffer in pages the title mapped but has not used yet (`kernelx::lazy`) is
/// committed first – the kernel's write there would fail rather than fault in the title.
unsafe extern "system" fn hk_read_file(h: *mut c_void, buf: *mut c_void, n: u32, read: *mut u32, ov: *mut c_void) -> i32 {
    type F = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut u32, *mut c_void) -> i32;
    let orig = orig!(O_READ_FILE as F);
    if !buf.is_null() {
        crate::hle::kernelx::lazy::ensure(buf as usize, n as usize);
    }
    orig(h, buf, n, read, ov)
}

unsafe extern "system" fn hk_delete_file_w(n: *const u16) -> i32 {
    let orig = orig!(O_DELETE_FILE_W as FnPathW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n) };
    let Some(path) = from_wide_ptr(n) else { return orig(n) };
    match v.classify(&path) {
        Target::Pass => orig(n),
        Target::Translated(p) => orig(wide(&p).as_ptr()),
        Target::Game { rel } => bool_result(v.delete_file(&rel)),
    }
}

/// Source for a copy/move: (real path to read, game rel if it lives in the game folder).
fn source_of(v: &vfs::Vfs, path: &str) -> Result<(String, Option<String>), u32> {
    match v.classify(path) {
        Target::Pass => Ok((path.to_string(), None)),
        Target::Translated(p) => Ok((p, None)),
        Target::Game { rel } => {
            let p = v.read_path_rel(&rel).ok_or(vfs::ERROR_FILE_NOT_FOUND)?;
            Ok((p, Some(rel)))
        }
    }
}

/// Destination for a copy/move: (real path to write, game rel if it lives in the game folder).
/// For game paths, whiteouts are NOT cleared here — call `v.commit_write(&rel)` only after
/// the underlying copy/move succeeds so a failed operation cannot resurrect a deleted file.
fn dest_of(v: &vfs::Vfs, path: &str) -> Result<(String, Option<String>), u32> {
    match v.classify(path) {
        Target::Pass => Ok((path.to_string(), None)),
        Target::Translated(p) => {
            if let Some(parent) = std::path::Path::new(&p).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            Ok((p, None))
        }
        Target::Game { rel } => Ok((v.stage_write(&rel)?, Some(rel))),
    }
}

const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
const MOVEFILE_COPY_ALLOWED: u32 = 0x2;
const COPY_FILE_FAIL_IF_EXISTS: u32 = 0x1;

fn io_err_code(e: &std::io::Error, fallback: u32) -> u32 {
    e.raw_os_error().map(|c| c as u32).unwrap_or(fallback)
}

fn vfs_path_exists(v: &vfs::Vfs, path: &str) -> Result<bool, u32> {
    match source_of(v, path) {
        Ok((real, _)) => Ok(std::path::Path::new(&real).exists()),
        Err(vfs::ERROR_FILE_NOT_FOUND) => Ok(false),
        Err(e) => Err(e),
    }
}

fn validate_replace_targets(v: &vfs::Vfs, target: &str, source: &str) -> Result<(), u32> {
    if !vfs_path_exists(v, target)? || !vfs_path_exists(v, source)? {
        return Err(vfs::ERROR_FILE_NOT_FOUND);
    }
    Ok(())
}

unsafe extern "system" fn hk_move_file_ex_w(src: *const u16, dst: *const u16, flags: u32) -> i32 {
    let orig = orig!(O_MOVE_FILE_EX_W as FnMoveFileExW);
    move_file(src, dst, flags, |s, d, f| orig(s, d, f))
}

/// MoveFileW reaches MoveFileWithProgressW without going through MoveFileExW (Windows 11; under
/// Wine the ANSI moves do too). Where MoveFileExW calls it, the guard lets the inner call through.
unsafe extern "system" fn hk_move_file_with_progress_w(src: *const u16, dst: *const u16, prog: *const c_void, data: *const c_void, flags: u32) -> i32 {
    let orig = orig!(O_MOVE_FILE_PROGRESS_W as FnMoveFileWithProgressW);
    move_file(src, dst, flags, |s, d, f| orig(s, d, prog, data, f))
}

/// kernel32's MoveFileA / MoveFileExA / MoveFileWithProgressA reach none of kernelbase's W move
/// entries on Windows 11 (an internal helper does the work), so they are hooked themselves: the
/// paths become wide and the move goes the MoveFileWithProgressW way.
unsafe fn move_file_a(src: *const u8, dst: *const u8, prog: *const c_void, data: *const c_void, flags: u32, orig: impl Fn() -> i32) -> i32 {
    let Some(s) = from_ansi_ptr(src) else { return orig() };
    let d = from_ansi_ptr(dst).map(|d| wide(&d));
    hk_move_file_with_progress_w(wide(&s).as_ptr(), d.as_ref().map_or(std::ptr::null(), |d| d.as_ptr()), prog, data, flags)
}

unsafe extern "system" fn hk_move_file_a(src: *const u8, dst: *const u8) -> i32 {
    let orig = orig!(O_MOVE_FILE_A as unsafe extern "system" fn(*const u8, *const u8) -> i32);
    move_file_a(src, dst, std::ptr::null(), std::ptr::null(), MOVEFILE_COPY_ALLOWED /* as MoveFileW */, || orig(src, dst))
}

unsafe extern "system" fn hk_move_file_ex_a(src: *const u8, dst: *const u8, flags: u32) -> i32 {
    let orig = orig!(O_MOVE_FILE_EX_A as unsafe extern "system" fn(*const u8, *const u8, u32) -> i32);
    move_file_a(src, dst, std::ptr::null(), std::ptr::null(), flags, || orig(src, dst, flags))
}

unsafe extern "system" fn hk_move_file_with_progress_a(src: *const u8, dst: *const u8, prog: *const c_void, data: *const c_void, flags: u32) -> i32 {
    let orig = orig!(O_MOVE_FILE_PROGRESS_A as FnMoveFileWithProgressA);
    move_file_a(src, dst, prog, data, flags, || orig(src, dst, prog, data, flags))
}

unsafe fn move_file_vfs(v: &vfs::Vfs, s: &str, d: &str, flags: u32, orig: impl Fn(*const u16, *const u16, u32) -> i32) -> i32 {
    let s_target = v.classify(s);
    let d_target = v.classify(d);
    let s_game = matches!(s_target, Target::Game { .. });
    let d_game = matches!(d_target, Target::Game { .. });
    if !s_game && !d_game {
        let (rs, _) = match source_of(v, s) {
            Ok(x) => x,
            Err(e) => return bool_result(Err(e)),
        };
        let (rd, _) = match dest_of(v, d) {
            Ok(x) => x,
            Err(e) => return bool_result(Err(e)),
        };
        return orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags);
    }
    let replace_existing = flags & MOVEFILE_REPLACE_EXISTING != 0;
    let drel_opt = match d_target {
        Target::Game { rel } => Some(rel),
        _ => None,
    };
    let r = (|| -> Result<(), u32> {
        let (rs, srel) = source_of(v, s)?;
        match &srel {
            // source in the game folder: copy, then hide the source
            Some(sr) => {
                let src_meta = std::fs::metadata(&rs).map_err(|e| io_err_code(&e, vfs::ERROR_FILE_NOT_FOUND))?;
                if src_meta.is_dir() {
                    return Err(vfs::ERROR_ACCESS_DENIED); // moving whole game dirs is not supported
                }
                let source_in_overlay = rs == v.overlay_path(sr);
                // Case-only or exact self-rename inside the game folder: never delete/whiteout!
                if let Some(dr) = &drel_opt {
                    if sr.to_lowercase() == dr.to_lowercase() {
                        if sr == dr {
                            return Ok(());
                        }
                        if source_in_overlay {
                            let rd = v.overlay_path(dr);
                            if orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags) == 0 {
                                // SAFETY: GetLastError reads thread-local Win32 error state.
                                return Err(windows_sys::Win32::Foundation::GetLastError());
                            }
                            v.commit_write(dr);
                            return Ok(());
                        }
                        let rd = v.stage_write(dr)?;
                        std::fs::copy(&rs, &rd).map_err(|e| io_err_code(&e, vfs::ERROR_ACCESS_DENIED))?;
                        v.commit_write(dr);
                        return Ok(());
                    }
                }
                if let Some(dr) = &drel_opt {
                    if v.is_dir(dr) {
                        return Err(vfs::ERROR_ACCESS_DENIED);
                    }
                    if !replace_existing && v.exists(dr) {
                        return Err(vfs::ERROR_ALREADY_EXISTS);
                    }
                }
                let (rd, drel) = dest_of(v, d)?;
                if let Some(dr) = &drel {
                    if source_in_overlay && flags & MOVEFILE_DELAY_UNTIL_REBOOT == 0 {
                        // An in-place rename checks delete sharing before replacing the destination.
                        let moved = orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags & !MOVEFILE_COPY_ALLOWED);
                        if moved != 0 {
                            v.commit_write(dr);
                            v.commit_overlay_removal(sr);
                            return Ok(());
                        }
                        let error = GetLastError();
                        if error != ERROR_NOT_SAME_DEVICE {
                            return Err(error);
                        }
                    }
                } else {
                    if let Ok(m) = std::fs::metadata(&rd) {
                        if m.is_dir() {
                            return Err(vfs::ERROR_ACCESS_DENIED);
                        }
                        if !replace_existing {
                            return Err(vfs::ERROR_ALREADY_EXISTS);
                        }
                    }
                    // Move an overlay file directly when both paths are on one volume.
                    // Windows checks delete sharing before replacing the destination.
                    if source_in_overlay && flags & MOVEFILE_DELAY_UNTIL_REBOOT == 0 {
                        if orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags & !MOVEFILE_COPY_ALLOWED) != 0 {
                            v.commit_overlay_removal(sr);
                            return Ok(());
                        }
                        let error = GetLastError();
                        if error != ERROR_NOT_SAME_DEVICE {
                            return Err(error);
                        }
                    }
                }
                std::fs::copy(&rs, &rd).map_err(|e| io_err_code(&e, vfs::ERROR_ACCESS_DENIED))?;
                if let Some(dr) = &drel {
                    v.commit_write(dr);
                }
                v.delete_file(sr).or_else(|e| if e == vfs::ERROR_FILE_NOT_FOUND { Ok(()) } else { Err(e) })?;
                Ok(())
            }
            None => {
                if let Some(dr) = &drel_opt {
                    if std::path::Path::new(&rs).exists() {
                        if v.is_dir(dr) {
                            return Err(vfs::ERROR_ACCESS_DENIED);
                        }
                        if !replace_existing && v.exists(dr) {
                            return Err(vfs::ERROR_ALREADY_EXISTS);
                        }
                    }
                }
                let (rd, drel) = dest_of(v, d)?;
                if orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags | MOVEFILE_COPY_ALLOWED) == 0 {
                    // SAFETY: GetLastError reads thread-local Win32 error state.
                    return Err(windows_sys::Win32::Foundation::GetLastError());
                }
                if let Some(dr) = &drel {
                    v.commit_write(dr);
                }
                Ok(())
            }
        }
    })();
    bool_result(r)
}

unsafe fn move_file(src: *const u16, dst: *const u16, flags: u32, orig: impl Fn(*const u16, *const u16, u32) -> i32) -> i32 {
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, flags) };
    let (Some(s), d) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, flags) };
    let Some(d) = d else { return orig(src, dst, flags) }; // delete-on-reboot form
    move_file_vfs(v, &s, &d, flags, orig)
}

fn prepare_copy(v: &vfs::Vfs, s: &str, d: &str, fail_if_exists: bool) -> Result<(String, String, Option<String>), u32> {
    let (rs, srel) = source_of(v, s)?;
    if srel.is_some() && !std::path::Path::new(&rs).exists() {
        return Err(vfs::ERROR_FILE_NOT_FOUND);
    }
    if let Target::Game { rel: drel } = v.classify(d) {
        if std::path::Path::new(&rs).exists() {
            if v.is_dir(&drel) {
                return Err(vfs::ERROR_ACCESS_DENIED);
            }
            if fail_if_exists && v.exists(&drel) {
                return Err(vfs::ERROR_FILE_EXISTS);
            }
        }
    }
    let (rd, drel) = dest_of(v, d)?;
    Ok((rs, rd, drel))
}

unsafe fn copy_file_ex_w_vfs(
    v: &vfs::Vfs,
    s: &str,
    d: &str,
    prog: *const c_void,
    data: *const c_void,
    cancel: *mut i32,
    flags: u32,
    orig: impl Fn(*const u16, *const u16, *const c_void, *const c_void, *mut i32, u32) -> i32,
) -> i32 {
    let (rs, rd, drel) = match prepare_copy(v, s, d, flags & COPY_FILE_FAIL_IF_EXISTS != 0) {
        Ok(x) => x,
        Err(e) => return bool_result(Err(e)),
    };
    let ok = orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), prog, data, cancel, flags);
    if ok != 0 {
        if let Some(dr) = drel {
            v.commit_write(&dr);
        }
    }
    ok
}

unsafe extern "system" fn hk_copy_file_ex_w(src: *const u16, dst: *const u16, prog: *const c_void, data: *const c_void, cancel: *mut i32, flags: u32) -> i32 {
    let orig = orig!(O_COPY_FILE_EX_W as FnCopyFileExW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, prog, data, cancel, flags) };
    let (Some(s), Some(d)) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, prog, data, cancel, flags) };
    copy_file_ex_w_vfs(v, &s, &d, prog, data, cancel, flags, |s, d, p, dt, c, f| orig(s, d, p, dt, c, f))
}

unsafe fn copy_file_w_vfs(
    v: &vfs::Vfs,
    s: &str,
    d: &str,
    fail_if_exists: i32,
    orig: impl Fn(*const u16, *const u16, i32) -> i32,
) -> i32 {
    let (rs, rd, drel) = match prepare_copy(v, s, d, fail_if_exists != 0) {
        Ok(x) => x,
        Err(e) => return bool_result(Err(e)),
    };
    let ok = orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), fail_if_exists);
    if ok != 0 {
        if let Some(dr) = drel {
            v.commit_write(&dr);
        }
    }
    ok
}

/// CopyFileW (and CopyFileA, which calls it) need not go through CopyFileExW. When it does,
/// the guard lets the inner call through.
unsafe extern "system" fn hk_copy_file_w(src: *const u16, dst: *const u16, fail_if_exists: i32) -> i32 {
    let orig = orig!(O_COPY_FILE_W as FnCopyFileW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, fail_if_exists) };
    let (Some(s), Some(d)) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, fail_if_exists) };
    copy_file_w_vfs(v, &s, &d, fail_if_exists, |s, d, f| orig(s, d, f))
}

unsafe fn copy_file_2_fail_if_exists(params: *const c_void) -> bool {
    if params.is_null() {
        return false;
    }
    // COPYFILE2_EXTENDED_PARAMETERS { dwSize: u32, dwCopyFlags: u32, ... }.
    // SAFETY: Caller passed a non-null COPYFILE2_EXTENDED_PARAMETERS pointer; check dwSize
    // covers dwCopyFlags before reading offset 4.
    let size = *(params as *const u32);
    if (size as usize) < 8 {
        return false;
    }
    let flags = *((params as *const u32).add(1));
    flags & COPY_FILE_FAIL_IF_EXISTS != 0
}

unsafe fn copy_file_2_vfs(
    v: &vfs::Vfs,
    s: &str,
    d: &str,
    params: *const c_void,
    orig: impl Fn(*const u16, *const u16, *const c_void) -> i32,
) -> i32 {
    let fail_if_exists = copy_file_2_fail_if_exists(params);
    let (rs, rd, drel) = match prepare_copy(v, s, d, fail_if_exists) {
        Ok(x) => x,
        Err(e) => return crate::util::hresult_from_win32(e),
    };
    let hr = orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), params);
    if hr >= 0 {
        if let Some(dr) = drel {
            v.commit_write(&dr);
        }
    }
    hr
}

unsafe extern "system" fn hk_copy_file_2(src: *const u16, dst: *const u16, params: *const c_void) -> i32 {
    let orig = orig!(O_COPY_FILE_2 as FnCopyFile2);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, params) };
    let (Some(s), Some(d)) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, params) };
    copy_file_2_vfs(v, &s, &d, params, |s, d, p| orig(s, d, p))
}

/// ReplaceFileW – the usual "write a temporary file, then replace" save. Console paths outside
/// the game folder are translated; when the game folder is involved, the backup and the
/// replacement go through the copy and move hooks, so the game's own files are not touched.
unsafe extern "system" fn hk_replace_file_w(replaced: *const u16, replacement: *const u16, backup: *const u16, flags: u32, excl: *const c_void, res: *const c_void) -> i32 {
    let orig = orig!(O_REPLACE_FILE_W as FnReplaceFileW);
    let (Some(g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(replaced, replacement, backup, flags, excl, res) };
    let (Some(target), Some(source)) = (from_wide_ptr(replaced), from_wide_ptr(replacement)) else {
        return orig(replaced, replacement, backup, flags, excl, res);
    };
    let backup_path = from_wide_ptr(backup);
    let is_game = |p: &str| matches!(v.classify(p), Target::Game { .. });
    if is_game(&target) || is_game(&source) || backup_path.as_deref().is_some_and(is_game) {
        if let Err(e) = validate_replace_targets(v, &target, &source) {
            return bool_result(Err(e));
        }
        drop(g); // the calls below go through their own hooks
        use windows_sys::Win32::Storage::FileSystem::{CopyFileW, MoveFileExW};
        if let Some(b) = &backup_path {
            if CopyFileW(replaced, wide(b).as_ptr(), 0) == 0 {
                return 0;
            }
        }
        return MoveFileExW(replacement, replaced, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED);
    }
    let r = (|| -> Result<(String, String, Option<String>), u32> {
        let (rt, _) = source_of(v, &target)?;
        let (rs, _) = source_of(v, &source)?;
        let rb = backup_path.as_deref().map(|b| dest_of(v, b).map(|(p, _)| p)).transpose()?;
        Ok((rt, rs, rb))
    })();
    match r {
        Ok((rt, rs, rb)) => {
            let rb = rb.map(|b| wide(&b));
            orig(wide(&rt).as_ptr(), wide(&rs).as_ptr(), rb.as_ref().map_or(std::ptr::null(), |b| b.as_ptr()), flags, excl, res)
        }
        Err(e) => bool_result(Err(e)),
    }
}

// ---------------------------------------------------------------------------------------
// Directory enumeration (merges overlay + original, hides whiteouts)
// ---------------------------------------------------------------------------------------

type FnFindFirstExW = unsafe extern "system" fn(*const u16, i32, *mut c_void, i32, *const c_void, u32) -> HANDLE;
type FnFindFirstW = unsafe extern "system" fn(*const u16, *mut WIN32_FIND_DATAW) -> HANDLE;
type FnFindFirstA = unsafe extern "system" fn(*const u8, *mut WIN32_FIND_DATAA) -> HANDLE;
type FnFindNextW = unsafe extern "system" fn(HANDLE, *mut WIN32_FIND_DATAW) -> i32;
type FnFindNextA = unsafe extern "system" fn(HANDLE, *mut WIN32_FIND_DATAA) -> i32;
type FnFindClose = unsafe extern "system" fn(HANDLE) -> i32;
static O_FIND_FIRST_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_FIND_FIRST_W: AtomicUsize = AtomicUsize::new(0);
static O_FIND_FIRST_A: AtomicUsize = AtomicUsize::new(0);
static O_FIND_NEXT_W: AtomicUsize = AtomicUsize::new(0);
static O_FIND_NEXT_A: AtomicUsize = AtomicUsize::new(0);
static O_FIND_CLOSE: AtomicUsize = AtomicUsize::new(0);

struct MergedFind {
    items: Vec<WIN32_FIND_DATAW>,
    pos: usize,
}

static MERGED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn is_merged(h: HANDLE) -> bool {
    MERGED.lock().ok().and_then(|g| g.as_ref().map(|s| s.contains(&(h as usize)))).unwrap_or(false)
}

fn name_of(d: &WIN32_FIND_DATAW) -> String {
    let n = d.cFileName.iter().position(|&c| c == 0).unwrap_or(d.cFileName.len());
    String::from_utf16_lossy(&d.cFileName[..n])
}

unsafe fn collect(pattern: &str) -> Vec<WIN32_FIND_DATAW> {
    let first = orig!(O_FIND_FIRST_EX_W as FnFindFirstExW);
    let next = orig!(O_FIND_NEXT_W as FnFindNextW);
    let close = orig!(O_FIND_CLOSE as FnFindClose);
    let mut out = Vec::new();
    let mut d: WIN32_FIND_DATAW = std::mem::zeroed();
    let h = first(wide(pattern).as_ptr(), 0, &mut d as *mut _ as *mut c_void, 0, std::ptr::null(), 0);
    if h == INVALID_HANDLE_VALUE {
        return out;
    }
    loop {
        out.push(d);
        if next(h, &mut d) == 0 {
            break;
        }
    }
    close(h);
    out
}

fn collect_game_entries(
    v: &vfs::Vfs,
    rel: &str,
    collect_fn: impl Fn(&str) -> Vec<WIN32_FIND_DATAW>,
) -> Result<Vec<WIN32_FIND_DATAW>, u32> {
    let (dir_rel, pattern) = match rel.rsplit_once('\\') {
        Some((d, p)) => (d.to_string(), p.to_string()),
        None => (String::new(), rel.to_string()),
    };
    if !dir_rel.is_empty() && !v.is_dir(&dir_rel) {
        return Err(vfs::ERROR_PATH_NOT_FOUND);
    }
    let wildcard = pattern.contains('*') || pattern.contains('?');
    if !wildcard && !v.visible_in_overlay(rel) && v.is_whiteout(rel) {
        return Err(vfs::ERROR_FILE_NOT_FOUND);
    }
    let join = |base: String| if pattern.is_empty() { base } else { format!("{base}\\{pattern}") };
    let mut items: Vec<WIN32_FIND_DATAW> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    if v.needs_merge(&dir_rel) || v.visible_in_overlay(rel) {
        for d in collect_fn(&join(v.overlay_path(&dir_rel))) {
            let n = name_of(&d);
            if n.eq_ignore_ascii_case(".xcl_whiteouts") {
                continue;
            }
            if seen.insert(n.to_lowercase()) {
                items.push(d);
            }
        }
    }
    if dir_rel.is_empty() || !v.is_whiteout(&dir_rel) {
        for d in collect_fn(&join(v.original_path(&dir_rel))) {
            let n = name_of(&d);
            let child = if dir_rel.is_empty() { n.clone() } else { format!("{dir_rel}\\{n}") };
            if n != "." && n != ".." && v.is_whiteout(&child) {
                continue;
            }
            if seen.insert(n.to_lowercase()) {
                items.push(d);
            }
        }
    }
    if items.is_empty() {
        return Err(vfs::ERROR_FILE_NOT_FOUND);
    }
    Ok(items)
}

/// Some(handle) when the VFS handled the call; None to pass through.
unsafe fn find_first_vfs(path: &str, out: *mut WIN32_FIND_DATAW) -> Option<Result<HANDLE, u32>> {
    let v = vfs::get()?;
    match v.classify(path) {
        Target::Pass => None,
        Target::Translated(p) => {
            let first = orig!(O_FIND_FIRST_EX_W as FnFindFirstExW);
            let h = first(wide(&p).as_ptr(), 0, out as *mut c_void, 0, std::ptr::null(), 0);
            Some(if h == INVALID_HANDLE_VALUE { Err(windows_sys::Win32::Foundation::GetLastError()) } else { Ok(h) })
        }
        Target::Game { rel } => {
            let items = match collect_game_entries(v, &rel, |p| collect(p)) {
                Ok(items) => items,
                Err(e) => return Some(Err(e)),
            };
            *out = items[0];
            let h = Box::into_raw(Box::new(MergedFind { items, pos: 1 })) as HANDLE;
            MERGED.lock().unwrap().get_or_insert_with(HashSet::new).insert(h as usize);
            Some(Ok(h))
        }
    }
}

unsafe extern "system" fn hk_find_first_ex_w(n: *const u16, lvl: i32, data: *mut c_void, op: i32, filt: *const c_void, fl: u32) -> HANDLE {
    let orig = orig!(O_FIND_FIRST_EX_W as FnFindFirstExW);
    // only the common info levels/ops are virtualised. FindExSearchLimitToDirectories (1) is
    // advisory – a file system without directory filtering ignores it and lists everything –
    // so it is listed like a name match.
    if lvl > 1 || op > 1 {
        return orig(n, lvl, data, op, filt, fl);
    }
    let Some(_g) = Guard::enter() else { return orig(n, lvl, data, op, filt, fl) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, lvl, data, op, filt, fl) };
    match find_first_vfs(&path, data as *mut WIN32_FIND_DATAW) {
        None => orig(n, lvl, data, op, filt, fl),
        Some(Ok(h)) => h,
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe extern "system" fn hk_find_first_w(n: *const u16, data: *mut WIN32_FIND_DATAW) -> HANDLE {
    let orig = orig!(O_FIND_FIRST_W as FnFindFirstW);
    let Some(_g) = Guard::enter() else { return orig(n, data) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, data) };
    match find_first_vfs(&path, data) {
        None => orig(n, data),
        Some(Ok(h)) => h,
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe fn w_to_a(w: &WIN32_FIND_DATAW, a: *mut WIN32_FIND_DATAA) {
    use windows_sys::Win32::Globalization::{WideCharToMultiByte, CP_ACP};
    let a = &mut *a;
    a.dwFileAttributes = w.dwFileAttributes;
    a.ftCreationTime = w.ftCreationTime;
    a.ftLastAccessTime = w.ftLastAccessTime;
    a.ftLastWriteTime = w.ftLastWriteTime;
    a.nFileSizeHigh = w.nFileSizeHigh;
    a.nFileSizeLow = w.nFileSizeLow;
    a.dwReserved0 = w.dwReserved0;
    a.dwReserved1 = w.dwReserved1;
    a.cFileName = [0; 260];
    a.cAlternateFileName = [0; 14];
    WideCharToMultiByte(CP_ACP, 0, w.cFileName.as_ptr(), -1, a.cFileName.as_mut_ptr() as *mut u8, 260, std::ptr::null(), std::ptr::null_mut());
    WideCharToMultiByte(CP_ACP, 0, w.cAlternateFileName.as_ptr(), -1, a.cAlternateFileName.as_mut_ptr() as *mut u8, 14, std::ptr::null(), std::ptr::null_mut());
}

unsafe extern "system" fn hk_find_first_a(n: *const u8, data: *mut WIN32_FIND_DATAA) -> HANDLE {
    let orig = orig!(O_FIND_FIRST_A as FnFindFirstA);
    let Some(_g) = Guard::enter() else { return orig(n, data) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, data) };
    let mut w: WIN32_FIND_DATAW = std::mem::zeroed();
    match find_first_vfs(&path, &mut w) {
        None => orig(n, data),
        Some(Ok(h)) => {
            w_to_a(&w, data);
            h
        }
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe fn merged_next(h: HANDLE) -> Option<WIN32_FIND_DATAW> {
    let m = &mut *(h as *mut MergedFind);
    let r = m.items.get(m.pos).copied();
    m.pos += 1;
    r
}

unsafe extern "system" fn hk_find_next_w(h: HANDLE, data: *mut WIN32_FIND_DATAW) -> i32 {
    if is_merged(h) {
        return match merged_next(h) {
            Some(d) => {
                *data = d;
                1
            }
            None => {
                SetLastError(18); // ERROR_NO_MORE_FILES
                0
            }
        };
    }
    orig!(O_FIND_NEXT_W as FnFindNextW)(h, data)
}

unsafe extern "system" fn hk_find_next_a(h: HANDLE, data: *mut WIN32_FIND_DATAA) -> i32 {
    if is_merged(h) {
        return match merged_next(h) {
            Some(d) => {
                w_to_a(&d, data);
                1
            }
            None => {
                SetLastError(18);
                0
            }
        };
    }
    orig!(O_FIND_NEXT_A as FnFindNextA)(h, data)
}

unsafe extern "system" fn hk_find_close(h: HANDLE) -> i32 {
    let was = MERGED
        .lock()
        .ok()
        .and_then(|mut g| g.as_mut().map(|s| s.remove(&(h as usize))))
        .unwrap_or(false);
    if was {
        drop(Box::from_raw(h as *mut MergedFind));
        return 1;
    }
    orig!(O_FIND_CLOSE as FnFindClose)(h)
}

// ---------------------------------------------------------------------------------------

pub fn install() -> Result<(), String> {
    let kb = "kernelbase.dll";
    let list: &[(&str, usize, &AtomicUsize)] = &[
        ("CreateFileW", hk_create_file_w as usize, &O_CREATE_FILE_W),
        ("CreateFileA", hk_create_file_a as usize, &O_CREATE_FILE_A),
        ("CreateFile2", hk_create_file_2 as usize, &O_CREATE_FILE_2),
        ("GetFileAttributesW", hk_get_attr_w as usize, &O_GET_ATTR_W),
        ("GetFileAttributesA", hk_get_attr_a as usize, &O_GET_ATTR_A),
        ("GetFileAttributesExW", hk_get_attr_ex_w as usize, &O_GET_ATTR_EX_W),
        ("GetFileAttributesExA", hk_get_attr_ex_a as usize, &O_GET_ATTR_EX_A),
        ("SetFileAttributesW", hk_set_attr_w as usize, &O_SET_ATTR_W),
        ("GetDiskFreeSpaceExW", hk_disk_free_ex_w as usize, &O_DISK_FREE_EX_W),
        ("GetDiskFreeSpaceExA", hk_disk_free_ex_a as usize, &O_DISK_FREE_EX_A),
        ("GetDiskFreeSpaceW", hk_disk_free_w as usize, &O_DISK_FREE_W),
        ("GetDiskFreeSpaceA", hk_disk_free_a as usize, &O_DISK_FREE_A),
        ("CreateDirectoryW", hk_create_dir_w as usize, &O_CREATE_DIR_W),
        ("RemoveDirectoryW", hk_remove_dir_w as usize, &O_REMOVE_DIR_W),
        ("DeleteFileW", hk_delete_file_w as usize, &O_DELETE_FILE_W),
        ("ReadFile", hk_read_file as usize, &O_READ_FILE),
        ("MoveFileExW", hk_move_file_ex_w as usize, &O_MOVE_FILE_EX_W),
        ("MoveFileWithProgressW", hk_move_file_with_progress_w as usize, &O_MOVE_FILE_PROGRESS_W),
        ("CopyFileExW", hk_copy_file_ex_w as usize, &O_COPY_FILE_EX_W),
        ("CopyFileW", hk_copy_file_w as usize, &O_COPY_FILE_W),
        ("CopyFile2", hk_copy_file_2 as usize, &O_COPY_FILE_2),
        ("ReplaceFileW", hk_replace_file_w as usize, &O_REPLACE_FILE_W),
        ("FindFirstFileExW", hk_find_first_ex_w as usize, &O_FIND_FIRST_EX_W),
        ("FindFirstFileW", hk_find_first_w as usize, &O_FIND_FIRST_W),
        ("FindFirstFileA", hk_find_first_a as usize, &O_FIND_FIRST_A),
        ("FindNextFileW", hk_find_next_w as usize, &O_FIND_NEXT_W),
        ("FindNextFileA", hk_find_next_a as usize, &O_FIND_NEXT_A),
        ("FindClose", hk_find_close as usize, &O_FIND_CLOSE),
    ];
    // ANSI entries kernel32 implements without calling a hooked kernelbase function (after the
    // kernelbase list: their hooks call MoveFileWithProgressW's trampoline)
    let k32: &[(&str, usize, &AtomicUsize)] = &[
        ("MoveFileA", hk_move_file_a as usize, &O_MOVE_FILE_A),
        ("MoveFileExA", hk_move_file_ex_a as usize, &O_MOVE_FILE_EX_A),
        ("MoveFileWithProgressA", hk_move_file_with_progress_a as usize, &O_MOVE_FILE_PROGRESS_A),
    ];
    let mut errors = Vec::new();
    for (module, list) in [(kb, list), ("kernel32.dll", k32)] {
        for (name, det, slot) in list {
            if let Err(e) = unsafe { hook(module, name, *det, slot) } {
                errors.push(e);
            }
        }
    }
    if errors.is_empty() {
        crate::xlog_info!("VFS: {} file hooks active", list.len() + k32.len());
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, GENERIC_WRITE};
    use windows_sys::Win32::Storage::FileSystem::{
        CopyFile2, CopyFileExW, CopyFileW, CreateFileW, MoveFileExW, COPYFILE2_EXTENDED_PARAMETERS,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
    };

    #[test]
    fn read_file_commits_a_lazy_destination_before_kernel_io() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::ReadFile;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
        struct RestoreOriginal(usize);
        impl Drop for RestoreOriginal {
            fn drop(&mut self) { O_READ_FILE.store(self.0, Ordering::Release); }
        }
        let fixture = HookFixture::new("lazy_read");
        let file = std::fs::File::open(fixture.game().join("data").join("hello.txt")).unwrap();
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 0x10000, MEM_RESERVE, PAGE_READWRITE);
            assert!(!page.is_null());
            assert!(crate::hle::kernelx::lazy::mark(page as usize));
            let previous = O_READ_FILE.load(Ordering::Acquire);
            let _restore = RestoreOriginal(previous);
            if previous == 0 { O_READ_FILE.store(ReadFile as usize, Ordering::Release); }
            let mut read = 0u32;
            assert_ne!(hk_read_file(file.as_raw_handle(), page, 8, &mut read, std::ptr::null_mut()), 0);
            assert_eq!(read, 8);
            assert_eq!(std::slice::from_raw_parts(page as *const u8, 8), b"original");
            assert_eq!(crate::hle::kernelx::VirtualFree(page, 0, 0x8000), 1);
        }
    }

    struct HookFixture {
        dir: PathBuf,
        vfs: vfs::Vfs,
    }

    impl HookFixture {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("xcl_vfs_hooks_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let game = dir.join("game");
            let data = dir.join("xcldata");
            let temp = data.join("temp");
            std::fs::create_dir_all(game.join("data")).unwrap();
            std::fs::create_dir_all(&temp).unwrap();
            std::fs::write(game.join("data").join("hello.txt"), b"original").unwrap();
            std::fs::write(game.join("data").join("gone.txt"), b"bye").unwrap();
            let mounts = vec![
                ("G:".to_string(), game.to_string_lossy().to_string()),
                ("T:".to_string(), temp.to_string_lossy().to_string()),
            ];
            let vfs = vfs::Vfs::new(&game, &data, mounts);
            Self { dir, vfs }
        }

        fn game(&self) -> PathBuf {
            self.dir.join("game")
        }

        fn temp(&self) -> PathBuf {
            self.dir.join("xcldata").join("temp")
        }

        fn read_vfs(&self, path: &str) -> Option<Vec<u8>> {
            let real = self.vfs.real_path_for_read(path)?;
            std::fs::read(real).ok()
        }
    }

    impl Drop for HookFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    unsafe extern "system" fn quota_limited_host(
        _: *const u16,
        available: *mut u64,
        total: *mut u64,
        free: *mut u64,
    ) -> i32 {
        available.write(123);
        total.write(1024);
        free.write(2048); // Volume-wide free can exceed a caller's total.
        SetLastError(9123);
        1
    }

    #[test]
    fn pls_disk_space_caps_auto_and_oversized_quotas_to_host_capacity() {
        let fixture = HookFixture::new("pls_host_limits");
        let root = fixture.temp();
        std::fs::write(root.join("existing.bin"), [0; 256]).unwrap();
        let path = wide(&root.to_string_lossy());
        for (mb, expected_quota) in [(0, 4096 << 20), (u64::MAX, u64::MAX)] {
            let config = crate::config::Config {
                pls_mb: mb,
                ..Default::default()
            };
            let quota = config.pls_quota_bytes();
            assert_eq!(quota, expected_quota);
            let (mut available, mut total, mut free) = (0, 0, 0);
            let ok = unsafe {
                pls_disk_free_ex(
                    quota_limited_host,
                    path.as_ptr(),
                    &root,
                    quota,
                    [&mut available, &mut total, &mut free],
                )
            };
            let err = unsafe { GetLastError() };
            assert_eq!(ok, 1);
            assert_eq!(err, 9123);
            assert_eq!((available, total, free), (123, 1024, 2048));
        }
    }

    #[test]
    fn create_always_reports_existing_game_file() {
        let f = HookFixture::new("create_always");
        let overwrite = |path| {
            let decision = f.vfs.open_decision(path, GENERIC_WRITE, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL);
            let handle = apply_open_decision(
                decision,
                CREATE_ALWAYS,
                || panic!("game paths must go through the overlay"),
                |p, disposition| unsafe {
                    CreateFileW(
                        p,
                        GENERIC_WRITE,
                        0,
                        std::ptr::null(),
                        disposition,
                        FILE_ATTRIBUTE_NORMAL,
                        std::ptr::null_mut(),
                    )
                },
            );
            let error = unsafe { GetLastError() };
            assert_ne!(handle, INVALID_HANDLE_VALUE, "CreateFileW({path}) failed with {error}");
            assert_ne!(unsafe { CloseHandle(handle) }, 0);
            error
        };

        let existing = r"G:\data\hello.txt";
        assert_eq!(overwrite(existing), vfs::ERROR_ALREADY_EXISTS, "first write to an existing game file");
        assert_eq!(overwrite(existing), vfs::ERROR_ALREADY_EXISTS, "next write to its overlay file");
        assert_eq!(f.read_vfs(existing).as_deref(), Some(&b""[..]));
        assert_eq!(std::fs::read(f.game().join("data/hello.txt")).unwrap(), b"original");

        assert_eq!(overwrite(r"G:\data\new.txt"), 0);
        let decision = f.vfs.open_decision(r"G:\data\gone.txt", GENERIC_WRITE, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL);
        let handle = apply_open_decision(
            decision,
            CREATE_ALWAYS,
            || panic!("expected overlay redirection"),
            |_, _| {
                unsafe { SetLastError(vfs::ERROR_ACCESS_DENIED) };
                INVALID_HANDLE_VALUE
            },
        );
        let error = unsafe { GetLastError() };
        assert_eq!(handle, INVALID_HANDLE_VALUE);
        assert_eq!(error, vfs::ERROR_ACCESS_DENIED);

        f.vfs.delete_file(r"data\gone.txt").unwrap();
        assert_eq!(overwrite(r"G:\data\gone.txt"), 0, "creating a whited-out file");
    }

    fn real_move(s: *const u16, d: *const u16, f: u32) -> i32 {
        // SAFETY: callers pass valid NUL-terminated UTF-16 paths to Win32 MoveFileExW.
        unsafe { MoveFileExW(s, d, f) }
    }

    fn real_copy(s: *const u16, d: *const u16, fail: i32) -> i32 {
        // SAFETY: callers pass valid NUL-terminated UTF-16 paths to Win32 CopyFileW.
        unsafe { CopyFileW(s, d, fail) }
    }

    fn real_copy_ex(s: *const u16, d: *const u16, p: *const c_void, dt: *const c_void, c: *mut i32, f: u32) -> i32 {
        // SAFETY: callers pass valid NUL-terminated UTF-16 paths and null callbacks to Win32 CopyFileExW.
        unsafe { CopyFileExW(s, d, std::mem::transmute(p), dt, c, f) }
    }

    fn real_copy_2(s: *const u16, d: *const u16, params: *const c_void) -> i32 {
        // SAFETY: callers pass valid NUL-terminated UTF-16 paths and optional COPYFILE2_EXTENDED_PARAMETERS.
        unsafe { CopyFile2(s, d, params as *const COPYFILE2_EXTENDED_PARAMETERS) }
    }

    #[test]
    fn move_case_only_and_self_rename_preserve_file() {
        let f = HookFixture::new("case_move");
        let v = &f.vfs;

        // Case-only rename on an un-overlaid game file must NOT delete or whiteout it.
        let ok = unsafe { move_file_vfs(v, "G:\\data\\hello.txt", "G:\\data\\HELLO.TXT", 0, real_move) };
        assert_eq!(ok, 1);
        assert!(!v.is_whiteout("data\\hello.txt"));
        assert!(!v.is_whiteout("data\\HELLO.TXT"));
        assert_eq!(f.read_vfs("G:\\data\\HELLO.TXT"), Some(b"original".to_vec()));
        assert_eq!(f.read_vfs("G:\\data\\hello.txt"), Some(b"original".to_vec()));
        assert_eq!(std::fs::read(f.game().join("data").join("hello.txt")).unwrap(), b"original");

        // Case-only rename on an already-overlaid file preserves modified overlay content.
        let ov = v.overlay_path("data\\HELLO.TXT");
        std::fs::write(&ov, b"modified").unwrap();
        let ok = unsafe { move_file_vfs(v, "G:\\data\\HELLO.TXT", "G:\\data\\Hello.txt", 0, real_move) };
        assert_eq!(ok, 1);
        assert!(!v.is_whiteout("data\\Hello.txt"));
        assert_eq!(f.read_vfs("G:\\data\\Hello.txt"), Some(b"modified".to_vec()));
        let ov_names: Vec<String> = std::fs::read_dir(v.overlay_path("data"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        if crate::util::under_wine() {
            // Wine's MoveFileExW succeeds on a case-only rename but keeps the old name
            assert_eq!(ov_names.len(), 1);
            assert!(ov_names[0].eq_ignore_ascii_case("Hello.txt"), "{ov_names:?}");
        } else {
            assert_eq!(ov_names, vec!["Hello.txt".to_string()]);
        }

        // Exact self-rename is a safe no-op.
        let ok = unsafe { move_file_vfs(v, "G:\\data\\gone.txt", "G:\\data\\gone.txt", 0, real_move) };
        assert_eq!(ok, 1);
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"bye".to_vec()));

        // Case-only rename of a nonexistent file fails with ERROR_FILE_NOT_FOUND.
        let ok = unsafe { move_file_vfs(v, "G:\\data\\missing.txt", "G:\\data\\MISSING.TXT", 0, real_move) };
        assert_eq!(ok, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_NOT_FOUND);
    }

    #[test]
    fn move_enforces_replace_existing_and_overwrites_cleanly() {
        let f = HookFixture::new("move_replace");
        let v = &f.vfs;
        std::fs::write(f.temp().join("ext.txt"), b"from-ext").unwrap();

        // Moving onto an existing un-overlaid game file without MOVEFILE_REPLACE_EXISTING fails.
        let ok = unsafe { move_file_vfs(v, "G:\\data\\hello.txt", "G:\\data\\gone.txt", 0, real_move) };
        assert_eq!(ok, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_ALREADY_EXISTS);
        assert_eq!(f.read_vfs("G:\\data\\hello.txt"), Some(b"original".to_vec()));
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"bye".to_vec()));

        // Moving external file onto un-overlaid game file without MOVEFILE_REPLACE_EXISTING fails.
        let ok = unsafe { move_file_vfs(v, "T:\\ext.txt", "G:\\data\\hello.txt", 0, real_move) };
        assert_eq!(ok, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_ALREADY_EXISTS);
        assert_eq!(std::fs::read(f.temp().join("ext.txt")).unwrap(), b"from-ext");
        assert_eq!(f.read_vfs("G:\\data\\hello.txt"), Some(b"original".to_vec()));

        // Moving game file onto existing external file without MOVEFILE_REPLACE_EXISTING fails.
        let ok = unsafe { move_file_vfs(v, "G:\\data\\hello.txt", "T:\\ext.txt", 0, real_move) };
        assert_eq!(ok, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_ALREADY_EXISTS);
        assert_eq!(std::fs::read(f.temp().join("ext.txt")).unwrap(), b"from-ext");

        // With MOVEFILE_REPLACE_EXISTING, game->game overwrite succeeds and preserves source content.
        let ok = unsafe { move_file_vfs(v, "G:\\data\\hello.txt", "G:\\data\\gone.txt", MOVEFILE_REPLACE_EXISTING, real_move) };
        assert_eq!(ok, 1);
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"original".to_vec()));
        assert_eq!(f.read_vfs("G:\\data\\hello.txt"), None);
        assert_eq!(std::fs::read(f.game().join("data").join("gone.txt")).unwrap(), b"bye");

        // With MOVEFILE_REPLACE_EXISTING, external->game overwrite succeeds and reads back intact.
        let ok = unsafe { move_file_vfs(v, "T:\\ext.txt", "G:\\data\\gone.txt", MOVEFILE_REPLACE_EXISTING, real_move) };
        assert_eq!(ok, 1);
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"from-ext".to_vec()));
        assert!(!f.temp().join("ext.txt").exists());
    }

    #[test]
    fn move_with_locked_overlay_source_preserves_destination() {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Foundation::ERROR_SHARING_VIOLATION;
        use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, FILE_SHARE_WRITE};

        let f = HookFixture::new("locked_move");
        let v = &f.vfs;
        let source = r"G:\data\hello.txt";
        let destination = r"G:\data\gone.txt";
        let overlay_source = v.prepare_write(r"data\hello.txt", false).unwrap();
        let overlay_destination = v.prepare_write(r"data\gone.txt", false).unwrap();
        std::fs::write(&overlay_source, b"overlay source").unwrap();
        std::fs::write(overlay_destination, b"old destination").unwrap();

        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .open(overlay_source)
            .unwrap();
        let result = unsafe { move_file_vfs(v, source, destination, MOVEFILE_REPLACE_EXISTING, real_move) };
        let error = unsafe { GetLastError() };
        assert_eq!(result, 0, "the open source does not share delete access");
        assert_eq!(f.read_vfs(destination).as_deref(), Some(&b"old destination"[..]));
        assert_eq!(f.read_vfs(source).as_deref(), Some(&b"overlay source"[..]));
        assert_eq!(error, ERROR_SHARING_VIOLATION);

        drop(held);
        let result = unsafe { move_file_vfs(v, source, destination, MOVEFILE_REPLACE_EXISTING, real_move) };
        assert_ne!(result, 0, "move should succeed after the source handle closes");
        assert_eq!(f.read_vfs(destination).as_deref(), Some(&b"overlay source"[..]));
        assert_eq!(f.read_vfs(source), None, "the original stays hidden after the move");
        assert_eq!(std::fs::read(f.game().join("data/hello.txt")).unwrap(), b"original");
        assert_eq!(std::fs::read(f.game().join("data/gone.txt")).unwrap(), b"bye");
    }

    #[test]
    fn locked_overlay_move_to_temp_preserves_destination() {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, FILE_SHARE_WRITE};

        let f = HookFixture::new("locked_move_temp");
        let source = r"G:\data\hello.txt";
        let destination = r"T:\destination.txt";
        let overlay_source = f.vfs.prepare_write(r"data\hello.txt", false).unwrap();
        std::fs::write(&overlay_source, b"overlay source").unwrap();
        std::fs::write(f.temp().join("destination.txt"), b"old destination").unwrap();
        let held = std::fs::OpenOptions::new().read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE).open(&overlay_source).unwrap();

        let result = unsafe { move_file_vfs(&f.vfs, source, destination, MOVEFILE_REPLACE_EXISTING, real_move) };
        assert_eq!(result, 0);
        assert_eq!(std::fs::read(f.temp().join("destination.txt")).unwrap(), b"old destination");
        assert_eq!(f.read_vfs(source).as_deref(), Some(&b"overlay source"[..]));
        drop(held);
        assert_ne!(unsafe { move_file_vfs(&f.vfs, source, destination, MOVEFILE_REPLACE_EXISTING, real_move) }, 0);
        assert_eq!(std::fs::read(f.temp().join("destination.txt")).unwrap(), b"overlay source");
        assert_eq!(f.read_vfs(source), None);
        assert_eq!(std::fs::read(f.game().join("data/hello.txt")).unwrap(), b"original");

        // A file that exists only in the overlay has no original to white out.
        let new_source = r"G:\data\new.txt";
        let new_overlay = f.vfs.prepare_write(r"data\new.txt", false).unwrap();
        std::fs::write(new_overlay, b"new file").unwrap();
        assert_ne!(unsafe { move_file_vfs(&f.vfs, new_source, r"T:\new.txt", 0, real_move) }, 0);
        assert_eq!(std::fs::read(f.temp().join("new.txt")).unwrap(), b"new file");
        assert_eq!(f.read_vfs(new_source), None);
    }

    #[test]
    fn copy_enforces_fail_if_exists_and_overwrites_cleanly() {
        let f = HookFixture::new("copy_flags");
        let v = &f.vfs;

        // CopyFileW with bFailIfExists=TRUE onto un-overlaid original fails with ERROR_FILE_EXISTS.
        let ok = unsafe { copy_file_w_vfs(v, "G:\\data\\hello.txt", "G:\\data\\gone.txt", 1, real_copy) };
        assert_eq!(ok, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_EXISTS);
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"bye".to_vec()));

        // CopyFileExW with COPY_FILE_FAIL_IF_EXISTS onto un-overlaid original fails with ERROR_FILE_EXISTS.
        let ok = unsafe {
            copy_file_ex_w_vfs(
                v,
                "G:\\data\\hello.txt",
                "G:\\data\\gone.txt",
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                COPY_FILE_FAIL_IF_EXISTS,
                real_copy_ex,
            )
        };
        assert_eq!(ok, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_EXISTS);
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"bye".to_vec()));

        // CopyFile2 with COPY_FILE_FAIL_IF_EXISTS onto un-overlaid original returns HRESULT(ERROR_FILE_EXISTS).
        let params = COPYFILE2_EXTENDED_PARAMETERS {
            dwSize: std::mem::size_of::<COPYFILE2_EXTENDED_PARAMETERS>() as u32,
            dwCopyFlags: COPY_FILE_FAIL_IF_EXISTS,
            pfCancel: std::ptr::null_mut(),
            pProgressRoutine: None,
            pvCallbackContext: std::ptr::null_mut(),
        };
        let hr = unsafe {
            copy_file_2_vfs(
                v,
                "G:\\data\\hello.txt",
                "G:\\data\\gone.txt",
                &params as *const _ as *const c_void,
                real_copy_2,
            )
        };
        assert_eq!(hr, crate::util::hresult_from_win32(vfs::ERROR_FILE_EXISTS));
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"bye".to_vec()));

        // CopyFileW with bFailIfExists=FALSE overwrites cleanly and reads back intact.
        let ok = unsafe { copy_file_w_vfs(v, "G:\\data\\hello.txt", "G:\\data\\gone.txt", 0, real_copy) };
        assert_eq!(ok, 1);
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"original".to_vec()));
        assert_eq!(std::fs::read(f.game().join("data").join("gone.txt")).unwrap(), b"bye");

        // Deleting gone.txt makes bFailIfExists=TRUE succeed and clear the whiteout.
        assert_eq!(v.delete_file("data\\gone.txt"), Ok(()));
        assert!(v.is_whiteout("data\\gone.txt"));
        let ok = unsafe { copy_file_w_vfs(v, "G:\\data\\hello.txt", "G:\\data\\gone.txt", 1, real_copy) };
        assert_eq!(ok, 1);
        assert!(!v.is_whiteout("data\\gone.txt"));
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), Some(b"original".to_vec()));
    }

    #[test]
    fn failed_copy_and_move_do_not_resurrect_whiteouts() {
        let f = HookFixture::new("no_resurrect");
        let v = &f.vfs;
        assert_eq!(v.delete_file("data\\gone.txt"), Ok(()));
        assert!(v.is_whiteout("data\\gone.txt"));

        // Failed CopyFileW (game missing src and external missing src) must keep whiteout.
        assert_eq!(unsafe { copy_file_w_vfs(v, "G:\\data\\missing.txt", "G:\\data\\gone.txt", 0, real_copy) }, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_NOT_FOUND);
        assert!(v.is_whiteout("data\\gone.txt"));
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), None);

        assert_eq!(unsafe { copy_file_w_vfs(v, "T:\\missing.txt", "G:\\data\\gone.txt", 0, real_copy) }, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_NOT_FOUND);
        assert!(v.is_whiteout("data\\gone.txt"));
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), None);

        // Failed CopyFileExW and CopyFile2 must keep whiteout.
        assert_eq!(
            unsafe {
                copy_file_ex_w_vfs(
                    v,
                    "T:\\missing.txt",
                    "G:\\data\\gone.txt",
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    real_copy_ex,
                )
            },
            0
        );
        assert!(v.is_whiteout("data\\gone.txt"));

        let hr = unsafe { copy_file_2_vfs(v, "T:\\missing.txt", "G:\\data\\gone.txt", std::ptr::null(), real_copy_2) };
        assert!(hr < 0);
        assert!(v.is_whiteout("data\\gone.txt"));

        // Failed MoveFileExW (both external and game missing src) must keep whiteout.
        assert_eq!(unsafe { move_file_vfs(v, "T:\\missing.txt", "G:\\data\\gone.txt", MOVEFILE_REPLACE_EXISTING, real_move) }, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_NOT_FOUND);
        assert!(v.is_whiteout("data\\gone.txt"));

        assert_eq!(unsafe { move_file_vfs(v, "G:\\data\\missing.txt", "G:\\data\\gone.txt", MOVEFILE_REPLACE_EXISTING, real_move) }, 0);
        assert_eq!(unsafe { GetLastError() }, vfs::ERROR_FILE_NOT_FOUND);
        assert!(v.is_whiteout("data\\gone.txt"));
        assert_eq!(f.read_vfs("G:\\data\\gone.txt"), None);
    }

    #[test]
    fn replace_file_rejects_missing_or_whited_out_targets() {
        let f = HookFixture::new("replace_check");
        let v = &f.vfs;
        std::fs::write(f.temp().join("ext.txt"), b"replacement").unwrap();

        assert_eq!(validate_replace_targets(v, "G:\\data\\missing.txt", "G:\\data\\hello.txt"), Err(vfs::ERROR_FILE_NOT_FOUND));
        assert_eq!(validate_replace_targets(v, "G:\\data\\hello.txt", "G:\\data\\missing.txt"), Err(vfs::ERROR_FILE_NOT_FOUND));

        assert_eq!(v.delete_file("data\\gone.txt"), Ok(()));
        assert_eq!(validate_replace_targets(v, "G:\\data\\gone.txt", "T:\\ext.txt"), Err(vfs::ERROR_FILE_NOT_FOUND));
        assert_eq!(validate_replace_targets(v, "G:\\data\\hello.txt", "T:\\ext.txt"), Ok(()));
    }

    fn real_collect(pattern: &str) -> Vec<WIN32_FIND_DATAW> {
        use windows_sys::Win32::Storage::FileSystem::{FindClose, FindFirstFileExW, FindNextFileW};
        let mut out = Vec::new();
        // SAFETY: zeroed WIN32_FIND_DATAW and NUL-terminated UTF-16 pattern passed to Win32 FindFirstFileExW/FindNextFileW/FindClose.
        unsafe {
            let mut d: WIN32_FIND_DATAW = std::mem::zeroed();
            let h = FindFirstFileExW(wide(pattern).as_ptr(), 0, &mut d as *mut _ as *mut c_void, 0, std::ptr::null(), 0);
            if h == INVALID_HANDLE_VALUE {
                return out;
            }
            loop {
                out.push(d);
                if FindNextFileW(h, &mut d) == 0 {
                    break;
                }
            }
            FindClose(h);
        }
        out
    }

    #[test]
    fn find_first_on_whited_out_and_recreated_directory() {
        let f = HookFixture::new("find_dir");
        let v = &f.vfs;

        let mut before: Vec<String> = collect_game_entries(v, "data\\*", real_collect)
            .unwrap()
            .iter()
            .map(name_of)
            .collect();
        before.sort();
        assert_eq!(before, vec![".", "..", "gone.txt", "hello.txt"]);

        // Removing `data` must make wildcard listing fail with ERROR_PATH_NOT_FOUND (no "." / "..").
        assert_eq!(v.remove_dir("data"), Ok(()));
        assert_eq!(collect_game_entries(v, "data\\*", real_collect).err(), Some(vfs::ERROR_PATH_NOT_FOUND));
        assert_eq!(collect_game_entries(v, "data\\hello.txt", real_collect).err(), Some(vfs::ERROR_PATH_NOT_FOUND));
        assert_eq!(collect_game_entries(v, "data", real_collect).err(), Some(vfs::ERROR_FILE_NOT_FOUND));

        // Recreating `data` makes it listable again as an empty directory (only "." and "..").
        assert_eq!(v.create_dir("data"), Ok(()));
        let mut after: Vec<String> = collect_game_entries(v, "data\\*", real_collect)
            .unwrap()
            .iter()
            .map(name_of)
            .collect();
        after.sort();
        assert_eq!(after, vec![".", ".."]);
    }
}
