//! Opt-in Bink 2 movie skipping (`XCL_SKIP_BINK`).
//!
//! Titles that ship `Bink2Scarlett.dll` (or another `bink2*.dll`) play splash/intro movies at
//! real-time frame cadence through `BinkOpen*` / `BinkWait` / `BinkNextFrame` / `BinkDoFrame*`.
//! When `XCL_SKIP_BINK=1` is set, XCL still loads the title's own Bink DLL so memory hooks,
//! buffer sizing (`BinkGetFrameBuffersInfo`), buffer registration (`BinkRegisterFrameBuffers`),
//! and `BinkClose` behave identically, while wrapping the playback-advancement exports to
//! fast-advance non-looping movies straight to their final frame without decoding frames.
//!
//! Modes:
//! - unset / `0` / `off` / `false`: disabled (zero overhead; imports are not wrapped)
//! - `1` / `on` / `true` / `intro`: skip non-looping movies (intro/logo/cutscenes); looping
//!   movies (`BINKWILLLOOP`, bit 7 of `OpenFlags`) still forward to the real Bink DLL
//! - `all`: skip all Bink movies, including looping ones

#![allow(non_snake_case)]

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, Once, OnceLock};

/// Public header of RAD Bink 2's `BINK` struct (`bink.h` / verified against `Bink2Scarlett.dll`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BinkHeader {
    pub width: u32,          // +0x00
    pub height: u32,         // +0x04
    pub frames: u32,         // +0x08
    pub frame_num: u32,      // +0x0C
    pub last_frame_num: u32, // +0x10
    pub frame_rate: u32,     // +0x14
    pub frame_rate_div: u32, // +0x18
    pub read_error: u32,     // +0x1C
    pub open_flags: u32,     // +0x20
}

/// `BINKWILLLOOP` flag in `BinkHeader::open_flags` (set by `BinkSetWillLoop`).
pub const BINK_WILL_LOOP: u32 = 0x0000_0080;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Off,
    NonLooping,
    All,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::NonLooping => "non-looping",
            Mode::All => "all",
        }
    }
}

pub fn parse_mode(val: Option<&str>) -> Mode {
    let Some(raw) = val else { return Mode::Off };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" | "no" => Mode::Off,
        "all" => Mode::All,
        _ => Mode::NonLooping,
    }
}

pub fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("XCL_SKIP_BINK").ok().as_deref()))
}

pub fn enabled() -> bool {
    mode() != Mode::Off
}

/// True for lowercase DLL keys belonging to Bink 2 (`bink2scarlett.dll`, `bink2w64.dll`, etc.).
pub fn is_bink_dll(key: &str) -> bool {
    key.starts_with("bink2") && key.ends_with(".dll")
}

// Saved addresses of the real DLL's exports so looping movies and open/close forward cleanly.
static REAL_OPEN: AtomicUsize = AtomicUsize::new(0);
static REAL_OPEN_WITH_OPTIONS: AtomicUsize = AtomicUsize::new(0);
static REAL_CLOSE: AtomicUsize = AtomicUsize::new(0);
static REAL_WAIT: AtomicUsize = AtomicUsize::new(0);
static REAL_SHOULD_SKIP: AtomicUsize = AtomicUsize::new(0);
static REAL_NEXT_FRAME: AtomicUsize = AtomicUsize::new(0);
static REAL_DO_FRAME: AtomicUsize = AtomicUsize::new(0);
static REAL_DO_FRAME_ASYNC: AtomicUsize = AtomicUsize::new(0);
static REAL_DO_FRAME_ASYNC_MULTI: AtomicUsize = AtomicUsize::new(0);
static REAL_DO_FRAME_ASYNC_WAIT: AtomicUsize = AtomicUsize::new(0);
static REAL_GOTO: AtomicUsize = AtomicUsize::new(0);

static LOGGED_HANDLES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
static SKIPPED_MOVIES: AtomicU64 = AtomicU64::new(0);

/// If `XCL_SKIP_BINK` is enabled and `function` is a wrapped Bink 2 export, store `real_addr`
/// and return the wrapper address; otherwise return `real_addr` unchanged.
pub fn wrap(function: &str, real_addr: usize) -> usize {
    if !enabled() || real_addr == 0 {
        return real_addr;
    }
    wrap_with_mode(mode(), function, real_addr)
}

fn wrap_with_mode(m: Mode, function: &str, real_addr: usize) -> usize {
    if m == Mode::Off || real_addr == 0 {
        return real_addr;
    }
    let wrapped = match function {
        "BinkOpen" => {
            REAL_OPEN.store(real_addr, Ordering::Release);
            BinkOpen_hook as usize
        }
        "BinkOpenWithOptions" => {
            REAL_OPEN_WITH_OPTIONS.store(real_addr, Ordering::Release);
            BinkOpenWithOptions_hook as usize
        }
        "BinkClose" => {
            REAL_CLOSE.store(real_addr, Ordering::Release);
            BinkClose_hook as usize
        }
        "BinkWait" => {
            REAL_WAIT.store(real_addr, Ordering::Release);
            BinkWait_hook as usize
        }
        "BinkShouldSkip" => {
            REAL_SHOULD_SKIP.store(real_addr, Ordering::Release);
            BinkShouldSkip_hook as usize
        }
        "BinkNextFrame" => {
            REAL_NEXT_FRAME.store(real_addr, Ordering::Release);
            BinkNextFrame_hook as usize
        }
        "BinkDoFrame" => {
            REAL_DO_FRAME.store(real_addr, Ordering::Release);
            BinkDoFrame_hook as usize
        }
        "BinkDoFrameAsync" => {
            REAL_DO_FRAME_ASYNC.store(real_addr, Ordering::Release);
            BinkDoFrameAsync_hook as usize
        }
        "BinkDoFrameAsyncMulti" => {
            REAL_DO_FRAME_ASYNC_MULTI.store(real_addr, Ordering::Release);
            BinkDoFrameAsyncMulti_hook as usize
        }
        "BinkDoFrameAsyncWait" => {
            REAL_DO_FRAME_ASYNC_WAIT.store(real_addr, Ordering::Release);
            BinkDoFrameAsyncWait_hook as usize
        }
        "BinkGoto" => {
            REAL_GOTO.store(real_addr, Ordering::Release);
            BinkGoto_hook as usize
        }
        _ => return real_addr,
    };
    static BANNER: Once = Once::new();
    BANNER.call_once(|| {
        crate::xlog_info!("Bink: XCL_SKIP_BINK={} – fast-advancing Bink 2 movies", m.as_str());
    });
    wrapped
}

#[inline]
unsafe fn should_skip_handle(bink: *mut BinkHeader, m: Mode) -> bool {
    if bink.is_null() || m == Mode::Off {
        return false;
    }
    let h = &*bink;
    if h.frames == 0 {
        return false;
    }
    match m {
        Mode::Off => false,
        Mode::All => true,
        Mode::NonLooping => (h.open_flags & BINK_WILL_LOOP) == 0,
    }
}

/// Fast-advance `FrameNum` and `LastFrameNum` to `Frames` so titles polling either field
/// observe the movie as finished on its first playback tick.
#[inline]
pub unsafe fn fast_advance(bink: *mut BinkHeader, m: Mode) -> bool {
    if !should_skip_handle(bink, m) {
        return false;
    }
    let h = &mut *bink;
    let first_time = {
        let mut g = LOGGED_HANDLES.lock().unwrap();
        g.get_or_insert_with(HashSet::new).insert(bink as usize)
    };
    if first_time {
        let n = SKIPPED_MOVIES.fetch_add(1, Ordering::Relaxed) + 1;
        let fps = if h.frame_rate_div != 0 {
            h.frame_rate as f64 / h.frame_rate_div as f64
        } else {
            0.0
        };
        let dur_s = if fps > 0.0 { h.frames as f64 / fps } else { 0.0 };
        crate::xlog_info!(
            "Bink: skipping movie #{n} ({}x{}, {} frames, {:.2} fps, ~{:.1} s, flags {:#x})",
            h.width,
            h.height,
            h.frames,
            fps,
            dur_s,
            h.open_flags
        );
    }
    h.frame_num = h.frames;
    h.last_frame_num = h.frames;
    true
}

type OpenFn = unsafe extern "system" fn(*const u8, u32) -> *mut BinkHeader;
type OpenWithOptionsFn = unsafe extern "system" fn(*const u8, *const c_void, u32) -> *mut BinkHeader;
type CloseFn = unsafe extern "system" fn(*mut BinkHeader);
type WaitFn = unsafe extern "system" fn(*mut BinkHeader) -> i32;
type ShouldSkipFn = unsafe extern "system" fn(*mut BinkHeader) -> i32;
type NextFrameFn = unsafe extern "system" fn(*mut BinkHeader);
type DoFrameFn = unsafe extern "system" fn(*mut BinkHeader) -> i32;
type DoFrameAsyncFn = unsafe extern "system" fn(*mut BinkHeader, u32, u32) -> i32;
type DoFrameAsyncMultiFn = unsafe extern "system" fn(*mut BinkHeader, *const u32, i32) -> i32;
type DoFrameAsyncWaitFn = unsafe extern "system" fn(*mut BinkHeader, i32) -> i32;
type GotoFn = unsafe extern "system" fn(*mut BinkHeader, u32, u32);

unsafe extern "system" fn BinkOpen_hook(name: *const u8, flags: u32) -> *mut BinkHeader {
    let real = REAL_OPEN.load(Ordering::Acquire);
    if real == 0 {
        return std::ptr::null_mut();
    }
    let f: OpenFn = std::mem::transmute(real);
    let bink = f(name, flags);
    if !bink.is_null() {
        let h = &*bink;
        crate::xlog_debug!(
            "Bink: BinkOpen -> {:p} ({}x{}, {} frames, flags {:#x})",
            bink,
            h.width,
            h.height,
            h.frames,
            h.open_flags
        );
    }
    bink
}

unsafe extern "system" fn BinkOpenWithOptions_hook(name: *const u8, opts: *const c_void, flags: u32) -> *mut BinkHeader {
    let real = REAL_OPEN_WITH_OPTIONS.load(Ordering::Acquire);
    if real == 0 {
        return std::ptr::null_mut();
    }
    let f: OpenWithOptionsFn = std::mem::transmute(real);
    let bink = f(name, opts, flags);
    if !bink.is_null() {
        let h = &*bink;
        crate::xlog_debug!(
            "Bink: BinkOpenWithOptions -> {:p} ({}x{}, {} frames, flags {:#x})",
            bink,
            h.width,
            h.height,
            h.frames,
            h.open_flags
        );
    }
    bink
}

unsafe extern "system" fn BinkClose_hook(bink: *mut BinkHeader) {
    if !bink.is_null() {
        if let Some(set) = LOGGED_HANDLES.lock().unwrap().as_mut() {
            set.remove(&(bink as usize));
        }
    }
    let real = REAL_CLOSE.load(Ordering::Acquire);
    if real != 0 {
        let f: CloseFn = std::mem::transmute(real);
        f(bink);
    }
}

unsafe extern "system" fn BinkWait_hook(bink: *mut BinkHeader) -> i32 {
    if should_skip_handle(bink, mode()) {
        // Ready immediately: do not pace the skipped movie against wall-clock time.
        return 0;
    }
    let real = REAL_WAIT.load(Ordering::Acquire);
    if real != 0 {
        let f: WaitFn = std::mem::transmute(real);
        return f(bink);
    }
    0
}

unsafe extern "system" fn BinkShouldSkip_hook(bink: *mut BinkHeader) -> i32 {
    if should_skip_handle(bink, mode()) {
        return 0;
    }
    let real = REAL_SHOULD_SKIP.load(Ordering::Acquire);
    if real != 0 {
        let f: ShouldSkipFn = std::mem::transmute(real);
        return f(bink);
    }
    0
}

unsafe extern "system" fn BinkNextFrame_hook(bink: *mut BinkHeader) {
    if fast_advance(bink, mode()) {
        return;
    }
    let real = REAL_NEXT_FRAME.load(Ordering::Acquire);
    if real != 0 {
        let f: NextFrameFn = std::mem::transmute(real);
        f(bink);
    }
}

unsafe extern "system" fn BinkDoFrame_hook(bink: *mut BinkHeader) -> i32 {
    if fast_advance(bink, mode()) {
        return 0;
    }
    let real = REAL_DO_FRAME.load(Ordering::Acquire);
    if real != 0 {
        let f: DoFrameFn = std::mem::transmute(real);
        return f(bink);
    }
    0
}

unsafe extern "system" fn BinkDoFrameAsync_hook(bink: *mut BinkHeader, thread1: u32, thread2: u32) -> i32 {
    if fast_advance(bink, mode()) {
        return 1;
    }
    let real = REAL_DO_FRAME_ASYNC.load(Ordering::Acquire);
    if real != 0 {
        let f: DoFrameAsyncFn = std::mem::transmute(real);
        return f(bink, thread1, thread2);
    }
    1
}

unsafe extern "system" fn BinkDoFrameAsyncMulti_hook(bink: *mut BinkHeader, threads: *const u32, thread_count: i32) -> i32 {
    if fast_advance(bink, mode()) {
        return 1;
    }
    let real = REAL_DO_FRAME_ASYNC_MULTI.load(Ordering::Acquire);
    if real != 0 {
        let f: DoFrameAsyncMultiFn = std::mem::transmute(real);
        return f(bink, threads, thread_count);
    }
    1
}

unsafe extern "system" fn BinkDoFrameAsyncWait_hook(bink: *mut BinkHeader, us: i32) -> i32 {
    if fast_advance(bink, mode()) {
        return 1;
    }
    let real = REAL_DO_FRAME_ASYNC_WAIT.load(Ordering::Acquire);
    if real != 0 {
        let f: DoFrameAsyncWaitFn = std::mem::transmute(real);
        return f(bink, us);
    }
    1
}

unsafe extern "system" fn BinkGoto_hook(bink: *mut BinkHeader, frame: u32, flags: u32) {
    if fast_advance(bink, mode()) {
        return;
    }
    let real = REAL_GOTO.load(Ordering::Acquire);
    if real != 0 {
        let f: GotoFn = std::mem::transmute(real);
        f(bink, frame, flags);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bink_header_abi_offsets() {
        assert_eq!(std::mem::offset_of!(BinkHeader, width), 0x00);
        assert_eq!(std::mem::offset_of!(BinkHeader, height), 0x04);
        assert_eq!(std::mem::offset_of!(BinkHeader, frames), 0x08);
        assert_eq!(std::mem::offset_of!(BinkHeader, frame_num), 0x0C);
        assert_eq!(std::mem::offset_of!(BinkHeader, last_frame_num), 0x10);
        assert_eq!(std::mem::offset_of!(BinkHeader, frame_rate), 0x14);
        assert_eq!(std::mem::offset_of!(BinkHeader, frame_rate_div), 0x18);
        assert_eq!(std::mem::offset_of!(BinkHeader, read_error), 0x1C);
        assert_eq!(std::mem::offset_of!(BinkHeader, open_flags), 0x20);
    }

    #[test]
    fn mode_and_dll_matching() {
        assert_eq!(parse_mode(None), Mode::Off);
        assert_eq!(parse_mode(Some("")), Mode::Off);
        assert_eq!(parse_mode(Some("0")), Mode::Off);
        assert_eq!(parse_mode(Some("off")), Mode::Off);
        assert_eq!(parse_mode(Some("false")), Mode::Off);
        assert_eq!(parse_mode(Some("1")), Mode::NonLooping);
        assert_eq!(parse_mode(Some("intro")), Mode::NonLooping);
        assert_eq!(parse_mode(Some("ALL")), Mode::All);

        assert!(is_bink_dll("bink2scarlett.dll"));
        assert!(is_bink_dll("bink2w64.dll"));
        assert!(!is_bink_dll("kernelx.dll"));
        assert!(!is_bink_dll("bink2scarlett.exe"));
    }

    #[test]
    fn fast_advance_respects_loop_flag_and_mode() {
        let mut intro = BinkHeader {
            width: 1920,
            height: 1080,
            frames: 600,
            frame_num: 1,
            last_frame_num: 0,
            frame_rate: 30000,
            frame_rate_div: 1001,
            read_error: 0,
            open_flags: 0,
        };
        unsafe {
            assert!(!fast_advance(&mut intro, Mode::Off));
            assert_eq!((intro.frame_num, intro.last_frame_num), (1, 0));

            assert!(fast_advance(&mut intro, Mode::NonLooping));
            assert_eq!((intro.frame_num, intro.last_frame_num), (600, 600));
        }

        let mut looping = BinkHeader {
            frames: 300,
            frame_num: 1,
            last_frame_num: 0,
            open_flags: BINK_WILL_LOOP,
            ..Default::default()
        };
        unsafe {
            assert!(!fast_advance(&mut looping, Mode::NonLooping));
            assert_eq!((looping.frame_num, looping.last_frame_num), (1, 0));

            assert!(fast_advance(&mut looping, Mode::All));
            assert_eq!((looping.frame_num, looping.last_frame_num), (300, 300));
        }
    }

    #[test]
    fn wrap_only_intercepts_known_exports_when_enabled() {
        let dummy = 0x1234_5678usize;
        assert_eq!(wrap_with_mode(Mode::Off, "BinkNextFrame", dummy), dummy);
        assert_eq!(wrap_with_mode(Mode::NonLooping, "BinkSetMemory", dummy), dummy);
        let hooked = wrap_with_mode(Mode::NonLooping, "BinkNextFrame", dummy);
        assert_eq!(hooked, BinkNextFrame_hook as usize);
        assert_eq!(REAL_NEXT_FRAME.load(Ordering::Acquire), dummy);
    }
}
