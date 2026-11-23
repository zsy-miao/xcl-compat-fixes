//! XCL test title – a fake "console executable".
//!
//! Its import table names console DLLs (kernelx.dll, xgameruntime.dll, d3d12_xs.dll) and
//! a DLL shipped "with the game" (testgame_mw.dll). Plain Windows cannot start it; XCL
//! can. Every check prints [PASS]/[FAIL]; the exit code is the number of failures.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

type HRESULT = i32;

#[repr(C)]
struct TitleMemoryStatus {
    dw_length: u32,
    dw_reserved: u32,
    v: [u64; 9],
}

#[repr(C)]
struct SystemOsVersionInfo {
    major: u8,
    minor: u8,
    build: u16,
    revision: u16,
}

// Fortnite imports these from xmem.dll (XCL finds console functions by name)
#[link(name = "xmem", kind = "raw-dylib")]
extern "system" {
    fn XMemGetWorkingSetStatistics(flags: u32, stats: *mut u64) -> i32;
    fn XMemAllocatePhysicalPages(attrs: u64, count: *mut usize, pages: *mut u64) -> i32;
    fn XMemMapPhysicalPages(va: *mut c_void, count: usize, pages: *mut u64) -> *mut c_void;
    fn XMemFreePhysicalPages(count: usize, pages: *mut u64) -> i32;
}

#[link(name = "kernelx", kind = "raw-dylib")]
extern "system" {
    fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void;
    fn XMemFree(p: *mut c_void, attrs: u64);
    fn XMemVirtualAlloc(addr: *const c_void, size: usize, t: u32, attrs: u64, protect: u32) -> *mut c_void;
    fn VirtualFree(p: *mut c_void, size: usize, t: u32) -> i32;
    fn GetConsoleType() -> u32;
    fn SetThreadName(h: *mut c_void, name: *const u16) -> i32;
    fn GetThreadName(h: *mut c_void, buf: *mut u16, len: usize, ret: *mut usize) -> i32;
    fn TitleMemoryStatus(buf: *mut TitleMemoryStatus) -> i32;
    fn GetSystemOSVersion(out: *mut SystemOsVersionInfo);
    /// Not implemented anywhere – must become a logging stub returning 0.
    fn XclDefinitelyMissingFunction(a: u64, b: u64) -> u64;
}

// Same DLL/names as Rust std's own imports, so no duplicate-symbol conflicts; XCL routes
// kernel32 to its kernelx module, so these hit XCL's shims.
#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn GetCurrentThread() -> *mut c_void;
    fn GetModuleHandleW(name: *const u16) -> usize;
    fn GetProcAddress(h: usize, name: *const u8) -> usize;
    fn LoadLibraryW(name: *const u16) -> usize;
}

// The API set console titles import these from (Minecraft does), not kernel32.
#[link(name = "api-ms-win-core-file-l1-1-0", kind = "raw-dylib")]
extern "system" {
    fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    fn GetDiskFreeSpaceW(root: *const u16, spc: *mut u32, bps: *mut u32, free: *mut u32, total: *mut u32) -> i32;
}

// the console's ntuser window API sets, as titles import them (Unicode and ANSI)
#[repr(C)]
struct WndClassW {
    style: u32,
    proc_: usize,
    cls_extra: i32,
    wnd_extra: i32,
    instance: usize,
    icon: usize,
    cursor: usize,
    background: usize,
    menu: *const u16,
    class: *const u16,
}

// user32 helpers; CreateWindowExW / A come from the console's API set names through
// GetProcAddress (an import of the same name would be bound to user32 by the linker)
#[link(name = "user32", kind = "raw-dylib")]
extern "system" {
    fn RegisterClassW(c: *const WndClassW) -> u16;
    fn DestroyWindow(h: *mut c_void) -> i32;
    fn GetWindowLongW(h: *mut c_void, index: i32) -> i32;
    fn GetClientRect(h: *mut c_void, r: *mut [i32; 4]) -> i32;
    fn DefWindowProcW(h: *mut c_void, m: u32, w: usize, l: isize) -> isize;
}

type CreateWindowExFn<C> = unsafe extern "system" fn(u32, *const C, *const C, u32, i32, i32, i32, i32, *mut c_void, *mut c_void, usize, *const c_void) -> *mut c_void;

#[link(name = "dwmapi", kind = "raw-dylib")]
extern "system" {
    fn DwmGetWindowAttribute(h: *mut c_void, attr: u32, value: *mut c_void, size: u32) -> i32;
}

#[link(name = "gameinput", kind = "raw-dylib")]
extern "system" {
    fn GameInputCreate(out: *mut *mut c_void) -> HRESULT;
}

#[link(name = "xgameruntime", kind = "raw-dylib")]
extern "system" {
    fn XGameRuntimeInitialize() -> HRESULT;
    fn XSystemGetDeviceType() -> u32;
    fn XTaskQueueCreate(work: u32, completion: u32, out: *mut *mut c_void) -> HRESULT;
    fn XTaskQueueGetPort(q: *mut c_void, port: u32, out: *mut *mut c_void) -> HRESULT;
    fn XTaskQueueCreateComposite(work: *mut c_void, completion: *mut c_void, out: *mut *mut c_void) -> HRESULT;
    fn XTaskQueueSubmitCallback(q: *mut c_void, port: u32, ctx: *mut c_void, cb: unsafe extern "system" fn(*mut c_void, bool)) -> HRESULT;
    fn XTaskQueueSubmitDelayedCallback(q: *mut c_void, port: u32, ms: u32, ctx: *mut c_void, cb: unsafe extern "system" fn(*mut c_void, bool)) -> HRESULT;
    fn XTaskQueueDispatch(q: *mut c_void, port: u32, timeout: u32) -> bool;
    fn XTaskQueueTerminate(q: *mut c_void, wait: bool, ctx: *mut c_void, cb: usize) -> HRESULT;
    fn XTaskQueueCloseHandle(q: *mut c_void);
    fn XAsyncGetStatus(b: *mut XAsyncBlock, wait: bool) -> HRESULT;
    fn XUserAddAsync(options: u32, b: *mut XAsyncBlock) -> HRESULT;
    fn XUserAddResult(b: *mut XAsyncBlock, user: *mut *mut c_void) -> HRESULT;
    fn XUserFindForDevice(device: *const c_void, user: *mut *mut c_void) -> HRESULT;
    fn XUserGetTokenAndSignatureAsync(
        u: *mut c_void,
        options: u32,
        method: *const u8,
        url: *const u8,
        header_count: usize,
        headers: *const c_void,
        body_size: usize,
        body: *const c_void,
        b: *mut XAsyncBlock,
    ) -> HRESULT;
    fn XUserGetTokenAndSignatureResultSize(b: *mut XAsyncBlock, size: *mut usize) -> HRESULT;
    fn XUserGetGamertag(u: *mut c_void, comp: u32, size: usize, buf: *mut u8, used: *mut usize) -> HRESULT;
    fn XUserCloseHandle(u: *mut c_void);
    fn XNetworkingQueryPreferredLocalUdpMultiplayerPort(port: *mut u16) -> HRESULT;
    fn XNetworkingGetConnectivityHint(out: *mut u8) -> HRESULT;
    fn XGameSaveFilesGetFolderWithUiAsync(u: *mut c_void, scid: *const u8, b: *mut XAsyncBlock) -> HRESULT;
    fn XGameSaveFilesGetFolderWithUiResult(b: *mut XAsyncBlock, size: usize, buf: *mut u8) -> HRESULT;
    fn XPersistentLocalStorageGetPath(size: usize, buf: *mut u8, used: *mut usize) -> HRESULT;
    fn XPackageGetCurrentProcessPackageIdentifier(size: usize, buf: *mut u8) -> HRESULT;
}

mod audio;
mod gpu;

#[link(name = "testgame_mw", kind = "raw-dylib")]
extern "C" {
    fn mw_add(a: i32, b: i32) -> i32;
    fn mw_attached() -> u32;
    fn mw_tls_bump() -> u32;
    fn mw_xmem_roundtrip() -> u32;
}

#[link(name = "api-ms-win-crt-stdio-l1-1-0", kind = "raw-dylib")]
extern "C" {
    fn fopen(name: *const u8, mode: *const u8) -> *mut c_void;
    fn fread(buf: *mut c_void, size: usize, n: usize, f: *mut c_void) -> usize;
    fn fclose(f: *mut c_void) -> i32;
}

extern "C" {
    static __ImageBase: u8;
}

#[repr(C)]
struct XAsyncBlock {
    queue: *mut c_void,
    context: *mut c_void,
    callback: usize,
    internal: [u8; 32],
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn cstr(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

thread_local! {
    static TL_VALUE: Cell<u64> = const { Cell::new(0x1122_3344_5566_7788) };
}

static FAILURES: AtomicU32 = AtomicU32::new(0);

fn check(name: &str, f: impl FnOnce() -> Result<(), String>) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => println!("[PASS] {name}"),
        Ok(Err(e)) => {
            println!("[FAIL] {name}: {e}");
            FAILURES.fetch_add(1, Ordering::SeqCst);
        }
        Err(_) => {
            println!("[FAIL] {name}: panic");
            FAILURES.fetch_add(1, Ordering::SeqCst);
        }
    }
}

macro_rules! ensure {
    ($c:expr, $($m:tt)*) => { if !$c { return Err(format!($($m)*)); } };
}

static FLAG: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn set_flag(ctx: *mut c_void, canceled: bool) {
    if !canceled {
        FLAG.fetch_add(ctx as usize, Ordering::SeqCst);
    }
}

fn main() {
    println!("xcl_testtitle: start");

    check("argv: command line belongs to the game", || {
        let args: Vec<String> = std::env::args().collect();
        ensure!(args[0].to_lowercase().ends_with("xcl_testtitle.exe"), "argv[0] = {}", args[0]);
        ensure!(args.iter().any(|a| a == "--hello"), "args = {args:?}");
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        ensure!(exe.file_name().unwrap().to_string_lossy().eq_ignore_ascii_case("xcl_testtitle.exe"), "current_exe = {exe:?}");
        Ok(())
    });

    check("static TLS: template on main thread", || {
        ensure!(TL_VALUE.with(|v| v.get()) == 0x1122_3344_5566_7788, "got {:#x}", TL_VALUE.with(|v| v.get()));
        TL_VALUE.with(|v| v.set(1));
        Ok(())
    });

    check("static TLS: every new thread gets its own initialised copy", || {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    let first = TL_VALUE.with(|v| v.get());
                    TL_VALUE.with(|v| v.set(100 + i));
                    std::thread::sleep(Duration::from_millis(10));
                    (first, TL_VALUE.with(|v| v.get()), i)
                })
            })
            .collect();
        for h in handles {
            let (first, later, i) = h.join().map_err(|_| "thread panicked")?;
            ensure!(first == 0x1122_3344_5566_7788, "thread {i} started with {first:#x}");
            ensure!(later == 100 + i, "thread {i} saw {later} (shared TLS?)");
        }
        ensure!(TL_VALUE.with(|v| v.get()) == 1, "main thread value changed");
        Ok(())
    });

    check("C++-style exceptions: panic is caught (RtlPcToFileHeader)", || {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(|| panic!("boom"));
        std::panic::set_hook(prev);
        ensure!(r.is_err(), "panic was not caught");
        Ok(())
    });

    check("CPU: exact retail CLWB encoding resumes and preserves memory", || unsafe {
        #[repr(align(64))]
        struct CacheLines([u8; 192]);
        let mut lines = Box::new(CacheLines([0xA5; 192]));
        let line = lines.0.as_mut_ptr().add(64);
        // clwb [r9], including the prefixes from the retail crash. On a CPU without CLWB,
        // XCL's vectored handler must advance past all five bytes and resume this title.
        std::arch::asm!(".byte 0x66, 0x41, 0x0F, 0xAE, 0x31", in("r9") line, options(nostack, preserves_flags));
        ensure!(lines.0.iter().all(|b| std::ptr::read_volatile(b) == 0xA5), "CLWB changed the buffer");
        Ok(())
    });

    check("CPU: CLZERO zeros one aligned cache line and preserves its neighbors", || unsafe {
        #[repr(align(64))]
        struct CacheLines([u8; 192]);
        let mut lines = Box::new(CacheLines([0xA5; 192]));
        // The instruction rounds an interior address down to its 64-byte cache line.
        let inside_line = lines.0.as_mut_ptr().add(64 + 17);
        std::arch::asm!(".byte 0x0F, 0x01, 0xFC", in("rax") inside_line, options(nostack, preserves_flags));
        ensure!(lines.0[64..128].iter().all(|b| std::ptr::read_volatile(b) == 0), "CLZERO did not zero the selected line");
        ensure!(lines.0[..64].iter().chain(&lines.0[128..]).all(|b| std::ptr::read_volatile(b) == 0xA5), "CLZERO changed a neighboring line");
        Ok(())
    });

    check("CPU: CLZERO batches preserve guarded bytes, registers and CMP flags", || unsafe {
        const MARKER: u64 = 0x98AB_CDEF_7654_3210;
        // Each body is the genuine retail do-while loop, including its short branch back
        // to CLZERO. The handler may finish one bounded batch, then resumes at native CMP.
        unsafe fn run_loop(register: u8, start: usize, bound: usize) -> (usize, usize, u64, u64) {
            let mut cursor = start;
            let mut limit = bound;
            let mut marker = MARKER;
            let flags: u64;
            macro_rules! body {
                ($reg:tt, $compare:literal) => {
                    std::arch::asm!(
                        ".byte 0x0F, 0x01, 0xFC",
                        ".byte 0x48, 0x83, 0xC0, 0x40",
                        $compare,
                        ".byte 0x72, 0xF4",
                        "pushfq",
                        "pop rcx",
                        inout("rax") cursor,
                        inout($reg) limit,
                        inout("r11") marker,
                        lateout("rcx") flags,
                    )
                };
            }
            match register {
                10 => body!("r10", ".byte 0x4C, 0x39, 0xD0"),
                9 => body!("r9", ".byte 0x4C, 0x39, 0xC8"),
                8 => body!("r8", ".byte 0x4C, 0x39, 0xC0"),
                2 => body!("rdx", ".byte 0x48, 0x39, 0xD0"),
                _ => unreachable!(),
            }
            (cursor, limit, marker, flags)
        }

        // A nonaligned bound, exact equality, and empty/backwards bounds check do-while
        // semantics. The last case crosses the handler's 1 MiB cap and must take two batches.
        for (register, name) in [(10, "r10"), (9, "r9"), (8, "r8"), (2, "rdx")] {
            for (case, cleared_bytes, bound_delta) in [
                ("nonaligned end", 192usize, 185isize),
                ("equal final pointer", 192, 192),
                ("equal initial bound", 64, 0),
                ("backwards bound", 64, -17),
                ("multiple bounded batches", (1 << 20) + 128, (1 << 20) + 121),
            ] {
                let mut bytes = vec![0xA5u8; cleared_bytes + 256];
                let base = bytes.as_mut_ptr() as usize;
                let first_line = (base + 127) & !63;
                let start = first_line + 17;
                let bound = if bound_delta >= 0 { start + bound_delta as usize } else { start - (-bound_delta) as usize };
                let expected_cursor = start + cleared_bytes;
                let (cursor, returned_bound, marker, flags) = run_loop(register, start, bound);
                ensure!(cursor == expected_cursor, "CLZERO {name}, {case}: RAX {cursor:#x}, expected {expected_cursor:#x}");
                ensure!(returned_bound == bound && marker == MARKER, "CLZERO {name}, {case}: changed another register");

                let expected_flags: u64;
                std::arch::asm!(
                    "cmp {left}, {right}", "pushfq", "pop {flags}",
                    left = in(reg) expected_cursor,
                    right = in(reg) bound,
                    flags = lateout(reg) expected_flags,
                );
                const CMP_FLAGS: u64 = 0x8D5; // CF, PF, AF, ZF, SF, OF
                ensure!(flags & CMP_FLAGS == expected_flags & CMP_FLAGS, "CLZERO {name}, {case}: CMP flags {flags:#x}, expected {expected_flags:#x}");
                let first = first_line - base;
                for (offset, byte) in bytes.iter().enumerate() {
                    let expected = if (first..first + cleared_bytes).contains(&offset) { 0 } else { 0xA5 };
                    ensure!(std::ptr::read_volatile(byte) == expected, "CLZERO {name}, {case}: wrong byte at {offset:#x}");
                }
            }
        }
        Ok(())
    });

    check("CPU: SSE4a INSERTQ and EXTRQ preserve defined bits, sources and flags", || unsafe {
        // Reference: AMD's INSERTQ/EXTRQ instruction definitions and LLVM ammintrin.h.
        // Immediate length/index use six bits; zero length means 64. Upper destination
        // bits are undefined, so only the defined lower 64 bits are checked.
        #[inline(never)]
        unsafe fn run<const INSERT: bool, const LENGTH: u8, const INDEX: u8, const HIGH: bool>(
            destination: u64,
            source_low: u64,
            expected: u64,
            name: &str,
        ) -> Result<(), String> {
            use std::arch::x86_64::__m128i;
            const NEIGHBOR: [u64; 2] = [0x8BAD_F00D_7654_3210, 0x0123_4567_89AB_CDEF];
            const MARKER: u64 = 0x8765_4321_AABB_CCDD;
            let source_words = [source_low, 0xD00D_1234_9876_ABCD];
            let mut result: __m128i = std::mem::transmute([destination, 0xA55A_2468_1357_FEDCu64]);
            let mut source: __m128i = std::mem::transmute(source_words);
            let mut neighbor: __m128i = std::mem::transmute(NEIGHBOR);
            let mut marker = MARKER;
            let flags_before: u64;
            let flags_after: u64;
            macro_rules! instruction {
                ($destination:literal, $source:literal, $bytes:literal) => {
                    std::arch::asm!(
                        "cmp rax, rdx",
                        "pushfq", "pop r8",
                        $bytes,
                        "pushfq", "pop rcx",
                        length = const LENGTH,
                        index = const INDEX,
                        in("rax") 7usize,
                        in("rdx") 8usize,
                        inout("r11") marker,
                        lateout("r8") flags_before,
                        lateout("rcx") flags_after,
                        inout($destination) result,
                        inout($source) source,
                        inout("xmm2") neighbor,
                    )
                };
            }
            match (INSERT, HIGH) {
                // The first form includes the exact retail crash instruction.
                (true, false) => instruction!("xmm0", "xmm1", ".byte 0xF2, 0x0F, 0x78, 0xC1, {length}, {index}"),
                (true, true) => instruction!("xmm8", "xmm9", ".byte 0xF2, 0x45, 0x0F, 0x78, 0xC1, {length}, {index}"),
                (false, false) => instruction!("xmm0", "xmm1", ".byte 0x66, 0x0F, 0x78, 0xC0, {length}, {index}"),
                (false, true) => instruction!("xmm8", "xmm1", ".byte 0x66, 0x41, 0x0F, 0x78, 0xC0, {length}, {index}"),
            }
            let result: [u64; 2] = std::mem::transmute(result);
            let source: [u64; 2] = std::mem::transmute(source);
            let neighbor: [u64; 2] = std::mem::transmute(neighbor);
            ensure!(result[0] == expected, "SSE4a {name}: lower result {:#x}, expected {expected:#x}", result[0]);
            ensure!(source == source_words, "SSE4a {name}: source XMM changed");
            ensure!(neighbor == NEIGHBOR && marker == MARKER, "SSE4a {name}: another register changed");
            ensure!(flags_before == flags_after, "SSE4a {name}: flags changed from {flags_before:#x} to {flags_after:#x}");
            Ok(())
        }

        const DESTINATION: u64 = 0x1122_3344_5566_7788;
        const SOURCE: u64 = 0xFEDC_BA98_7654_3210;
        run::<true, 16, 16, false>(DESTINATION, SOURCE, 0x1122_3344_3210_7788, "INSERTQ retail 16/16")?;
        run::<true, 0x50, 0x50, false>(DESTINATION, SOURCE, 0x1122_3344_3210_7788, "INSERTQ six-bit controls")?;
        run::<true, 0, 0, false>(DESTINATION, SOURCE, SOURCE, "INSERTQ zero means 64")?;
        run::<true, 0x40, 0x40, false>(DESTINATION, SOURCE, SOURCE, "INSERTQ masked zero controls")?;
        run::<true, 1, 63, false>(DESTINATION, 1, 0x9122_3344_5566_7788, "INSERTQ highest bit")?;
        run::<true, 8, 24, false>(DESTINATION, SOURCE, 0x1122_3344_1066_7788, "INSERTQ keeps outside bits")?;
        run::<true, 16, 16, true>(DESTINATION, SOURCE, 0x1122_3344_3210_7788, "INSERTQ high XMM registers")?;

        run::<false, 16, 16, false>(DESTINATION, SOURCE, 0x5566, "EXTRQ 16/16")?;
        run::<false, 0x50, 0x50, false>(DESTINATION, SOURCE, 0x5566, "EXTRQ six-bit controls")?;
        run::<false, 0, 0, false>(DESTINATION, SOURCE, DESTINATION, "EXTRQ zero means 64")?;
        run::<false, 0x40, 0x40, false>(DESTINATION, SOURCE, DESTINATION, "EXTRQ masked zero controls")?;
        run::<false, 1, 63, false>(0x9122_3344_5566_7788, SOURCE, 1, "EXTRQ highest bit")?;
        run::<false, 8, 24, false>(DESTINATION, SOURCE, 0x55, "EXTRQ clears remaining lower bits")?;
        run::<false, 16, 16, true>(DESTINATION, SOURCE, 0x5566, "EXTRQ high XMM destination")?;
        Ok(())
    });

    check("kernelx: GetConsoleType / XSystemGetDeviceType", || unsafe {
        ensure!(GetConsoleType() == 6, "GetConsoleType = {}", GetConsoleType());
        ensure!(XSystemGetDeviceType() == 7, "device type = {}", XSystemGetDeviceType());
        let mut v = SystemOsVersionInfo { major: 0, minor: 0, build: 0, revision: 0 };
        GetSystemOSVersion(&mut v);
        ensure!(v.major >= 10, "os major {}", v.major);
        Ok(())
    });

    check("kernelx: XMemAlloc alignment + zeroing", || unsafe {
        let attrs = 8u64 << 24; // 256-byte alignment, heap
        let p = XMemAlloc(1000, attrs);
        ensure!(!p.is_null(), "null");
        ensure!(p as usize % 256 == 0, "misaligned {p:?}");
        ensure!(std::slice::from_raw_parts(p as *const u8, 1000).iter().all(|&b| b == 0), "not zeroed");
        XMemFree(p, attrs);
        Ok(())
    });

    check("kernelx: XMemVirtualAlloc – graphics memory type in UMA, others from Windows", || unsafe {
        let (commit, rw) = (0x3000u32, 4u32);
        let g = XMemVirtualAlloc(std::ptr::null(), 0x1_0000, commit, 1u64 << 29, rw);
        let p = XMemVirtualAlloc(std::ptr::null(), 0x1_0000, commit, 0, rw);
        ensure!(!g.is_null() && !p.is_null(), "null: {g:?} {p:?}");
        ensure!(g as usize >> 32 >= 4, "graphics memory outside UMA: {g:?}");
        *(g as *mut u32) = 7;
        *(p as *mut u32) = 7;
        ensure!(VirtualFree(g, 0, 0x8000) != 0 && VirtualFree(p, 0, 0x8000) != 0, "VirtualFree");
        Ok(())
    });

    check("xmem: working-set statistics, physical pages mapped and freed", || unsafe {
        let mut ws = [u64::MAX; 15];
        ensure!(XMemGetWorkingSetStatistics(0x1000_0001, ws.as_mut_ptr()) != 0, "XMemGetWorkingSetStatistics");
        // only the limit and the use are written: the structure's size differs between GDK
        // versions (an older one has less room than this buffer)
        ensure!(ws[0] >= 8 << 30 && ws[1] > 0 && ws[1] <= ws[0] && ws[2..].iter().all(|&x| x == u64::MAX), "limit / used {:#x} / {:#x}", ws[0], ws[1]);
        let (mut n, mut pages) = (4usize, [0u64; 4]);
        ensure!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()) != 0 && n == 4, "XMemAllocatePhysicalPages");
        let p = XMemMapPhysicalPages(std::ptr::null_mut(), 4, pages.as_mut_ptr()) as *mut u32;
        ensure!(!p.is_null() && p as usize >> 32 >= 4, "XMemMapPhysicalPages: {p:?}");
        *p.add(0x1_0000 / 4) = 7; // second page
        ensure!(XMemFreePhysicalPages(4, pages.as_mut_ptr()) != 0, "XMemFreePhysicalPages");
        Ok(())
    });

    check("xmem: physical pages mapped into reserved address space keep their contents", || unsafe {
        let (reserve, rw, graphics) = (0x2000u32, 4u32, 1u64 << 29);
        let page = 0x1_0000usize;
        let va = XMemVirtualAlloc(std::ptr::null(), 4 * page, reserve, graphics, rw) as usize;
        ensure!(va != 0 && va.is_multiple_of(page), "reservation {va:#x}");
        let (mut n, mut pages) = (2usize, [0u64; 2]);
        ensure!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()) != 0, "XMemAllocatePhysicalPages");
        // in the reverse order, one page in: page b at va + 1, page a at va + 2
        let mut ba = [pages[1], pages[0]];
        ensure!(XMemMapPhysicalPages((va + page) as *mut c_void, 2, ba.as_mut_ptr()) as usize == va + page, "map b, a");
        *((va + page) as *mut u32) = 0xB;
        *((va + 2 * page) as *mut u32) = 0xA;
        // page a moves to va: its data with it
        let mut a = [pages[0]];
        ensure!(XMemMapPhysicalPages(va as *mut c_void, 1, a.as_mut_ptr()) as usize == va, "map a at va");
        ensure!(*(va as *const u32) == 0xA, "page a moved without its data");
        // unmapped (null pages) and mapped again further on: still its data
        ensure!(XMemMapPhysicalPages(va as *mut c_void, 1, std::ptr::null_mut()) as usize == va, "unmap");
        ensure!(XMemMapPhysicalPages((va + 3 * page) as *mut c_void, 1, a.as_mut_ptr()) as usize == va + 3 * page, "map a again");
        ensure!(*((va + 3 * page) as *const u32) == 0xA && *((va + page) as *const u32) == 0xB, "contents after remapping");
        ensure!(XMemFreePhysicalPages(2, pages.as_mut_ptr()) != 0, "XMemFreePhysicalPages");
        ensure!(VirtualFree(va as *mut c_void, 0, 0x8000) != 0, "VirtualFree");
        Ok(())
    });

    check("kernelx: graphics memory lives in the GPU-visible 4 GiB-aligned region (UMA)", || unsafe {
        let attrs = 1u64 << 29; // memory type 1: graphics
        let a = XMemAlloc(0x20000, attrs);
        let b = XMemAlloc(0x10000, attrs);
        ensure!(!a.is_null() && !b.is_null(), "null");
        // in the region (from 16 GiB up), 64 KiB aligned; the region's first range is XCL's own
        // (32-bit descriptor heaps), so the title's first allocation is not its start
        ensure!(a as usize >> 32 >= 4 && a as usize & 0xFFFF == 0, "a = {a:?}");
        ensure!(b as usize == a as usize + 0x20000, "b = {b:?}, a = {a:?}");
        *(b as *mut u64) = 0x1122_3344;
        ensure!(*(b as *const u64) == 0x1122_3344, "read back");
        XMemFree(b, attrs);
        XMemFree(a, attrs);
        let c = XMemAlloc(0x30000, attrs);
        ensure!(c == a, "freed blocks merge and are reused: {c:?}");
        ensure!(std::slice::from_raw_parts(c as *const u8, 0x30000).iter().all(|&x| x == 0), "reused block not zeroed");
        XMemFree(c, attrs);
        // 4 MiB page-size semantics (XALLOC_PAGESIZE_4MB = 2 << 14): 4 MiB-aligned & zeroed
        let attrs_4m = (1u64 << 29) | (2u64 << 14);
        let big = XMemAlloc(0x1000, attrs_4m);
        ensure!(!big.is_null() && (big as usize) % (4 << 20) == 0, "4MB page alloc not 4 MiB aligned: {big:?}");
        ensure!(*(big as *const u64) == 0 && *((big as usize + (4 << 20) - 8) as *const u64) == 0, "4MB page alloc not zeroed");
        XMemFree(big, attrs_4m);
        Ok(())
    });

    check("user32: a top-level window made either way (CreateWindowExW / A) is the hidden 1920x1080 console screen", || unsafe {
        let wide = |t: &str| t.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let class = wide("XclTestTitleWindow");
        let wc = WndClassW { style: 0, proc_: DefWindowProcW as *const () as usize, cls_extra: 0, wnd_extra: 0, instance: GetModuleHandleW(std::ptr::null()), icon: 0, cursor: 0, background: 0, menu: std::ptr::null(), class: class.as_ptr() };
        ensure!(RegisterClassW(&wc) != 0, "RegisterClassW");
        // WS_OVERLAPPEDWINDOW | WS_VISIBLE, an ordinary PC frame of 1280x720
        let style = 0x00CF_0000 | 0x1000_0000;
        let title = wide("test title");
        let lookup = |dll: &str, name: &[u8]| {
            let h = LoadLibraryW(wide(dll).as_ptr());
            if h == 0 { 0 } else { GetProcAddress(h, name.as_ptr()) }
        };
        let (fw, fa) = (lookup("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", b"CreateWindowExW "), lookup("ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll", b"CreateWindowExA "));
        ensure!(fw != 0 && fa != 0, "CreateWindowExW {fw:#x} / A {fa:#x} from the ntuser API sets");
        let (cw, ca): (CreateWindowExFn<u16>, CreateWindowExFn<u8>) = (std::mem::transmute(fw), std::mem::transmute(fa));
        let w = cw(0x4_0000, class.as_ptr(), title.as_ptr(), style, 100, 100, 1280, 720, std::ptr::null_mut(), std::ptr::null_mut(), wc.instance, std::ptr::null());
        let a = ca(0, b"XclTestTitleWindow ".as_ptr(), b"test title ".as_ptr(), style, 100, 100, 1280, 720, std::ptr::null_mut(), std::ptr::null_mut(), wc.instance, std::ptr::null());
        ensure!(!w.is_null() && !a.is_null(), "CreateWindowExW {w:?} / A {a:?}");
        for (name, h) in [("W", w), ("A", a)] {
            let ex = GetWindowLongW(h, -20) as u32;
            let st = GetWindowLongW(h, -16) as u32;
            let mut r = [0i32; 4];
            GetClientRect(h, &mut r);
            let mut cloaked = 0u32;
            DwmGetWindowAttribute(h, 14, &mut cloaked as *mut u32 as *mut c_void, 4);
            ensure!(ex & 0x80 != 0 && ex & 0x4_0000 == 0, "{name}: not a tool window (no taskbar button) – exstyle {ex:#x}");
            ensure!(st & 0x8000_0000 != 0 && st & 0x00C0_0000 == 0, "{name}: not a frameless popup – style {st:#x}");
            ensure!(r[2] - r[0] == 1920 && r[3] - r[1] == 1080, "{name}: client area {}x{}", r[2] - r[0], r[3] - r[1]);
            ensure!(cloaked != 0, "{name}: not cloaked");
        }
        DestroyWindow(w);
        DestroyWindow(a);
        Ok(())
    });

    check("GameInput: GameInputCreate reaches the system GameInput (timestamps advance)", || unsafe {
        let mut gi: *mut c_void = std::ptr::null_mut();
        let hr = GameInputCreate(&mut gi);
        ensure!(hr >= 0 && !gi.is_null(), "GameInputCreate {hr:#x}");
        let vt = *(gi as *const *const usize);
        // IGameInput: IUnknown (3 slots), then GetCurrentTimestamp
        let ts: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(*vt.add(3));
        let a = ts(gi);
        std::thread::sleep(Duration::from_millis(2));
        let b = ts(gi);
        ensure!(a > 0 && b > a, "timestamps {a} {b}");
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
        release(gi);
        Ok(())
    });

    check("kernelx: TitleMemoryStatus", || unsafe {
        let mut s = TitleMemoryStatus { dw_length: 80, dw_reserved: 0, v: [0; 9] };
        ensure!(TitleMemoryStatus(&mut s) == 1, "failed");
        ensure!(s.v[0] > 1 << 30, "total = {}", s.v[0]);
        Ok(())
    });

    check("kernelx: SetThreadName / GetThreadName", || unsafe {
        let h = std::thread::spawn(|| {
            let ok = SetThreadName(GetCurrentThread(), wide("xcl-test-worker").as_ptr());
            let mut buf = [0u16; 64];
            let mut n = 0usize;
            let ok2 = GetThreadName(GetCurrentThread(), buf.as_mut_ptr(), 64, &mut n);
            (ok, ok2, String::from_utf16_lossy(&buf[..n]))
        });
        let (ok, ok2, name) = h.join().map_err(|_| "panic")?;
        ensure!(ok != 0 && ok2 != 0, "ok={ok} ok2={ok2}");
        ensure!(name == "xcl-test-worker", "name = {name}");
        Ok(())
    });

    check("kernelx: unimplemented import becomes a stub and the title keeps running", || unsafe {
        ensure!(XclDefinitelyMissingFunction(1, 2) == 0, "stub returned non-zero");
        Ok(())
    });

    check("modules: GetModuleHandle(NULL) is the game image", || unsafe {
        let me = &__ImageBase as *const u8 as usize;
        ensure!(GetModuleHandleW(std::ptr::null()) == me, "{:#x} != {me:#x}", GetModuleHandleW(std::ptr::null()));
        Ok(())
    });

    check("modules: LoadLibrary/GetProcAddress on an HLE module", || unsafe {
        let h = LoadLibraryW(wide("kernelx.dll").as_ptr());
        ensure!(h != 0, "LoadLibrary(kernelx) failed");
        let f = GetProcAddress(h, b"XMemAlloc\0".as_ptr());
        ensure!(f != 0, "XMemAlloc not found");
        let alloc: unsafe extern "system" fn(usize, u64) -> *mut c_void = std::mem::transmute(f);
        let p = alloc(64, 0);
        ensure!(!p.is_null(), "alloc via GetProcAddress failed");
        XMemFree(p, 0);
        ensure!(GetProcAddress(h, b"XNoSuchApiAnywhere\0".as_ptr()) == 0, "missing API must return NULL");
        Ok(())
    });

    check("game DLL: static import + DllMain + TLS + its own kernelx import", || unsafe {
        ensure!(mw_attached() == 1, "DllMain(PROCESS_ATTACH) not run");
        ensure!(mw_add(2, 3) == 5, "mw_add");
        ensure!(mw_tls_bump() == 8, "DLL TLS template");
        let t = std::thread::spawn(|| mw_tls_bump()).join().map_err(|_| "panic")?;
        ensure!(t == 8, "DLL TLS in new thread = {t}");
        ensure!(mw_xmem_roundtrip() == 1, "DLL kernelx import");
        let h = LoadLibraryW(wide("testgame_mw.dll").as_ptr());
        ensure!(h != 0, "LoadLibrary(testgame_mw)");
        let f = GetProcAddress(h, b"mw_add\0".as_ptr());
        ensure!(f != 0, "GetProcAddress(mw_add)");
        let add: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(f);
        ensure!(add(40, 2) == 42, "dynamic mw_add");
        Ok(())
    });

    check("files: console paths G:\\ and GAME: read from the game folder", || {
        let a = std::fs::read_to_string("G:\\data\\hello.txt").map_err(|e| format!("G: {e}"))?;
        ensure!(a.starts_with("hello from the game folder"), "content {a:?}");
        let b = std::fs::read("GAME:\\data\\sub\\nested.bin").map_err(|e| format!("GAME: {e}"))?;
        ensure!(b == [0u8, 1, 2, 3, 4, 5, 6, 7], "nested {b:?}");
        let c = std::fs::read_to_string("data\\hello.txt").map_err(|e| format!("relative: {e}"))?;
        ensure!(a == c, "relative differs");
        Ok(())
    });

    check("files: CRT fopen goes through the VFS too", || unsafe {
        let f = fopen(b"G:\\data\\hello.txt\0".as_ptr(), b"rb\0".as_ptr());
        ensure!(!f.is_null(), "fopen failed");
        let mut buf = [0u8; 64];
        let n = fread(buf.as_mut_ptr() as *mut c_void, 1, 64, f);
        fclose(f);
        ensure!(buf[..n].starts_with(b"hello from the game folder"), "read {n} bytes");
        Ok(())
    });

    check("files: writes go to the overlay, game folder untouched", || {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open("G:\\data\\hello.txt").map_err(|e| e.to_string())?;
        f.write_all(b"+modified by title\n").map_err(|e| e.to_string())?;
        drop(f);
        let s = std::fs::read_to_string("G:\\data\\hello.txt").map_err(|e| e.to_string())?;
        ensure!(s.contains("+modified by title"), "overlay not visible: {s:?}");
        std::fs::write("G:\\data\\new_save.bin", b"save-data").map_err(|e| e.to_string())?;
        ensure!(std::fs::read("G:\\data\\new_save.bin").map_err(|e| e.to_string())? == b"save-data", "new file");
        std::fs::create_dir_all("G:\\data\\newdir\\deeper").map_err(|e| e.to_string())?;
        ensure!(std::path::Path::new("G:\\data\\newdir\\deeper").is_dir(), "new dir");
        std::fs::rename("G:\\data\\new_save.bin", "G:\\data\\renamed.bin").map_err(|e| format!("rename: {e}"))?;
        ensure!(!std::path::Path::new("G:\\data\\new_save.bin").exists(), "rename source still visible");
        ensure!(std::fs::read("G:\\data\\renamed.bin").map_err(|e| e.to_string())? == b"save-data", "rename target");
        Ok(())
    });

    check("files: delete hides the file without touching the original", || {
        std::fs::remove_file("G:\\data\\delete_me.txt").map_err(|e| e.to_string())?;
        ensure!(!std::path::Path::new("G:\\data\\delete_me.txt").exists(), "still visible");
        ensure!(std::fs::read("G:\\data\\delete_me.txt").is_err(), "still readable");
        Ok(())
    });

    check("files: directory listing merges overlay and hides deleted files", || {
        let mut names: Vec<String> = std::fs::read_dir("G:\\data")
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for want in ["hello.txt", "renamed.bin", "newdir", "sub"] {
            ensure!(names.iter().any(|n| n == want), "missing {want} in {names:?}");
        }
        ensure!(!names.iter().any(|n| n == "delete_me.txt" || n == "new_save.bin"), "stale entries {names:?}");
        Ok(())
    });

    check("files: T:\\ title scratch storage", || {
        std::fs::write("T:\\scratch.tmp", b"tmp").map_err(|e| e.to_string())?;
        ensure!(std::fs::read("T:\\scratch.tmp").map_err(|e| e.to_string())? == b"tmp", "readback");
        Ok(())
    });

    check("files: T:\\ free space (the host volume behind the mount)", || unsafe {
        let t = wide("T:\\");
        let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
        ensure!(GetDiskFreeSpaceExW(t.as_ptr(), &mut avail, &mut total, &mut free) != 0, "GetDiskFreeSpaceExW(T:\\) failed");
        ensure!(total > 0 && avail > 0 && avail <= total && free <= total, "avail {avail} free {free} total {total}");
        let (mut spc, mut bps, mut nfc, mut tnc) = (0u32, 0u32, 0u32, 0u32);
        ensure!(GetDiskFreeSpaceW(t.as_ptr(), &mut spc, &mut bps, &mut nfc, &mut tnc) != 0, "GetDiskFreeSpaceW(T:\\) failed");
        ensure!(spc > 0 && bps > 0 && tnc > 0, "spc {spc} bps {bps} total clusters {tnc}");
        Ok(())
    });

    check("XGameRuntime: initialize + package identity", || unsafe {
        ensure!(XGameRuntimeInitialize() == 0, "init");
        let mut buf = [0u8; 256];
        ensure!(XPackageGetCurrentProcessPackageIdentifier(256, buf.as_mut_ptr()) == 0, "package id");
        ensure!(cstr(&buf).starts_with("XCL.TestTitle"), "id = {}", cstr(&buf));
        Ok(())
    });

    check("XTaskQueue: manual, delayed and thread-pool dispatch", || unsafe {
        let mut q = std::ptr::null_mut();
        ensure!(XTaskQueueCreate(0, 0, &mut q) == 0, "create manual");
        FLAG.store(0, Ordering::SeqCst);
        XTaskQueueSubmitCallback(q, 0, 1 as _, set_flag);
        ensure!(FLAG.load(Ordering::SeqCst) == 0, "manual ran early");
        ensure!(XTaskQueueDispatch(q, 0, 0), "dispatch");
        ensure!(FLAG.load(Ordering::SeqCst) == 1, "manual did not run");
        XTaskQueueSubmitDelayedCallback(q, 0, 20, 10 as _, set_flag);
        ensure!(XTaskQueueDispatch(q, 0, 3000), "delayed dispatch");
        ensure!(FLAG.load(Ordering::SeqCst) == 11, "delayed");
        XTaskQueueTerminate(q, true, std::ptr::null_mut(), 0);
        XTaskQueueCloseHandle(q);
        let mut p = std::ptr::null_mut();
        ensure!(XTaskQueueCreate(1, 1, &mut p) == 0, "create pool");
        XTaskQueueSubmitCallback(p, 0, 100 as _, set_flag);
        let t0 = Instant::now();
        while FLAG.load(Ordering::SeqCst) != 111 && t0.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(2));
        }
        ensure!(FLAG.load(Ordering::SeqCst) == 111, "pool callback");
        XTaskQueueTerminate(p, true, std::ptr::null_mut(), 0);
        XTaskQueueCloseHandle(p);
        Ok(())
    });

    check("XUser: silent sign-in, device association, and no XSTS token offline", || unsafe {
        let mut b: XAsyncBlock = std::mem::zeroed();
        ensure!(XUserAddAsync(1, &mut b) == 0, "XUserAddAsync");
        ensure!(XAsyncGetStatus(&mut b, true) == 0, "status");
        let mut u = std::ptr::null_mut();
        ensure!(XUserAddResult(&mut b, &mut u) == 0 && !u.is_null(), "result");
        let mut tag = [0u8; 16];
        let mut used = 0usize;
        ensure!(XUserGetGamertag(u, 0, 16, tag.as_mut_ptr(), &mut used) == 0, "gamertag");
        ensure!(cstr(&tag) == "Player1", "gamertag = {}", cstr(&tag));
        let dev = [1u8; 32];
        let mut du = std::ptr::null_mut();
        ensure!(XUserFindForDevice(dev.as_ptr() as *const c_void, &mut du) == 0 && du == u, "XUserFindForDevice");
        XUserCloseHandle(du);

        let mut tb: XAsyncBlock = std::mem::zeroed();
        ensure!(
            XUserGetTokenAndSignatureAsync(
                u,
                0,
                b"GET\0".as_ptr(),
                b"https://example.invalid/hotfix\0".as_ptr(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                &mut tb,
            ) == 0,
            "XUserGetTokenAndSignatureAsync"
        );
        // no Xbox Live (`xbox_live_tokens` off, the default): the request fails as on a console
        // that cannot reach it, and there is no result
        let st = XAsyncGetStatus(&mut tb, true);
        ensure!(st == 0x8007_2EFD_u32 as i32, "token status {st:#x}");
        let mut tsz = 0usize;
        ensure!(XUserGetTokenAndSignatureResultSize(&mut tb, &mut tsz) != 0, "a token result offline ({tsz} bytes)");
        XUserCloseHandle(u);
        Ok(())
    });

    check("XTaskQueue composite + XNetworking + advapi32legacy LSA fallback", || unsafe {
        let mut base = std::ptr::null_mut();
        ensure!(XTaskQueueCreate(0, 0, &mut base) == 0, "create base");
        let (mut wp, mut cp) = (std::ptr::null_mut(), std::ptr::null_mut());
        ensure!(XTaskQueueGetPort(base, 0, &mut wp) == 0 && XTaskQueueGetPort(base, 1, &mut cp) == 0, "get ports");
        let mut comp = std::ptr::null_mut();
        ensure!(XTaskQueueCreateComposite(wp, cp, &mut comp) == 0, "create composite");
        FLAG.store(0, Ordering::SeqCst);
        ensure!(XTaskQueueSubmitCallback(base, 0, 5 as _, set_flag) == 0, "submit base");
        ensure!(XTaskQueueSubmitCallback(comp, 0, 7 as _, set_flag) == 0, "submit comp");
        // Terminating composite must not cancel base's queued callback.
        ensure!(XTaskQueueTerminate(comp, true, std::ptr::null_mut(), 0) == 0, "terminate comp");
        ensure!(FLAG.load(Ordering::SeqCst) == 0, "base item ran/canceled early");
        ensure!(XTaskQueueDispatch(base, 0, 0) && FLAG.load(Ordering::SeqCst) == 5, "base dispatch");
        XTaskQueueCloseHandle(comp);
        XTaskQueueTerminate(base, true, std::ptr::null_mut(), 0);
        XTaskQueueCloseHandle(base);

        let mut udp = 0u16;
        ensure!(XNetworkingQueryPreferredLocalUdpMultiplayerPort(&mut udp) == 0 && udp == 3074, "udp={udp}");
        let mut hint = [0u8; 16];
        ensure!(XNetworkingGetConnectivityHint(hint.as_mut_ptr()) == 0 && hint[12] == 1, "hint={hint:?}");

        let adv = LoadLibraryW(wide("advapi32legacy.dll").as_ptr());
        ensure!(adv != 0, "LoadLibrary(advapi32legacy.dll)");
        ensure!(GetProcAddress(adv, b"LsaOpenPolicy\0".as_ptr()) != 0, "advapi32legacy!LsaOpenPolicy");
        Ok(())
    });

    check("storage: save folder + persistent local storage", || unsafe {
        let mut b: XAsyncBlock = std::mem::zeroed();
        ensure!(XGameSaveFilesGetFolderWithUiAsync(std::ptr::null_mut(), b"test-scid\0".as_ptr(), &mut b) == 0, "async");
        ensure!(XAsyncGetStatus(&mut b, true) == 0, "status");
        let mut buf = [0u8; 512];
        ensure!(XGameSaveFilesGetFolderWithUiResult(&mut b, 512, buf.as_mut_ptr()) == 0, "result");
        let dir = cstr(&buf);
        std::fs::write(format!("{dir}\\slot0.sav"), b"progress").map_err(|e| format!("write save: {e}"))?;
        let mut pls = [0u8; 512];
        let mut used = 0usize;
        ensure!(XPersistentLocalStorageGetPath(512, pls.as_mut_ptr(), &mut used) == 0, "pls");
        ensure!(std::path::Path::new(&cstr(&pls)).is_dir(), "pls dir {}", cstr(&pls));
        Ok(())
    });

    check("XPersistentLocalStorage: a drive root, as on the console", || unsafe {
        let mut pls = [0u8; 512];
        let mut used = 0usize;
        ensure!(XPersistentLocalStorageGetPath(512, pls.as_mut_ptr(), &mut used) == 0, "pls");
        let root = cstr(&pls);
        ensure!(root.len() == 3 && root.ends_with(":\\"), "PLS path {root:?} is not a drive root");
        // Minecraft keeps only the drive name and builds "<drive>:/Users/<xuid>/games/com.mojang"
        let dir = format!("{}:/Users/xcl/games", &root[..1]);
        std::fs::create_dir_all(&dir).map_err(|e| format!("create {dir}: {e}"))?;
        std::fs::write(format!("{dir}/world.dat"), b"blocks").map_err(|e| format!("write {dir}/world.dat: {e}"))?;
        let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
        ensure!(GetDiskFreeSpaceExW(wide(&dir).as_ptr(), &mut avail, &mut total, &mut free) != 0, "GetDiskFreeSpaceExW({dir}) failed");
        ensure!(avail > 0 && avail <= total, "avail {avail} total {total}");
        Ok(())
    });

    check("D3D12.X: queue + fence signal on the GPU", gpu::queue_fence);
    check("D3D12.X: shaders captured; console shader gets a fallback PSO", gpu::pso_capture_and_fallback);
    check("D3D12.X: retail-format console shader (XDAT/XBCH/FSSM) captured + fallback PSO", gpu::xbox_native_pso);
    check("D3D12.X: triangle rendered on the real GPU and read back (console + derived pipeline)", gpu::render_triangle);
    check("D3D12.X: frame clock + PresentX of the title's back buffer", gpu::present_x_frame);
    check("D3D12.X: console device methods (placed title memory, DMA queue + list, page pools)", gpu::console_device_methods);
    check("D3D12.X: console command list memory methods (writes, timestamp, fill, copy, KickoffX, Wait32/64)", gpu::console_list_methods);
    check("D3D12.X: sparse VRAM buffer heaps (placed aliasing, persistent Map coherence, readback)", gpu::sparse_buffer_heap_coherence);
    check("D3D12.X: coherent window headroom (CUSTOM+WRITE_BACK & oversized READBACK in sparse VRAM)", gpu::coherent_window_headroom);
    check("D3D12.X: async ring-buffered sparse uploads (80 MiB across 20 submits + reused-tile zeroing)", gpu::sparse_async_ring_upload_stress);
    check("D3D12.X: descriptor heaps still in window 0 after an 8 GiB+ graphics reservation", gpu::descriptor_heaps_after_a_large_graphics_reservation);
    check("D3D12.X: committed textures beyond the coherent room are created; a readback buffer still gets coherent memory", gpu::committed_textures_leave_room_for_readback);
    check("D3D12.X: dynamic depth bias via RSSetDepthBiasX (#94)", gpu::dynamic_depth_bias);
    check("D3D12.X: raytracing surface (OPTIONS5, AS SRV descriptor, #146/#148 BVH list slots)", gpu::raytracing_surface);
    check("DirectStorage (xcrdapi): a file read into memory, status + D3D12 fence", gpu::direct_storage);
    check("D3D12.X: a texture in title memory (XG layout, #44) follows CPU writes", gpu::title_texture);
    check("D3D12.X: a texture in a streaming pool gets its texels once pages are mapped in", gpu::streamed_texture);
    check("MFPlat: MFResetDXGIDeviceManagerX resets IMFDXGIDeviceManager with the d3d12_xs device", gpu::mfplat_reset_dxgi_device_manager);
    check("XAudio2: console flags/device, XMA2 & Opus voices decoded, PCM passes, callbacks", audio::xaudio2_console_voices);

    let failures = FAILURES.load(Ordering::SeqCst);
    println!("xcl_testtitle: done, failures: {failures}");
    std::process::exit(failures as i32);
}
