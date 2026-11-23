//! Exception-handling support for manually mapped images.
//!
//! * `.pdata` is registered with `RtlAddFunctionTable` so the unwinder finds our frames.
//! * `RtlPcToFileHeader` is hooked: x64 C++ `throw` (`_CxxThrowException`) and the
//!   catch matching code use it to find the image base of the thrower's `ThrowInfo`;
//!   the real function only knows modules on the Windows loader list.
//! * An unhandled-exception filter logs crashes with module+RVA and writes the report.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::System::Diagnostics::Debug::{
    RtlAddFunctionTable, SetUnhandledExceptionFilter, EXCEPTION_POINTERS, IMAGE_RUNTIME_FUNCTION_ENTRY,
};

use crate::hooks;
use crate::modules;

pub fn register_function_table(base: usize, rva: u32, size: u32) -> bool {
    let count = size / std::mem::size_of::<IMAGE_RUNTIME_FUNCTION_ENTRY>() as u32;
    let ok = unsafe {
        RtlAddFunctionTable((base + rva as usize) as *const IMAGE_RUNTIME_FUNCTION_ENTRY, count, base as u64)
    };
    if !ok {
        crate::xlog_warn!("RtlAddFunctionTable({base:#x}) failed");
    }
    ok
}

static ORIG_PC_TO_FILE_HEADER: AtomicUsize = AtomicUsize::new(0);
type PcToFileHeader = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> *mut c_void;

unsafe extern "system" fn hk_rtl_pc_to_file_header(pc: *mut c_void, base_out: *mut *mut c_void) -> *mut c_void {
    if let Some(b) = modules::image_base_for_addr(pc as usize) {
        if !base_out.is_null() {
            *base_out = b as *mut c_void;
        }
        return b as *mut c_void;
    }
    let orig: PcToFileHeader = std::mem::transmute(ORIG_PC_TO_FILE_HEADER.load(Ordering::Acquire));
    orig(pc, base_out)
}

pub fn install_pc_hook() -> Result<(), String> {
    unsafe {
        hooks::hook(
            "ntdll.dll",
            "RtlPcToFileHeader",
            hk_rtl_pc_to_file_header as usize,
            &ORIG_PC_TO_FILE_HEADER,
        )
    }
}

/// The title's unhandled-exception filter (0: none), called after XCL's report.
static TITLE_FILTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The title sets its filter: kept behind XCL's; returns the title's previous one.
pub fn set_title_filter(f: usize) -> usize {
    TITLE_FILTER.swap(f, Ordering::AcqRel)
}

unsafe extern "system" fn unhandled_filter(info: *const EXCEPTION_POINTERS) -> i32 {
    crate::hle::d3d12x::real::log_device_removed();
    if !info.is_null() && !(*info).ExceptionRecord.is_null() {
        let rec = &*(*info).ExceptionRecord;
        let addr = rec.ExceptionAddress as usize;
        let mut extra = String::new();
        if rec.ExceptionCode as u32 == 0xC000_0005 && rec.NumberParameters >= 2 {
            let op = match rec.ExceptionInformation[0] {
                0 => "read",
                1 => "write",
                8 => "execute",
                _ => "?",
            };
            extra = format!(" ({op} at address {:#x})", rec.ExceptionInformation[1]);
        }
        if rec.ExceptionCode as u32 == ILLEGAL_INSTRUCTION {
            extra = format!(" (code bytes: {})", code_at(addr));
        }
        let ctx = (*info).ContextRecord;
        let regs = if ctx.is_null() {
            String::new()
        } else {
            let c = &*ctx;
            // registers that point at objects: the module of their vtable (which object it is)
            let obj = |v: u64| -> String {
                let p = v as usize;
                if p < 0x10000 || !crate::gpu::shader::readable(p, 8) {
                    return String::new();
                }
                let vt = (p as *const usize).read_unaligned();
                let d = modules::describe_address(vt);
                if d.contains('+') { format!(" (vtable {d})") } else { String::new() }
            };
            let regs = [("rax", c.Rax), ("rbx", c.Rbx), ("rcx", c.Rcx), ("rdx", c.Rdx), ("rsi", c.Rsi), ("rdi", c.Rdi), ("rbp", c.Rbp), ("r8", c.R8), ("r9", c.R9), ("r12", c.R12), ("r13", c.R13), ("r14", c.R14), ("r15", c.R15), ("rsp", c.Rsp)];
            let callers = stack_callers(c.Rsp as usize);
            regs.iter().map(|(n, v)| format!(" {n}={v:#x}{}", obj(*v))).collect::<String>() + &if callers.is_empty() { String::new() } else { format!("; on the stack: {callers}") }
        };
        let msg = format!(
            "unhandled exception {:#010x} in {}{extra}{regs}",
            rec.ExceptionCode as u32,
            modules::describe_address(addr)
        );
        crate::xlog_error!("{msg}");
        crate::report::write_final(Some(msg));
        crate::log::flush();
    }
    match TITLE_FILTER.load(Ordering::Acquire) {
        0 => 0, // EXCEPTION_CONTINUE_SEARCH
        f => std::mem::transmute::<usize, unsafe extern "system" fn(*const EXCEPTION_POINTERS) -> i32>(f)(info),
    }
}

static ORIG_NT_TERMINATE: AtomicUsize = AtomicUsize::new(0);
static ENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// ntdll!NtTerminateProcess: the process ending (ExitProcess, abort()'s _exit(3),
/// TerminateProcess of itself) is logged once – status, the calls that led there and, when the
/// status is not 0, every other thread – and the log written out. abort() leaves no other
/// trace: the buffered log lines went with the process.
unsafe extern "system" fn hk_nt_terminate_process(h: isize, status: i32) -> i32 {
    if (h == 0 || h == -1) && !ENDING.swap(true, Ordering::AcqRel) {
        let probe = 0usize;
        let (tid, stack) = (windows_sys::Win32::System::Threading::GetCurrentThreadId(), stack_callers_n(&probe as *const usize as usize, 16));
        if status != 0 {
            crate::xlog_error!("process ends with status {status:#x} (thread {tid}); on the stack: {stack}");
            dump_threads();
        } else {
            crate::xlog_info!("process ends (thread {tid}); on the stack: {stack}");
        }
        crate::log::flush();
    }
    let orig: unsafe extern "system" fn(isize, i32) -> i32 = std::mem::transmute(ORIG_NT_TERMINATE.load(Ordering::Acquire));
    orig(h, status)
}

pub fn install_crash_handler() {
    unsafe {
        if let Err(e) = hooks::hook("ntdll.dll", "NtTerminateProcess", hk_nt_terminate_process as usize, &ORIG_NT_TERMINATE) {
            crate::xlog_warn!("{e}");
        }
        SetUnhandledExceptionFilter(Some(unhandled_filter));
        install_invalid_parameter_handler();
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(0, Some(cpp_exception_logger));
    }
}

/// MSVC's x64 `__security_check_cookie` (vcruntime): `cmp rcx, [__security_cookie]; jne fail;
/// rol rcx, 16; test cx, 0xFFFF; jne restore; ret; restore: ror rcx, 16; fail: jmp
/// __report_gsfailure` – fixed but for the cookie's address (`None`) and the jump. The
/// console's vcruntime has `bnd` prefixes (F2) on the branches and the return.
const GS_CHECKS: [&[Option<u8>]; 2] = [
    &[Some(0x48), Some(0x3B), Some(0x0D), None, None, None, None, Some(0x75), Some(0x10), Some(0x48), Some(0xC1), Some(0xC1), Some(0x10), Some(0x66), Some(0xF7), Some(0xC1), Some(0xFF), Some(0xFF), Some(0x75), Some(0x01), Some(0xC3), Some(0x48), Some(0xC1), Some(0xC9), Some(0x10), Some(0xE9)],
    &[
        Some(0x48), Some(0x3B), Some(0x0D), None, None, None, None, Some(0xF2), Some(0x75), Some(0x12), Some(0x48), Some(0xC1), Some(0xC1), Some(0x10), Some(0x66), Some(0xF7), Some(0xC1), Some(0xFF), Some(0xFF), Some(0xF2), Some(0x75), Some(0x02),
        Some(0xF2), Some(0xC3), Some(0x48), Some(0xC1), Some(0xC9), Some(0x10), Some(0xE9),
    ],
];

/// The offset of `__security_check_cookie`'s jump to `__report_gsfailure` (E9 rel32) in
/// `code` (None: not there).
pub fn find_gs_check(code: &[u8]) -> Option<usize> {
    for p in GS_CHECKS {
        if code.len() < p.len() + 4 {
            continue;
        }
        if let Some(i) = (0..code.len() - p.len() - 4).find(|&i| code[i] == 0x48 && p.iter().enumerate().all(|(k, b)| b.map_or(true, |b| code[i + k] == b))) {
            return Some(i + p.len() - 1);
        }
    }
    None
}

/// Executable memory `[at, at + len)` of a title image: its `__report_gsfailure` (the target of
/// `__security_check_cookie`'s failure jump) is redirected to a report that names the
/// function whose stack cookie did not match (the return address on the stack), then ends
/// the process with the same code (0xC0000409) – a stack-cookie failure is a `__fastfail`,
/// which no exception filter sees, and often means a function of XCL's wrote more into a
/// title's buffer than the title expects.
pub unsafe fn hook_gs_failure(at: usize, len: usize, image: &str) {
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE};
    let code = std::slice::from_raw_parts(at as *const u8, len);
    let Some(i) = find_gs_check(code) else { return };
    let jmp = at + i;
    let rel = (jmp as *const u8).add(1).cast::<i32>().read_unaligned();
    let target = (jmp as isize + 5 + rel as isize) as usize;
    // thunk: rdx = [rsp] (the checked function's return address), then the report
    let t = crate::stubs::alloc_code(32);
    let mut c: Vec<u8> = vec![0x48, 0x8B, 0x14, 0x24, 0x48, 0xB8];
    c.extend_from_slice(&(gs_failure as usize).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xE0]);
    std::ptr::copy_nonoverlapping(c.as_ptr(), t, c.len());
    // __report_gsfailure: mov rax, thunk; jmp rax
    let mut j: Vec<u8> = vec![0x48, 0xB8];
    j.extend_from_slice(&(t as usize).to_le_bytes());
    j.extend_from_slice(&[0xFF, 0xE0]);
    let mut old = 0u32;
    if VirtualProtect(target as *const std::ffi::c_void, j.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
        return;
    }
    std::ptr::copy_nonoverlapping(j.as_ptr(), target as *mut u8, j.len());
    VirtualProtect(target as *const std::ffi::c_void, j.len(), old, &mut old);
    crate::xlog_info!("{image}: stack-cookie failures are reported (__report_gsfailure at {target:#x})");
}

unsafe extern "system" fn gs_failure(cookie: usize, ret: usize) {
    let here = 0usize;
    crate::xlog_error!(
        "stack cookie check failed (cookie {cookie:#x}) in the function returning to {} – its stack frame was overwritten; on the stack: {}",
        modules::describe_address(ret),
        stack_callers(&here as *const usize as usize)
    );
    crate::log::flush();
    windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), 0xC000_0409);
}

/// First-chance C++ exceptions (0xE06D7363), the first 16: type, what() and where they were
/// thrown. One nobody catches ends the process through std::terminate → abort → __fastfail
/// (0xC0000409), which leaves no other trace (a static initializer in a title DLL's DllMain).
/// The C string at `p`, at most `max` bytes, read only as far as the memory is readable (a
/// vectored handler faulting on an unmapped byte after a title's string would turn the
/// exception it looks at into a fatal one).
unsafe fn bounded_cstr(p: usize, max: usize) -> String {
    let mut bytes = Vec::new();
    let mut a = p;
    while bytes.len() < max {
        // readability is checked page by page: at the start and at every page crossed
        if (a == p || a & 0xFFF == 0) && !crate::gpu::shader::readable(a, 1) {
            break;
        }
        let b = *(a as *const u8);
        if b == 0 {
            break;
        }
        bytes.push(b);
        a += 1;
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

const ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;

/// The first 16 code bytes at `rip`, as hex (as far as they are readable).
unsafe fn code_at(rip: usize) -> String {
    let n = (1..=16).rev().find(|&n| crate::gpu::shader::readable(rip, n)).unwrap_or(0);
    crate::insn::hex(std::slice::from_raw_parts(rip as *const u8, n))
}

/// A title's STATUS_ILLEGAL_INSTRUCTION: an instruction the console's Zen 2 has and this CPU
/// lacks (`insn`) is emulated and stepped over (EXCEPTION_CONTINUE_EXECUTION); anything else is
/// logged with its code bytes and goes on to the title's handlers.
unsafe fn emulate_console_only(ctx: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT) -> i32 {
    use crate::insn::ConsoleOnly;
    let rip = ctx.Rip as usize;
    if modules::image_base_for_addr(rip).is_none() || !crate::gpu::shader::readable(rip, 1) {
        return 0;
    }
    let n = (1..=16).rev().find(|&n| crate::gpu::shader::readable(rip, n)).unwrap_or(0);
    let code = std::slice::from_raw_parts(rip as *const u8, n);
    let Some((what, len)) = crate::insn::decode(code) else {
        crate::xlog_error!("illegal instruction in {}: code bytes {}", modules::describe_address(rip), crate::insn::hex(code));
        return 0;
    };
    static SEEN: std::sync::Mutex<Vec<ConsoleOnly>> = std::sync::Mutex::new(Vec::new());
    if let Ok(mut seen) = SEEN.lock() {
        if !seen.contains(&what) {
            seen.push(what);
            if what == ConsoleOnly::Ud2 {
                crate::xlog_error!("ud2 in {} – the title's own trap; on the stack: {}", modules::describe_address(rip), stack_callers(ctx.Rsp as usize));
            } else {
                crate::xlog_warn!("{what:?} in {} – a console CPU instruction this CPU lacks: emulated (logged once)", modules::describe_address(rip));
            }
        }
    }
    match what {
        ConsoleOnly::Ud2 => return 0,
        ConsoleOnly::Clwb | ConsoleOnly::Monitorx | ConsoleOnly::Mwaitx => {}
        ConsoleOnly::Clzero => {
            let line = (ctx.Rax as usize) & !63;
            if crate::gpu::shader::readable(line, 64) {
                std::ptr::write_bytes(line as *mut u8, 0, 64);
            } else {
                return 0; // the real instruction faults there too
            }
        }
        ConsoleOnly::Rdpru => {
            let mut v = 0i64;
            windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v);
            let v = if ctx.Rcx as u32 <= 1 { v as u64 } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
        }
        ConsoleOnly::Rdpid(r) => {
            let cpu = windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() as u64;
            let regs: [&mut u64; 16] = [
                &mut ctx.Rax, &mut ctx.Rcx, &mut ctx.Rdx, &mut ctx.Rbx, &mut ctx.Rsp, &mut ctx.Rbp, &mut ctx.Rsi, &mut ctx.Rdi,
                &mut ctx.R8, &mut ctx.R9, &mut ctx.R10, &mut ctx.R11, &mut ctx.R12, &mut ctx.R13, &mut ctx.R14, &mut ctx.R15,
            ];
            *regs[r as usize & 15] = cpu;
        }
    }
    ctx.Rip += len as u64;
    -1 // EXCEPTION_CONTINUE_EXECUTION
}

unsafe extern "system" fn cpp_exception_logger(info: *mut EXCEPTION_POINTERS) -> i32 {
    const CPP: i32 = 0xE06D_7363_u32 as i32;
    const THREAD_NAME: i32 = 0x406D_1388;
    if info.is_null() || (*info).ExceptionRecord.is_null() || (*info).ContextRecord.is_null() {
        return 0;
    }
    let rec = &*(*info).ExceptionRecord;
    // the debugger's thread-name exception (THREADNAME_INFO: 0x1000, name, thread id, flags):
    // the name goes to SetThreadDescription, where tools – and XCL's thread dump – see it
    if rec.ExceptionCode == THREAD_NAME && rec.NumberParameters >= 3 && rec.ExceptionInformation[0] == 0x1000 {
        let (name, tid) = (rec.ExceptionInformation[1], rec.ExceptionInformation[2] as u32);
        if name != 0 {
            let n: Vec<u16> = bounded_cstr(name, 256).encode_utf16().chain([0]).collect();
            if n.len() > 1 {
                name_thread(tid, &n);
            }
        }
        return 0;
    }
    if rec.ExceptionCode as u32 == ILLEGAL_INSTRUCTION {
        return emulate_console_only(&mut *(*info).ContextRecord);
    }
    if rec.ExceptionCode != CPP || rec.NumberParameters < 4 {
        return 0;
    }
    // thrown by the title's code (its image holds the throw info), not by XCL's libraries
    if !crate::modules::mapped_modules().iter().any(|m| m.base == rec.ExceptionInformation[3]) {
        return 0;
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 16 {
        return 0;
    }
    let (object, throw_info, base) = (rec.ExceptionInformation[1], rec.ExceptionInformation[2], rec.ExceptionInformation[3]);
    let rd32 = |a: usize| if crate::gpu::shader::readable(a, 4) { (a as *const i32).read_unaligned() } else { 0 };
    // ThrowInfo { attributes, unwind, forward compat, catchable type array } (image-relative)
    let mut name = String::from("?");
    if throw_info != 0 && base != 0 {
        let cta = rd32(throw_info + 12);
        if cta > 0 {
            let first = rd32(base + cta as usize + 4);
            if first > 0 {
                let td = rd32(base + first as usize + 4);
                if td > 0 {
                    let p = base + td as usize + 16;
                    if crate::gpu::shader::readable(p, 1) {
                        name = bounded_cstr(p, 256);
                    }
                }
            }
        }
    }
    // std::exception { vfptr, const char* what, … } – for other thrown types the second word is
    // whatever the object holds: read as a string only where it points at readable memory
    let mut what = String::new();
    if object != 0 && crate::gpu::shader::readable(object + 8, 8) {
        let s = (object as *const usize).add(1).read_unaligned();
        if s != 0 {
            what = bounded_cstr(s, 200);
        }
    }
    let ctx = &*(*info).ContextRecord;
    crate::xlog_warn!("C++ exception {name} \"{what}\" thrown; on the stack: {}", stack_callers(ctx.Rsp as usize));
    0
}

/// The UCRT's invalid-parameter handler for the whole process (titles and their DLLs share the
/// host's ucrtbase.dll): without one a CRT function given a bad argument ends the process with
/// `__fastfail` (0xC0000409), which no exception filter sees – the log then just stops. With
/// it the call is logged with the code that made it; functions that can return an error then
/// return EINVAL, the `_noreturn` ones still end the process.
unsafe fn install_invalid_parameter_handler() {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    let h = GetModuleHandleA(b"ucrtbase.dll\0".as_ptr());
    if h.is_null() {
        return;
    }
    let Some(set) = GetProcAddress(h, b"_set_invalid_parameter_handler\0".as_ptr()) else { return };
    type Handler = unsafe extern "C" fn(*const u16, *const u16, *const u16, u32, usize);
    let set: unsafe extern "C" fn(Option<Handler>) -> Option<Handler> = std::mem::transmute(set);
    set(Some(invalid_parameter));
}

unsafe extern "C" fn invalid_parameter(_expr: *const u16, _function: *const u16, _file: *const u16, _line: u32, _reserved: usize) {
    static N: AtomicUsize = AtomicUsize::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 16 {
        let here = 0usize;
        let callers = stack_callers(&here as *const usize as usize);
        crate::xlog_warn!("CRT: invalid parameter (the CRT would end the process with 0xC0000409); called from: {callers}");
    }
}

/// Return addresses into loaded modules (not the system's D3D12 / ntdll / kernel ones) among
/// the first 512 stack slots from `rsp`: the calls that led to a crash (heuristic: values on
/// the stack that point into code).
unsafe fn stack_callers(rsp: usize) -> String {
    stack_callers_n(rsp, 6)
}

/// `stack_callers` with up to `max` callers.
unsafe fn stack_callers_n(rsp: usize, max: usize) -> String {
    let mut callers = Vec::new();
    for k in 0..512usize {
        let at = rsp + 8 * k;
        if !crate::gpu::shader::readable(at, 8) {
            break;
        }
        let d = modules::describe_address((at as *const usize).read_unaligned());
        if d.contains('+') && !d.starts_with("D3D12") && !d.starts_with("ntdll") && !d.starts_with("KERNEL") && !d.ends_with("+0x0") {
            callers.push(d);
            if callers.len() == max {
                break;
            }
        }
    }
    callers.join(", ")
}

#[link(name = "kernel32")]
extern "system" {
    fn GetThreadDescription(thread: *mut std::ffi::c_void, description: *mut *mut u16) -> i32;
    fn LocalFree(mem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
}

/// A thread's description (SetThreadDescription – engines name their threads).
unsafe fn thread_name(h: *mut std::ffi::c_void) -> String {
    let mut p: *mut u16 = std::ptr::null_mut();
    if GetThreadDescription(h, &mut p) < 0 || p.is_null() {
        return String::new();
    }
    let n = (0..256).take_while(|&i| *p.add(i) != 0).count();
    let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
    LocalFree(p as *mut _);
    s
}

#[link(name = "kernel32")]
extern "system" {
    fn SetThreadDescription(thread: *mut std::ffi::c_void, description: *const u16) -> i32;
}

/// Thread `tid` (u32::MAX: the current one) is called `name` (UTF-16, terminated).
unsafe fn name_thread(tid: u32, name: &[u16]) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, OpenThread, THREAD_SET_LIMITED_INFORMATION};
    if tid == u32::MAX || tid == windows_sys::Win32::System::Threading::GetCurrentThreadId() {
        SetThreadDescription(GetCurrentThread(), name.as_ptr());
        return;
    }
    let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, tid);
    if !h.is_null() {
        SetThreadDescription(h, name.as_ptr());
        CloseHandle(h);
    }
}

/// The return addresses of the frames from `ctx` on (its RIP first), unwound with the
/// functions' unwind data (registered for the title's images too); up to `n`.
unsafe fn unwind(ctx: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT, n: usize) -> Vec<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER};
    let mut out = Vec::new();
    for _ in 0..n {
        let rip = ctx.Rip;
        if rip == 0 {
            break;
        }
        out.push(rip as usize);
        let mut base = 0u64;
        let f = RtlLookupFunctionEntry(rip, &mut base, std::ptr::null_mut());
        if f.is_null() {
            // a leaf function: the return address is at RSP
            if !crate::gpu::shader::readable(ctx.Rsp as usize, 8) {
                break;
            }
            ctx.Rip = (ctx.Rsp as *const u64).read_unaligned();
            ctx.Rsp += 8;
        } else {
            let mut data = std::ptr::null_mut();
            let mut frame = 0u64;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, rip, f, ctx, &mut data, &mut frame, std::ptr::null_mut());
        }
    }
    out
}

/// `XCL_THREAD_DUMP_AT`: every other thread of the process stopped for a moment – where it is
/// (RIP) and the calls on its stack that lead there, in the log.
pub fn dump_threads() {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME};
    unsafe {
        let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let mut lines = Vec::new();
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut te) != 0;
        while ok {
            if te.th32OwnerProcessID == pid && te.th32ThreadID != me {
                let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, te.th32ThreadID);
                if !h.is_null() {
                    // CONTEXT must be 16-byte aligned
                    #[repr(align(16))]
                    struct Aligned(CONTEXT);
                    let mut c: Aligned = std::mem::zeroed();
                    c.0.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
                    // only the context is taken while the thread is suspended: the unwinding
                    // (RtlLookupFunctionEntry), the module lookup (loader lock), the name and the
                    // formatting (heap) take locks the suspended thread may hold (#106)
                    let got = SuspendThread(h) != u32::MAX && {
                        let ok = GetThreadContext(h, &mut c.0) != 0;
                        ResumeThread(h);
                        ok
                    };
                    if got {
                        let frames: Vec<String> = unwind(&mut c.0, 14).into_iter().map(modules::describe_address).collect();
                        lines.push(format!("thread {} '{}': {}", te.th32ThreadID, thread_name(h), frames.join(" < ")));
                    }
                    CloseHandle(h);
                }
            }
            ok = Thread32Next(snap, &mut te) != 0;
        }
        CloseHandle(snap);
        crate::xlog_info!("thread dump ({} threads):
{}", lines.len(), lines.join("
"));
        crate::log::flush();
    }
}

// ---------------------------------------------------------------------------------------
// write watchpoint (diagnostics): who overwrites a given 8 bytes
// ---------------------------------------------------------------------------------------

static WATCHED: AtomicUsize = AtomicUsize::new(0);
static WATCH_HITS: AtomicUsize = AtomicUsize::new(0);

/// the exception `watch_writes` raises: the handler arms the raising thread at once
const WATCH_ARM: u32 = 0xE0C1_A7E1;

unsafe extern "system" fn watch_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    const SINGLE_STEP: u32 = 0x8000_0004;
    let rec = &*(*info).ExceptionRecord;
    let ctx = &mut *(*info).ContextRecord;
    let addr = WATCHED.load(Ordering::Acquire);
    if rec.ExceptionCode as u32 == WATCH_ARM {
        ctx.Dr0 = addr as u64;
        // L0, RW0 = 01 (writes), LEN0 = 10 (8 bytes)
        ctx.Dr7 = (ctx.Dr7 & !0xF_0003) | 1 | (0b01 << 16) | (0b10 << 18);
        ctx.ContextFlags |= windows_sys::Win32::System::Diagnostics::Debug::CONTEXT_DEBUG_REGISTERS_AMD64;
        return -1;
    }
    if rec.ExceptionCode as u32 != SINGLE_STEP || addr == 0 || ctx.Dr6 & 1 == 0 {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    ctx.Dr6 = 0;
    if WATCH_HITS.fetch_add(1, Ordering::Relaxed) < 32 {
        let callers = stack_callers(ctx.Rsp as usize);
        crate::xlog_warn!(
            "watch: {:#x} now {:#x}, written by the instruction before {} (thread {}); on the stack: {}",
            addr,
            (addr as *const u64).read_volatile(),
            modules::describe_address(ctx.Rip as usize),
            windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            callers
        );
    }
    -1 // EXCEPTION_CONTINUE_EXECUTION
}

/// Arm a hardware write watchpoint on the 8 bytes at `addr` in every thread of the process
/// (threads created later are armed as they appear, for a minute); each write is logged
/// with the writing code's address. `XCL_WATCH_DEVICE=1`: on the title's D3D12 device.
pub fn watch_writes(addr: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, GetThreadContext, SetThreadContext, CONTEXT, CONTEXT_DEBUG_REGISTERS_AMD64};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};
    if WATCHED.swap(addr, Ordering::AcqRel) != 0 {
        return;
    }
    unsafe {
        AddVectoredExceptionHandler(1, Some(watch_handler));
        // the calling thread now (the others below, as the arming loop reaches them)
        windows_sys::Win32::System::Diagnostics::Debug::RaiseException(WATCH_ARM, 0, 0, std::ptr::null());
    }
    crate::xlog_warn!("watch: write watchpoint on {addr:#x}");
    std::thread::spawn(move || {
        let me = unsafe { GetCurrentThreadId() };
        let mut armed = std::collections::HashSet::new();
        let start = std::time::Instant::now();
        while start.elapsed().as_secs() < 60 {
            unsafe {
                let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if snap == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let mut e: THREADENTRY32 = std::mem::zeroed();
                e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
                let mut ok = Thread32First(snap, &mut e) != 0;
                while ok {
                    if e.th32OwnerProcessID == GetCurrentProcessId() && e.th32ThreadID != me && armed.insert(e.th32ThreadID) {
                        let h = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, 0, e.th32ThreadID);
                        if !h.is_null() {
                            // CONTEXT must be 16-byte aligned
                            #[repr(align(16))]
                            struct Aligned(CONTEXT);
                            let mut c: Aligned = std::mem::zeroed();
                            c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                            if SuspendThread(h) != u32::MAX {
                                if GetThreadContext(h, &mut c.0) != 0 {
                                    c.0.Dr0 = addr as u64;
                                    // L0, RW0 = 01 (writes), LEN0 = 10 (8 bytes)
                                    c.0.Dr7 = (c.0.Dr7 & !0xF_0003) | 1 | (0b01 << 16) | (0b10 << 18);
                                    SetThreadContext(h, &c.0);
                                }
                                ResumeThread(h);
                            } else {
                                armed.remove(&e.th32ThreadID);
                            }
                            windows_sys::Win32::Foundation::CloseHandle(h);
                        }
                    }
                    ok = Thread32Next(snap, &mut e) != 0;
                }
                windows_sys::Win32::Foundation::CloseHandle(snap);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    });
}

// ---------------------------------------------------------------------------------------
// call probes (diagnostics): `XCL_PROBE=rva,…` – what the title's code at those addresses
// is called with
// ---------------------------------------------------------------------------------------

/// probed addresses (DR1..DR3; DR0 stays the write watchpoint's)
static PROBES: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static PROBE_HITS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
/// the last logged values of each probe (a hit that repeats them is only counted)
static PROBE_LAST: Mutex<[u64; 3]> = Mutex::new([0; 3]);
static PROBE_LINES: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

unsafe fn probe_bytes(at: u64, n: usize) -> Option<Vec<u8>> {
    (at > 0x10000 && crate::gpu::shader::readable(at as usize, n)).then(|| std::slice::from_raw_parts(at as *const u8, n).to_vec())
}

/// the exception `arm_current_thread` raises: the handler arms the raising thread
const PROBE_ARM: u32 = 0xE0C1_A7E0;

/// The probes in the calling thread at once (XCL's own short-lived threads that run title
/// callbacks end before the arming loop sees them).
pub fn arm_current_thread() {
    if PROBES[0].load(Ordering::Acquire) != 0 {
        unsafe { windows_sys::Win32::System::Diagnostics::Debug::RaiseException(PROBE_ARM, 0, 0, std::ptr::null()) };
    }
}

unsafe fn arm_context(c: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT) {
    for k in 0..3 {
        let a = PROBES[k].load(Ordering::Acquire) as u64;
        if a == 0 {
            continue;
        }
        match k {
            0 => c.Dr1 = a,
            1 => c.Dr2 = a,
            _ => c.Dr3 = a,
        }
        // L(k+1) enabled; RW = 00 (execution), LEN = 00
        let i = k as u32 + 1;
        c.Dr7 = (c.Dr7 & !(0xF_u64 << (16 + 4 * i)) & !(3_u64 << (2 * i))) | (1 << (2 * i));
    }
}

unsafe extern "system" fn probe_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    const SINGLE_STEP: u32 = 0x8000_0004;
    let rec = &*(*info).ExceptionRecord;
    let ctx = &mut *(*info).ContextRecord;
    if rec.ExceptionCode as u32 == PROBE_ARM {
        arm_context(ctx);
        ctx.ContextFlags |= windows_sys::Win32::System::Diagnostics::Debug::CONTEXT_DEBUG_REGISTERS_AMD64;
        return -1;
    }
    if rec.ExceptionCode as u32 != SINGLE_STEP {
        return 0;
    }
    let Some(k) = (0..3).find(|&k| ctx.Dr6 & (2 << k) != 0 && PROBES[k].load(Ordering::Relaxed) == ctx.Rip as usize) else { return 0 };
    ctx.Dr6 = 0;
    // resume flag: the instruction runs without breaking again
    ctx.EFlags |= 0x1_0000;
    let n = PROBE_HITS[k].fetch_add(1, Ordering::Relaxed) + 1;
    // at a function's entry: arguments 5 and 6 above the return address and the home area
    let stack = |i: u64| probe_bytes(ctx.Rsp + 8 * i, 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    let (s5, s6) = (stack(5), stack(6));
    let at_r8 = probe_bytes(ctx.R8, 24);
    let at_rcx = probe_bytes(ctx.Rcx, 48);
    // values that tell calls apart (pointers left out: they differ per call)
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&(ctx.Rax as u32, ctx.Rdx as u32 & 0xFF, ctx.R9 as u32, s5 as u8, s6 as u8, &at_r8, &at_rcx), &mut h);
    let key = std::hash::Hasher::finish(&h);
    let new = {
        let mut g = PROBE_LAST.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::replace(&mut g[k], key) != key
    };
    if (new || n.is_power_of_two()) && PROBE_LINES[k].fetch_add(1, Ordering::Relaxed) < 200 {
        crate::xlog_info!(
            "probe {}: hit #{n} (thread {}): rcx {:#x} rdx {:#x} r8 {:#x} r9 {:#x} rax {:#x}, stack args 5/6 {s5:#x} {s6:#x}; at r8 {:x?}; at rcx {:x?}; from {}",
            modules::describe_address(ctx.Rip as usize),
            windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            ctx.Rcx,
            ctx.Rdx,
            ctx.R8,
            ctx.R9,
            ctx.Rax,
            at_r8,
            at_rcx,
            modules::describe_address(stack(0) as usize)
        );
    }
    -1
}

/// `XCL_PROBE=rva,…` (hex, up to three, in the title's exe): hardware breakpoints on those
/// instructions in every thread; each hit with new argument values is logged (registers,
/// stack arguments 5 and 6, the bytes at rcx and r8) and the title goes on – which of the
/// title's code runs, with what, without changing it.
pub fn start_probes() {
    // `XCL_WATCH=rva` (hex, in the title's exe): every write to those 8 bytes is logged with
    // the writing code (`watch_writes`) – who fills a field the title later trips over
    if let Some(rva) = std::env::var("XCL_WATCH").ok().and_then(|v| usize::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) {
        let _ = std::thread::Builder::new().name("xcl-watch".into()).spawn(move || loop {
            if let Some(m) = modules::main_module() {
                watch_writes(m.base + rva);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        });
    }
    let rvas: Vec<usize> = std::env::var("XCL_PROBE").unwrap_or_default().split(',').filter_map(|v| usize::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).take(3).collect();
    if rvas.is_empty() {
        return;
    }
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, GetThreadContext, SetThreadContext, CONTEXT, CONTEXT_DEBUG_REGISTERS_AMD64};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};
    unsafe {
        AddVectoredExceptionHandler(1, Some(probe_handler));
    }
    let _ = std::thread::Builder::new().name("xcl-probe".into()).spawn(move || {
        let base = loop {
            if let Some(m) = modules::main_module() {
                break m.base;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        for (k, rva) in rvas.iter().enumerate() {
            PROBES[k].store(base + rva, Ordering::Release);
            crate::xlog_info!("probe: {}", modules::describe_address(base + rva));
        }
        let me = unsafe { GetCurrentThreadId() };
        let mut armed = std::collections::HashSet::new();
        loop {
            unsafe {
                let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if snap == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let mut e: THREADENTRY32 = std::mem::zeroed();
                e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
                let mut ok = Thread32First(snap, &mut e) != 0;
                while ok {
                    if e.th32OwnerProcessID == GetCurrentProcessId() && e.th32ThreadID != me && armed.insert(e.th32ThreadID) {
                        let h = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, 0, e.th32ThreadID);
                        if !h.is_null() {
                            #[repr(align(16))]
                            struct Aligned(CONTEXT);
                            let mut c: Aligned = std::mem::zeroed();
                            c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                            if SuspendThread(h) != u32::MAX {
                                if GetThreadContext(h, &mut c.0) != 0 {
                                    arm_context(&mut c.0);
                                    SetThreadContext(h, &c.0);
                                }
                                ResumeThread(h);
                            } else {
                                armed.remove(&e.th32ThreadID);
                            }
                            windows_sys::Win32::Foundation::CloseHandle(h);
                        }
                    }
                    ok = Thread32Next(snap, &mut e) != 0;
                }
                windows_sys::Win32::Foundation::CloseHandle(snap);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cstr_stops_at_nul_limit_and_unreadable_memory() {
        unsafe {
            let s = b"hello\0tail";
            assert_eq!(bounded_cstr(s.as_ptr() as usize, 64), "hello");
            assert_eq!(bounded_cstr(s.as_ptr() as usize, 3), "hel");
            assert_eq!(bounded_cstr(0, 64), "");
            // a string running to the end of a page followed by an unmapped one
            use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
            let p = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualAlloc(p as *const _, 0x1000, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            std::ptr::write_bytes(p as *mut u8, b'x', 0x1000);
            assert_eq!(bounded_cstr(p + 0x1000 - 4, 64), "xxxx");
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn finds_both_cookie_checks() {
        let pc = [0x48, 0x3B, 0x0D, 1, 2, 3, 4, 0x75, 0x10, 0x48, 0xC1, 0xC1, 0x10, 0x66, 0xF7, 0xC1, 0xFF, 0xFF, 0x75, 0x01, 0xC3, 0x48, 0xC1, 0xC9, 0x10, 0xE9, 0, 0, 0, 0, 0xCC];
        assert_eq!(find_gs_check(&pc), Some(25));
        let console: Vec<u8> = [0xCC, 0x48, 0x3B, 0x0D, 0x99, 0x7E, 0x54, 0x00, 0xF2, 0x75, 0x12, 0x48, 0xC1, 0xC1, 0x10, 0x66, 0xF7, 0xC1, 0xFF, 0xFF, 0xF2, 0x75, 0x02, 0xF2, 0xC3, 0x48, 0xC1, 0xC9, 0x10, 0xE9, 0x4F, 0x06, 0, 0, 0xCC].to_vec();
        assert_eq!(find_gs_check(&console), Some(29));
        assert_eq!(find_gs_check(&[0x48; 64]), None);
    }
}
