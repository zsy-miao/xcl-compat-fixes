//! Slot guards for extended vtables.
//!
//! Real D3D12 objects keep internal virtual methods right after their public
//! interface (probe on Windows 11: device 125 entries vs 104 public, queue 42 vs 19).
//! The runtime calls those internally, so they must stay intact – but a console title
//! calling the same slot index means an Xbox-only method (e.g. PresentX).
//!
//! A guard thunk looks at its return address: called from the game's image → log it
//! as an Xbox method and return 0; called from anywhere else (the runtime itself) →
//! jump to the original function with all argument registers untouched.
//!
//! ```text
//! push rcx/rdx/r8/r9 ; sub rsp,0x68 ; save xmm0-3
//! rcx = [rsp+0x88] (return address), rdx = StubInfo*, call classify
//! restore xmm0-3 ; add rsp,0x68 ; pop r9/r8/rdx/rcx
//! test rax,rax ; jnz stub ; mov rax, original ; jmp rax
//! stub: xor eax,eax ; ret
//! ```

use crate::stubs::{self, StubInfo};

unsafe extern "system" fn classify(caller: usize, info: *const StubInfo) -> usize {
    if crate::modules::image_base_for_addr(caller).is_some() {
        stubs::record_call(&*info, caller);
        1
    } else {
        0
    }
}

pub fn make_guard(original: usize, info: &'static StubInfo) -> usize {
    thunk(Tail::GuardOr(original), info as *const StubInfo as u64, classify as usize)
}

unsafe extern "system" fn pick(caller: usize, targets: *const [usize; 2]) -> usize {
    if crate::modules::image_base_for_addr(caller).is_some() {
        (*targets)[0]
    } else {
        (*targets)[1]
    }
}

/// A slot whose console method differs from the PC one: calls from a game image go to
/// `console`, everything else (the runtime) to `pc`.
pub fn make_split(console: usize, pc: usize) -> usize {
    thunk(Tail::JumpToResult, Box::leak(Box::new([console, pc])) as *const [usize; 2] as u64, pick as usize)
}

/// The targets of a slot whose console method depends on the title's runtime layout.
pub struct LayoutTargets {
    /// a title on the older layout
    pub older: usize,
    /// a title on the newer layout
    pub newer: usize,
    /// everything else (the runtime itself)
    pub pc: usize,
    pub newer_layout: &'static std::sync::atomic::AtomicBool,
}

unsafe extern "system" fn pick_layout(caller: usize, t: *const LayoutTargets) -> usize {
    let t = &*t;
    if crate::modules::image_base_for_addr(caller).is_none() {
        t.pc
    } else if t.newer_layout.load(std::sync::atomic::Ordering::Acquire) {
        t.newer
    } else {
        t.older
    }
}

/// A slot whose console method differs between the title runtimes' layouts: chosen at each
/// call (the layout is found out while the title runs), the argument registers untouched
/// (floating-point ones too).
pub fn make_layout_split(t: LayoutTargets) -> usize {
    thunk(Tail::JumpToResult, Box::leak(Box::new(t)) as *const LayoutTargets as u64, pick_layout as usize)
}

enum Tail {
    /// decide ≠ 0 → `xor eax, eax; ret`, 0 → jump to the original
    GuardOr(usize),
    /// jump to the address decide returned
    JumpToResult,
}

/// A pointer argument: what it points at – a string when it reads like one (a path, an event
/// name), otherwise the first words (a descriptor, the hidden return slot of a struct-returning
/// method, …).
unsafe fn describe_ptr(p: u64) -> String {
    if p <= 0x10000 || !crate::gpu::shader::readable(p as usize, 32) {
        return String::new();
    }
    let b = std::slice::from_raw_parts(p as *const u8, 32);
    let printable = |c: u8| (0x20..0x7f).contains(&c);
    // how far a string may be scanned without leaving readable memory
    let span = if crate::gpu::shader::readable(p as usize, 128) { 128 } else { 32 };
    if b[..8].chunks(2).all(|c| printable(c[0]) && c[1] == 0) {
        let n = (0..span / 2).take_while(|&k| *(p as *const u16).add(k) != 0).count();
        let s = String::from_utf16_lossy(std::slice::from_raw_parts(p as *const u16, n));
        return format!(" -> L\"{s}\"");
    }
    if b[..4].iter().all(|&c| printable(c)) {
        let n = (0..span).take_while(|&k| *(p as *const u8).add(k) != 0).count();
        let s = String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n));
        return format!(" -> \"{s}\"");
    }
    let w: Vec<String> = (0..4).map(|k| format!("{:#x}", *(p as *const u64).add(k))).collect();
    format!(" -> [{}]", w.join(", "))
}

/// `frame`: the thunk's stack at the call – the saved r9, r8, rdx, rcx at +0x68.., the return
/// address at +0x88 and the caller's stack arguments from +0x90 + 0x20 (the shadow space).
unsafe extern "system" fn trace(caller: usize, name: *const String, frame: *const u64) -> usize {
    if crate::modules::image_base_for_addr(caller).is_some() {
        let at = |off: usize| *frame.add(off / 8);
        let (a1, a2, a3) = (at(0x78), at(0x70), at(0x68));
        let (a4, a5) = (at(0x90 + 0x20), at(0x90 + 0x28));
        crate::xlog_info!(
            "D3D12 trace: {} ← {}: ({a1:#x}{}, {a2:#x}{}, {a3:#x}{}, {a4:#x}, {a5:#x})",
            *name,
            crate::modules::describe_address(caller),
            describe_ptr(a1),
            describe_ptr(a2),
            describe_ptr(a3)
        );
    }
    0
}

/// Like a guard, but always forwards: calls from a game image are logged (`XCL_D3D12_TRACE`).
/// The original sees the caller's return address (the slot splits classify by it), so the
/// result is not seen here; `make_call_tracer` logs it but hides the caller from the callee.
pub fn make_tracer(original: usize, name: String) -> usize {
    thunk(Tail::GuardOr(original), Box::leak(Box::new(name)) as *const String as u64, trace as usize)
}

/// The call-through tracer's frame (see `make_call_tracer`).
const CALL_FRAME: usize = 0xA8;
/// How many of the caller's stack arguments the call-through tracer forwards.
const CALL_STACK_ARGS: usize = 8;

/// `frame`: the call-through tracer's frame after the original returned – the copied stack
/// arguments at +0x20.., the entry rcx, rdx, r8, r9 at +0x60.., rax at +0x80, the caller's
/// return address at +0xA8.
unsafe extern "system" fn trace_ret(caller: usize, name: *const String, frame: *const u64) {
    let at = |off: usize| *frame.add(off / 8);
    let (a1, a2, a3, a4) = (at(0x60), at(0x68), at(0x70), at(0x78));
    let (a5, a6, ret) = (at(0x20), at(0x28), at(0x80));
    crate::xlog_info!(
        "trace: {} ← {}: ({a1:#x}{}, {a2:#x}{}, {a3:#x}{}, {a4:#x}{}, {a5:#x}, {a6:#x}) → {ret:#x}",
        *name,
        crate::modules::describe_address(caller),
        describe_ptr(a1),
        describe_ptr(a2),
        describe_ptr(a3),
        describe_ptr(a4)
    );
}

/// A tracer that logs the arguments *and the result* of every call (`XCL_TRACE_IMPORTS`): it
/// calls the original from a frame of its own (forwarding `CALL_STACK_ARGS` stack arguments –
/// enough for every console import; a variadic one past that is truncated) and logs when it
/// returns. The callee sees the thunk as its caller, so this is for plain functions, not for
/// D3D12.X slots whose guards classify by the return address. The frame carries unwind info
/// so an exception raised by the original unwinds through it.
///
/// ```text
/// sub rsp, 0xA8 ; save rcx, rdx, r8, r9 at +0x60.. ; copy stack args [+0xD0..] → [+0x20..]
/// call original ; save rax at +0x80, xmm0 at +0x88
/// rcx = [rsp+0xA8] (return address), rdx = name, r8 = rsp ; call trace_ret
/// restore xmm0, rax ; add rsp, 0xA8 ; ret
/// ```
pub fn make_call_tracer(original: usize, name: String) -> usize {
    let name = Box::leak(Box::new(name)) as *const String as u64;
    let mut c: Vec<u8> = Vec::with_capacity(256);
    c.extend_from_slice(&[0x48, 0x81, 0xEC]); // sub rsp, imm32
    c.extend_from_slice(&(CALL_FRAME as u32).to_le_bytes());
    let prolog = c.len() as u8;
    c.extend_from_slice(&[0x48, 0x89, 0x4C, 0x24, 0x60]); // mov [rsp+0x60], rcx
    c.extend_from_slice(&[0x48, 0x89, 0x54, 0x24, 0x68]); // mov [rsp+0x68], rdx
    c.extend_from_slice(&[0x4C, 0x89, 0x44, 0x24, 0x70]); // mov [rsp+0x70], r8
    c.extend_from_slice(&[0x4C, 0x89, 0x4C, 0x24, 0x78]); // mov [rsp+0x78], r9
    for k in 0..CALL_STACK_ARGS {
        // the caller's k-th stack argument: above our frame, its return address and shadow space
        let src = (CALL_FRAME + 8 + 0x20 + 8 * k) as u32;
        c.extend_from_slice(&[0x48, 0x8B, 0x84, 0x24]); // mov rax, [rsp+disp32]
        c.extend_from_slice(&src.to_le_bytes());
        c.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]); // mov [rsp+disp32], rax
        c.extend_from_slice(&((0x20 + 8 * k) as u32).to_le_bytes());
    }
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    c.extend_from_slice(&(original as u64).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xD0]); // call rax
    c.extend_from_slice(&[0x48, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]); // mov [rsp+0x80], rax
    c.extend_from_slice(&[0xF3, 0x0F, 0x7F, 0x84, 0x24, 0x88, 0x00, 0x00, 0x00]); // movdqu [rsp+0x88], xmm0
    c.extend_from_slice(&[0x48, 0x8B, 0x8C, 0x24]); // mov rcx, [rsp+CALL_FRAME]
    c.extend_from_slice(&(CALL_FRAME as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0xBA]); // mov rdx, imm64
    c.extend_from_slice(&name.to_le_bytes());
    c.extend_from_slice(&[0x49, 0x89, 0xE0]); // mov r8, rsp
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    c.extend_from_slice(&(trace_ret as usize as u64).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xD0]); // call rax
    c.extend_from_slice(&[0xF3, 0x0F, 0x6F, 0x84, 0x24, 0x88, 0x00, 0x00, 0x00]); // movdqu xmm0, [rsp+0x88]
    c.extend_from_slice(&[0x48, 0x8B, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]); // mov rax, [rsp+0x80]
    c.extend_from_slice(&[0x48, 0x81, 0xC4]); // add rsp, imm32
    c.extend_from_slice(&(CALL_FRAME as u32).to_le_bytes());
    c.push(0xC3); // ret
    let code_len = c.len();
    // UNWIND_INFO (4-aligned): version 1, no flags, the prolog, one 2-slot code: UWOP_ALLOC_LARGE
    // (op 1, info 0) of CALL_FRAME / 8, no frame register.
    while c.len() % 4 != 0 {
        c.push(0xCC);
    }
    let unwind = c.len() as u32;
    c.extend_from_slice(&[0x01, prolog, 2, 0x00]);
    c.extend_from_slice(&[prolog, 0x01]);
    c.extend_from_slice(&((CALL_FRAME / 8) as u16).to_le_bytes());
    // RUNTIME_FUNCTION, relative to the thunk itself
    let func = c.len();
    c.extend_from_slice(&0u32.to_le_bytes());
    c.extend_from_slice(&(code_len as u32).to_le_bytes());
    c.extend_from_slice(&unwind.to_le_bytes());
    let p = stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            p as *const _,
            c.len(),
        );
        windows_sys::Win32::System::Diagnostics::Debug::RtlAddFunctionTable(p.add(func) as *const _, 1, p as u64);
    }
    p as usize
}

/// Thunk: `decide(return address, data)` with all argument registers preserved, then `tail`.
fn thunk(tail: Tail, data: u64, decide: usize) -> usize {
    let mut c: Vec<u8> = Vec::with_capacity(128);
    c.extend_from_slice(&[0x51, 0x52, 0x41, 0x50, 0x41, 0x51]); // push rcx, rdx, r8, r9
    c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x68]); // sub rsp, 0x68
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x7F, reg, 0x24, off]); // movdqu [rsp+off], xmmN
    }
    c.extend_from_slice(&[0x48, 0x8B, 0x8C, 0x24, 0x88, 0x00, 0x00, 0x00]); // mov rcx, [rsp+0x88]
    c.extend_from_slice(&[0x48, 0xBA]); // mov rdx, imm64
    c.extend_from_slice(&data.to_le_bytes());
    c.extend_from_slice(&[0x49, 0x89, 0xE0]); // mov r8, rsp (the saved registers and the caller's stack, for `trace`)
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    c.extend_from_slice(&(decide as u64).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xD0]); // call rax
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x6F, reg, 0x24, off]); // movdqu xmmN, [rsp+off]
    }
    c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x68]); // add rsp, 0x68
    c.extend_from_slice(&[0x41, 0x59, 0x41, 0x58, 0x5A, 0x59]); // pop r9, r8, rdx, rcx
    match tail {
        Tail::GuardOr(original) => {
            c.extend_from_slice(&[0x48, 0x85, 0xC0]); // test rax, rax
            c.extend_from_slice(&[0x75, 0x0C]); // jnz +12
            c.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
            c.extend_from_slice(&(original as u64).to_le_bytes());
            c.extend_from_slice(&[0xFF, 0xE0]); // jmp rax
            c.extend_from_slice(&[0x31, 0xC0, 0xC3]); // xor eax, eax ; ret
        }
        Tail::JumpToResult => c.extend_from_slice(&[0xFF, 0xE0]), // jmp rax
    }
    let p = stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            p as *const _,
            c.len(),
        );
    }
    p as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn original(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
        a + 10 * b + 100 * c + 1000 * d + 10000 * e + (f * 100000.0) as u64
    }

    extern "system" fn console(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
        original(a, b, c, d, e, f) + 7
    }

    #[test]
    fn split_sends_non_game_callers_to_the_pc_method() {
        let s = make_split(console as usize, original as usize);
        let f: extern "system" fn(u64, u64, u64, u64, u64, f64) -> u64 = unsafe { std::mem::transmute(s) };
        assert_eq!(f(1, 2, 3, 4, 5, 6.0), 654321);
    }

    extern "system" fn newer(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
        original(a, b, c, d, e, f) + 9
    }

    /// Non-game callers reach the PC method whatever the layout, with every argument (stack
    /// and floating-point ones too) intact; the layout flag is read at each call.
    #[test]
    fn layout_split_keeps_the_runtime_on_the_pc_method() {
        static NEWER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let s = make_layout_split(LayoutTargets { older: console as usize, newer: newer as usize, pc: original as usize, newer_layout: &NEWER });
        let f: extern "system" fn(u64, u64, u64, u64, u64, f64) -> u64 = unsafe { std::mem::transmute(s) };
        assert_eq!(f(1, 2, 3, 4, 5, 6.0), 654321);
        NEWER.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(f(1, 2, 3, 4, 5, 6.0), 654321);
        // the choice itself
        let t = LayoutTargets { older: 1, newer: 2, pc: 3, newer_layout: &NEWER };
        assert_eq!(unsafe { pick_layout(0, &t) }, 3, "not from a game image");
    }

    extern "system" fn halves(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> f64 {
        original(a, b, c, d, e, f) as f64 / 2.0
    }

    /// Integer, stack and floating-point arguments reach the original and its result (integer
    /// or floating-point) comes back through the call-through tracer.
    #[test]
    fn call_tracer_forwards_arguments_and_results() {
        let t = make_call_tracer(original as usize, "unit!original".into());
        let f: extern "system" fn(u64, u64, u64, u64, u64, f64) -> u64 = unsafe { std::mem::transmute(t) };
        assert_eq!(f(1, 2, 3, 4, 5, 6.0), 654321);
        let t = make_call_tracer(halves as usize, "unit!halves".into());
        let f: extern "system" fn(u64, u64, u64, u64, u64, f64) -> f64 = unsafe { std::mem::transmute(t) };
        assert_eq!(f(1, 2, 3, 4, 5, 6.0), 327160.5);
    }

    #[test]
    fn guard_forwards_calls_that_do_not_come_from_a_game_image() {
        let info = stubs::new_info("unit.dll", "IGuard::#7", 0);
        let g = make_guard(original as usize, info);
        let f: extern "system" fn(u64, u64, u64, u64, u64, f64) -> u64 = unsafe { std::mem::transmute(g) };
        // all integer args, a stack arg and a float arg survive the round trip
        assert_eq!(f(1, 2, 3, 4, 5, 6.0), 654321);
        assert_eq!(info.calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}
