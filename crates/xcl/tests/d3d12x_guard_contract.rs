//! Contract tests for `xcl::hle::d3d12x::guard` x64 thunks (`make_guard`, `make_split`,
//! `make_layout_split`, `make_tracer`, and `make_call_tracer`) under both mapped game-image
//! callers and non-game (runtime) callers.

#![allow(function_casts_as_integer)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use xcl::hle::d3d12x::guard::{
    make_call_tracer, make_guard, make_layout_split, make_split, make_tracer, LayoutTargets,
};
use xcl::modules::{self, MappedModule};
use xcl::pe::OwnedExports;
use xcl::stubs;

const PAGE_SIZE: usize = 0x1000;
const TRAMP_FRAME: u8 = 0x68; // 0x20 shadow + 8*8 stack args + 8 align

/// Executable page registered as a `MappedModule` so calls emitted from it have return
/// addresses classified as game-image callers by `modules::image_base_for_addr`.
struct GameImage {
    base: usize,
    next: usize,
    registered: bool,
}

impl GameImage {
    fn new(name: &str) -> Self {
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
        } as usize;
        assert_ne!(p, 0, "VirtualAlloc for GameImage failed");
        let module = Arc::new(MappedModule {
            name: name.to_ascii_lowercase(),
            path: PathBuf::from(name),
            base: p,
            size: PAGE_SIZE,
            entry_rva: 0,
            is_dll: false,
            exports: OwnedExports::default(),
            tls_callbacks: Mutex::new(Vec::new()),
            no_thread_calls: AtomicBool::new(false),
            initialized: AtomicBool::new(true),
            stack_reserve: 0,
        });
        modules::register_mapped(module);
        Self { base: p, next: p, registered: true }
    }

    fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.base + PAGE_SIZE
    }

    fn unregister(&mut self) {
        if self.registered {
            modules::unregister_mapped(self.base);
            self.registered = false;
        }
    }

    /// Emit a Win64 call trampoline inside this game image that forwards all register args
    /// (`rcx/xmm0..r9/xmm3`) and `stack_args` (up to 8) to `target`, preserving `rax` and `xmm0`
    /// on return. Returns `(trampoline_entry, expected_return_address)`.
    fn emit_trampoline(&mut self, target: usize, stack_args: usize) -> (usize, usize) {
        assert!(stack_args <= 8, "trampoline supports at most 8 stack arguments");
        let mut c: Vec<u8> = Vec::with_capacity(128);
        c.extend_from_slice(&[0x48, 0x83, 0xEC, TRAMP_FRAME]); // sub rsp, 0x68
        for k in 0..stack_args {
            let src = (TRAMP_FRAME as u32) + 0x28 + (8 * k as u32);
            let dst = 0x20u8 + (8 * k as u8);
            c.extend_from_slice(&[0x48, 0x8B, 0x84, 0x24]); // mov rax, [rsp+disp32]
            c.extend_from_slice(&src.to_le_bytes());
            c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, dst]); // mov [rsp+disp8], rax
        }
        c.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
        c.extend_from_slice(&(target as u64).to_le_bytes());
        c.extend_from_slice(&[0xFF, 0xD0]); // call rax
        let ret_offset = c.len();
        c.extend_from_slice(&[0x48, 0x83, 0xC4, TRAMP_FRAME]); // add rsp, 0x68
        c.push(0xC3); // ret

        let entry = (self.next + 15) & !15;
        assert!(entry + c.len() <= self.base + PAGE_SIZE, "GameImage page exhausted");
        unsafe {
            std::ptr::copy_nonoverlapping(c.as_ptr(), entry as *mut u8, c.len());
            FlushInstructionCache(GetCurrentProcess(), entry as *const _, c.len());
        }
        self.next = entry + c.len();
        (entry, entry + ret_offset)
    }
}

impl Drop for GameImage {
    fn drop(&mut self) {
        self.unregister();
        unsafe {
            VirtualFree(self.base as *mut _, 0, MEM_RELEASE);
        }
    }
}

type MixedFn = extern "system" fn(u64, u64, u64, u64, u64, f64) -> u64;
type XmmFn = extern "system" fn(f64, f64, f64, f64, u64, u64) -> f64;
type TwelveU64Fn =
    extern "system" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
type TwelveF64Fn =
    extern "system" fn(f64, f64, f64, f64, u64, u64, u64, u64, u64, u64, u64, u64) -> f64;

extern "system" fn pc_mixed(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
    a + 10 * b + 100 * c + 1000 * d + 10_000 * e + (f * 100_000.0) as u64
}

extern "system" fn console_mixed(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
    pc_mixed(a, b, c, d, e, f) + 7
}

extern "system" fn newer_mixed(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
    pc_mixed(a, b, c, d, e, f) + 9
}

extern "system" fn pc_xmm(x0: f64, x1: f64, x2: f64, x3: f64, s0: u64, s1: u64) -> f64 {
    x0 + 2.0 * x1 + 4.0 * x2 + 8.0 * x3 + (s0 as f64) * 16.0 + (s1 as f64) * 32.0
}

extern "system" fn console_xmm(x0: f64, x1: f64, x2: f64, x3: f64, s0: u64, s1: u64) -> f64 {
    pc_xmm(x0, x1, x2, x3, s0, s1) + 0.25
}

extern "system" fn newer_xmm(x0: f64, x1: f64, x2: f64, x3: f64, s0: u64, s1: u64) -> f64 {
    pc_xmm(x0, x1, x2, x3, s0, s1) + 0.75
}

static GUARD_ORIG_CALLS: AtomicU64 = AtomicU64::new(0);

extern "system" fn counted_mixed(a: u64, b: u64, c: u64, d: u64, e: u64, f: f64) -> u64 {
    GUARD_ORIG_CALLS.fetch_add(1, Ordering::Relaxed);
    pc_mixed(a, b, c, d, e, f)
}

#[allow(clippy::too_many_arguments)]
extern "system" fn twelve_u64(
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
) -> u64 {
    let args = [a0, a1, a2, a3, s0, s1, s2, s3, s4, s5, s6, s7];
    let mut acc = 0xDEAD_BEEF_0000_0000u64;
    for (i, &v) in args.iter().enumerate() {
        acc = acc.wrapping_add(v.wrapping_mul((i as u64) + 1));
    }
    acc
}

#[allow(clippy::too_many_arguments)]
extern "system" fn twelve_f64(
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
) -> f64 {
    let stack_sum = s0 + 2 * s1 + 3 * s2 + 4 * s3 + 5 * s4 + 6 * s5 + 7 * s6 + 8 * s7;
    x0 + 2.0 * x1 + 4.0 * x2 + 8.0 * x3 + (stack_sum as f64) + 0.375
}

#[test]
fn guard_suppresses_game_callers_and_records_stub_info() {
    GUARD_ORIG_CALLS.store(0, Ordering::Relaxed);
    let info = stubs::new_info("d3d12_xs.dll", "ID3D12Device::GuardContract", 0);
    let guard = make_guard(counted_mixed as usize, info);

    let mut game = GameImage::new("guard_test.exe");
    let (tramp1, ret1) = game.emit_trampoline(guard, 2);
    let (tramp2, ret2) = game.emit_trampoline(guard, 2);
    assert_ne!(ret1, ret2);
    assert!(game.contains(ret1) && game.contains(ret2));

    let game_call_1: MixedFn = unsafe { std::mem::transmute(tramp1) };
    let game_call_2: MixedFn = unsafe { std::mem::transmute(tramp2) };
    let runtime_call: MixedFn = unsafe { std::mem::transmute(guard) };

    // Game callers return 0 without calling original, count every call, and lock first_caller.
    assert_eq!(game_call_1(1, 2, 3, 4, 5, 6.0), 0);
    assert_eq!(GUARD_ORIG_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(info.calls.load(Ordering::Relaxed), 1);
    assert_eq!(info.first_caller.load(Ordering::Relaxed), ret1);

    assert_eq!(game_call_2(9, 8, 7, 6, 5, 4.0), 0);
    assert_eq!(GUARD_ORIG_CALLS.load(Ordering::Relaxed), 0);
    assert_eq!(info.calls.load(Ordering::Relaxed), 2);
    assert_eq!(info.first_caller.load(Ordering::Relaxed), ret1);

    // Non-game callers bypass the stub and forward register + stack + float args intact.
    assert_eq!(runtime_call(1, 2, 3, 4, 5, 6.0), 654_321);
    assert_eq!(GUARD_ORIG_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(info.calls.load(Ordering::Relaxed), 2);

    // XMM0..3 register arguments also survive the non-game guard path.
    let xmm_info = stubs::new_info("d3d12_xs.dll", "ID3D12Device::GuardXmm", 0);
    let xmm_guard = make_guard(pc_xmm as usize, xmm_info);
    let runtime_xmm: XmmFn = unsafe { std::mem::transmute(xmm_guard) };
    assert_eq!(runtime_xmm(1.5, 2.25, 3.5, 4.75, 5, 6), pc_xmm(1.5, 2.25, 3.5, 4.75, 5, 6));
    assert_eq!(xmm_info.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn split_routes_game_to_console_and_runtime_to_pc() {
    let split_mixed = make_split(console_mixed as usize, pc_mixed as usize);
    let split_xmm = make_split(console_xmm as usize, pc_xmm as usize);

    let mut game = GameImage::new("split_test.exe");
    let (tramp_mixed, _) = game.emit_trampoline(split_mixed, 2);
    let (tramp_xmm, _) = game.emit_trampoline(split_xmm, 2);

    let game_mixed: MixedFn = unsafe { std::mem::transmute(tramp_mixed) };
    let runtime_mixed: MixedFn = unsafe { std::mem::transmute(split_mixed) };
    assert_eq!(game_mixed(1, 2, 3, 4, 5, 6.0), 654_328);
    assert_eq!(runtime_mixed(1, 2, 3, 4, 5, 6.0), 654_321);

    let game_xmm: XmmFn = unsafe { std::mem::transmute(tramp_xmm) };
    let runtime_xmm: XmmFn = unsafe { std::mem::transmute(split_xmm) };
    assert_eq!(game_xmm(1.5, 2.5, 3.5, 4.5, 5, 6), console_xmm(1.5, 2.5, 3.5, 4.5, 5, 6));
    assert_eq!(runtime_xmm(1.5, 2.5, 3.5, 4.5, 5, 6), pc_xmm(1.5, 2.5, 3.5, 4.5, 5, 6));

    // Once unregistered, the same executable page is classified as non-game and routes to pc.
    game.unregister();
    assert_eq!(game_mixed(1, 2, 3, 4, 5, 6.0), 654_321);
    assert_eq!(game_xmm(1.5, 2.5, 3.5, 4.5, 5, 6), pc_xmm(1.5, 2.5, 3.5, 4.5, 5, 6));
}

#[test]
fn layout_split_switches_game_targets_dynamically_while_runtime_stays_on_pc() {
    static NEWER: AtomicBool = AtomicBool::new(false);
    NEWER.store(false, Ordering::Release);

    let split_mixed = make_layout_split(LayoutTargets {
        older: console_mixed as usize,
        newer: newer_mixed as usize,
        pc: pc_mixed as usize,
        newer_layout: &NEWER,
    });
    let split_xmm = make_layout_split(LayoutTargets {
        older: console_xmm as usize,
        newer: newer_xmm as usize,
        pc: pc_xmm as usize,
        newer_layout: &NEWER,
    });

    let mut game = GameImage::new("layout_split_test.exe");
    let (tramp_mixed, _) = game.emit_trampoline(split_mixed, 2);
    let (tramp_xmm, _) = game.emit_trampoline(split_xmm, 2);

    let game_mixed: MixedFn = unsafe { std::mem::transmute(tramp_mixed) };
    let runtime_mixed: MixedFn = unsafe { std::mem::transmute(split_mixed) };
    let game_xmm: XmmFn = unsafe { std::mem::transmute(tramp_xmm) };
    let runtime_xmm: XmmFn = unsafe { std::mem::transmute(split_xmm) };

    assert_eq!(game_mixed(1, 2, 3, 4, 5, 6.0), 654_328);
    assert_eq!(runtime_mixed(1, 2, 3, 4, 5, 6.0), 654_321);
    assert_eq!(game_xmm(1.0, 2.0, 3.0, 4.0, 5, 6), console_xmm(1.0, 2.0, 3.0, 4.0, 5, 6));
    assert_eq!(runtime_xmm(1.0, 2.0, 3.0, 4.0, 5, 6), pc_xmm(1.0, 2.0, 3.0, 4.0, 5, 6));

    NEWER.store(true, Ordering::Release);
    assert_eq!(game_mixed(1, 2, 3, 4, 5, 6.0), 654_330);
    assert_eq!(runtime_mixed(1, 2, 3, 4, 5, 6.0), 654_321);
    assert_eq!(game_xmm(1.0, 2.0, 3.0, 4.0, 5, 6), newer_xmm(1.0, 2.0, 3.0, 4.0, 5, 6));
    assert_eq!(runtime_xmm(1.0, 2.0, 3.0, 4.0, 5, 6), pc_xmm(1.0, 2.0, 3.0, 4.0, 5, 6));

    NEWER.store(false, Ordering::Release);
    assert_eq!(game_mixed(1, 2, 3, 4, 5, 6.0), 654_328);
}

#[test]
fn tracer_forwards_both_callers_and_preserves_return_address_for_nested_split() {
    let traced_plain = make_tracer(pc_mixed as usize, "unit!traced_plain".into());
    let nested_split = make_split(console_mixed as usize, pc_mixed as usize);
    let traced_split = make_tracer(nested_split, "unit!traced_split".into());

    let guard_info = stubs::new_info("d3d12_xs.dll", "ID3D12Device::TracedGuard", 0);
    let traced_guard =
        make_tracer(make_guard(pc_mixed as usize, guard_info), "unit!traced_guard".into());

    let mut game = GameImage::new("tracer_test.exe");
    let (tramp_plain, _) = game.emit_trampoline(traced_plain, 2);
    let (tramp_split, _) = game.emit_trampoline(traced_split, 2);
    let (tramp_guard, ret_guard) = game.emit_trampoline(traced_guard, 2);

    // Plain tracer forwards both game and non-game callers with args and return intact.
    let game_plain: MixedFn = unsafe { std::mem::transmute(tramp_plain) };
    let runtime_plain: MixedFn = unsafe { std::mem::transmute(traced_plain) };
    assert_eq!(game_plain(1, 2, 3, 4, 5, 6.0), 654_321);
    assert_eq!(runtime_plain(1, 2, 3, 4, 5, 6.0), 654_321);

    // Tail-jump preserves the caller's return address so a nested split classifies accurately.
    let game_split: MixedFn = unsafe { std::mem::transmute(tramp_split) };
    let runtime_split: MixedFn = unsafe { std::mem::transmute(traced_split) };
    assert_eq!(game_split(1, 2, 3, 4, 5, 6.0), 654_328);
    assert_eq!(runtime_split(1, 2, 3, 4, 5, 6.0), 654_321);

    // A nested guard also sees the original game return address, not the tracer thunk.
    let game_guard: MixedFn = unsafe { std::mem::transmute(tramp_guard) };
    assert_eq!(game_guard(1, 2, 3, 4, 5, 6.0), 0);
    assert_eq!(guard_info.calls.load(Ordering::Relaxed), 1);
    assert_eq!(guard_info.first_caller.load(Ordering::Relaxed), ret_guard);
}

#[test]
fn call_tracer_forwards_all_twelve_args_and_preserves_int_and_float_returns() {
    let traced_u64 = make_call_tracer(twelve_u64 as usize, "unit!twelve_u64".into());
    let traced_f64 = make_call_tracer(twelve_f64 as usize, "unit!twelve_f64".into());
    let traced_split = make_call_tracer(
        make_split(console_mixed as usize, pc_mixed as usize),
        "unit!call_traced_split".into(),
    );

    let mut game = GameImage::new("call_tracer_test.exe");
    let (tramp_u64, _) = game.emit_trampoline(traced_u64, 8);
    let (tramp_f64, _) = game.emit_trampoline(traced_f64, 8);
    let (tramp_split, _) = game.emit_trampoline(traced_split, 2);

    let expected_u64 = twelve_u64(11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333);
    let game_u64: TwelveU64Fn = unsafe { std::mem::transmute(tramp_u64) };
    let runtime_u64: TwelveU64Fn = unsafe { std::mem::transmute(traced_u64) };
    assert_eq!(game_u64(11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333), expected_u64);
    assert_eq!(runtime_u64(11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333), expected_u64);

    let expected_f64 = twelve_f64(1.25, 2.5, 3.75, 5.0, 10, 20, 30, 40, 50, 60, 70, 80);
    let game_f64: TwelveF64Fn = unsafe { std::mem::transmute(tramp_f64) };
    let runtime_f64: TwelveF64Fn = unsafe { std::mem::transmute(traced_f64) };
    assert_eq!(game_f64(1.25, 2.5, 3.75, 5.0, 10, 20, 30, 40, 50, 60, 70, 80), expected_f64);
    assert_eq!(runtime_f64(1.25, 2.5, 3.75, 5.0, 10, 20, 30, 40, 50, 60, 70, 80), expected_f64);

    // Unlike make_tracer, make_call_tracer calls from its own frame (non-game), so a nested
    // split sees the thunk as the caller and routes to pc even when invoked from a game image.
    let game_split: MixedFn = unsafe { std::mem::transmute(tramp_split) };
    assert_eq!(game_split(1, 2, 3, 4, 5, 6.0), 654_321);
}
