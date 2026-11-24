//! The console's audio processor (`acphal.dll`, `xapu.dll`): memory the audio hardware
//! addresses, the processor's command / message channel, SHAPE contexts and the decoder queue.
//!
//! The processor answers as hardware with little to do: commands are acknowledged, audio frames
//! tick (one every 128 samples at 48 kHz), SHAPE context arrays exist. A title's audio engine then
//! starts and runs (XDefiant: `ApuCreateHeap` failing made it drop its whole audio setup), and
//! nothing waits forever on the processor. The SHAPE mixer is not run.
//!
//! What runs is the decoder (xapu): Opus packets a title sends are decoded (`hle::opus`: Windows'
//! Media Foundation decoder, or libopus where Windows has none – without either the decoder answers
//! silence, logged once) and the PCM is written into the title's output buffer. Nothing is played
//! from here: the title reads that output back itself, mixes it and plays its own mix through a
//! system audio endpoint (Windows Spatial Audio), as a listening test showed – with the decoder's
//! PCM played on a voice of XCL's own as well, the sound came out twice. So XCL only has to decode
//! into the buffer. The SHAPE mixer is not run. `XCL_TRACE_APU=1` logs what a title sends (acphal
//! commands, xapu commands with their structures and the start of the Opus input, the caller in the
//! title, the rate per kind of command).
//!
//! # The decoder's commands as a retail game sends them (observed with `XCL_TRACE_APU=1`)
//!
//! Every command starts with the 24-byte `XApuCommand` (`kind`, `stream`, `sequence`, `options`,
//! `context` u64); the title builds it on its stack and sets only the fields it uses (`kind` and
//! `stream`, the others are left as they were – `sequence`, `options`, `context` mean nothing to it),
//! then calls `XApuEnqueueCommand(handle, &command, &count)` and polls `XApuDequeueResult`.
//! The game drives stream 0 only (so far) and is on one thread.
//!
//! * kind 1, Activate: `u32 channels` at +24 (1: mono, 2: stereo), the output rate 48000 at +56 (the
//!   decoder is 48 kHz). Starts a stream.
//! * kind 3, Process: one Opus packet (RFC 6716, no length prefix, no container: 80 / 160 B mono
//!   CELT full band 20 ms, TOC 0xF8; 320 B stereo, TOC 0xFC) in `input` / `input_bytes`; `frames` = 960
//!   (one packet), `first_frame` = 312 on the first command of a stream (the Opus pre-skip: decoded
//!   frames to leave out), 0 after; the output is `frames × channels` float32 at `output`
//!   (3 840 B mono). The input and output buffers are the same two for every command of the stream
//!   (a ring the title refills); `ramp` is 0, the f32 "pitch" field holds 48000.0. A command every ~20 ms
//!   once the title's own prefetch (a burst of ~20 commands, 4–6 ms apart) is done.
//! * kind 2, Deactivate: nothing after the header (the 40 bytes behind are the title's stack); the
//!   title sends it when it releases the stream (`kind`, `stream`, sequence 0), then frees the
//!   stream's slot. Activate / Deactivate alternate with no Process in between for short sounds.
//! * results: `state` bits 0 (output ready) and 1 (more input wanted), `frames` / `output_bytes`
//!   the frames written (those after `first_frame`), `input` / `input_bytes` as sent.
//! * acphal: one command (`type` 2, id 0xF0F0F0F0, no data) right after `Connect`, nothing after
//!   it during menus / sound (the SHAPE contexts are written in APU memory, not through commands);
//!   the DSP (chad) gets nothing.
//! * APU memory: `ApuCreateHeap`, then `ApuAlloc` of 0x5780 (align 0x100), 0x222E00 (align 0x800,
//!   flags 2) and 0x1913AFA0 (align 0x800, flags 1); `ApuMapVirtualAddress` is not called.
//!
//! Interfaces as titles call them (observed on retail titles; another project's notes on
//! the same titles agree): `ApuAlloc(&va, &apu, size, alignment, flags)`,
//! `ApuMapVirtualAddress(&apu, va, size)`, `ApuMapApuAddress(apu) -> va`,
//! `AcpHalCreate(&hal)` – an object whose vtable is Connect(commands, messages), Disconnect,
//! SubmitCommand(type, id, frame, data, size), PopMessage(&message) -> bool,
//! GetNumMessages, Release; messages {u32 type, u32, u64 value, u64 id}: 2 audio frame,
//! 0x10 command done (id), 0x80 disconnected. `chad.dll` (XDsp, the processor's DSP) connects
//! and takes commands that do nothing: a retail game stops at a breakpoint when it cannot connect
//! (XDefiant took E_NOTIMPL, the stub database's answer before).
//! `XDspConnectWithMaximumStreamLimit(a, b, c, max streams, flags, &handle)`,
//! `XDspActivate(handle, …)`, `XDspSubmitCommand(handle, …)`, `XDspDeactivate`, `XDspDisconnect`.

#![allow(non_snake_case)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::hle::{opus, xaudio2};
use crate::util::{HRESULT, S_OK};

const E_INVALIDARG: HRESULT = 0x8007_0057_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
const E_UNEXPECTED: HRESULT = 0x8000_FFFF_u32 as i32;
/// XApuDequeueResult: no result waiting
const XAPU_NO_RESULTS: HRESULT = 0x8AC8_1001_u32 as i32;

/// one audio frame: 128 samples at 48 kHz
const FRAME: Duration = Duration::from_micros(2667);

// ---------------------------------------------------------------------------------------
// XCL_TRACE_APU=1: what the title sends the audio processor and the decoder
// ---------------------------------------------------------------------------------------

/// `XCL_TRACE_APU` set (not "0"): log the commands, their structures and buffers
fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TRACE_APU").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Per kind of command: how many came, when the first and the last did (for the call frequency),
/// and how many were logged in full.
#[derive(Default)]
struct TraceKind {
    count: u64,
    logged: u32,
    first: Option<Instant>,
    last_report: Option<Instant>,
    since_report: u64,
}

/// the first commands of each kind are logged in full, then one line per second and kind
struct Tracer {
    kinds: Vec<(&'static str, u32, TraceKind)>,
}

static TRACER: Mutex<Tracer> = Mutex::new(Tracer { kinds: Vec::new() });
/// commands of one kind logged in full
const TRACE_FULL: u32 = 24;

/// true: log this command in full; false: tracing is off, or the command is only counted (a rate line
/// is logged here when due)
fn trace_command(origin: &'static str, kind: u32) -> bool {
    if !trace_on() {
        return false;
    }
    let now = Instant::now();
    let mut t = TRACER.lock().unwrap();
    let i = match t.kinds.iter().position(|k| k.0 == origin && k.1 == kind) {
        Some(i) => i,
        None => {
            t.kinds.push((origin, kind, TraceKind::default()));
            t.kinds.len() - 1
        }
    };
    let k = &mut t.kinds[i].2;
    k.count += 1;
    k.since_report += 1;
    k.first.get_or_insert(now);
    if k.logged < TRACE_FULL {
        k.logged += 1;
        return true;
    }
    match k.last_report {
        None => k.last_report = Some(now),
        Some(r) if now.duration_since(r) >= Duration::from_secs(5) => {
            let rate = k.since_report as f64 / now.duration_since(r).as_secs_f64();
            crate::xlog_info!("apu-trace: {origin} command {kind:#x}: {} so far, {rate:.1}/s", k.count);
            k.last_report = Some(now);
            k.since_report = 0;
        }
        _ => {}
    }
    false
}

/// the return addresses of the call into the decoder's queue (`module+offset`), for the first
/// commands of a kind: where in the title's code it comes from
unsafe fn trace_callers(kind: u32) {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    static LOGGED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    {
        let mut l = LOGGED.lock().unwrap();
        if l.iter().filter(|&&k| k == kind).count() >= 2 {
            return;
        }
        l.push(kind);
    }
    let mut frames = [std::ptr::null_mut::<c_void>(); 8];
    let n = RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut());
    let list: Vec<String> = frames[..n as usize].iter().map(|&f| crate::modules::describe_address(f as usize)).collect();
    crate::xlog_info!("apu-trace:   command {kind:#x} called from {}", list.join(" < "));
}

/// the bytes of an xapu command as the title sends them: its own structure, not a fixed size (a
/// command is 24 B for most kinds; reading more would run into whatever the title has behind it)
fn command_len(kind: u32) -> usize {
    match kind {
        XAPU_ACTIVATE => std::mem::size_of::<XApuActivate>(),
        XAPU_PROCESS => std::mem::size_of::<XApuProcess>(),
        _ => std::mem::size_of::<XApuCommand>(),
    }
}

/// up to `n` bytes at `p`, in hex (null: "null")
unsafe fn hex(p: *const c_void, n: usize) -> String {
    if p.is_null() {
        return "null".into();
    }
    let b = std::slice::from_raw_parts(p as *const u8, n);
    b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join("")
}

// ---------------------------------------------------------------------------------------
// memory the audio processor addresses (32-bit APU addresses)
// ---------------------------------------------------------------------------------------

struct Region {
    va: usize,
    size: u32,
    apu: u32,
    /// allocated here (ApuAlloc), released by ApuFree
    owned: bool,
}

struct Apu {
    regions: Vec<Region>,
    next: u32,
}

static APU: Mutex<Apu> = Mutex::new(Apu { regions: Vec::new(), next: 0x1000_0000 });

const APU_BASE: u64 = 0x1000_0000;
const APU_LIMIT: u64 = (u32::MAX as u64) + 1;

impl Apu {
    fn region_end(r: &Region) -> u64 {
        let len = (r.size.max(1) as u64 + 0xFFF) & !0xFFF;
        (r.apu as u64).saturating_add(len)
    }

    fn overlaps(&self, start: u64, end: u64) -> Option<u64> {
        self.regions
            .iter()
            .filter(|r| {
                let rs = r.apu as u64;
                let re = Self::region_end(r);
                start < re && rs < end
            })
            .map(Self::region_end)
            .max()
    }

    fn find_free_range(&self, step: u64) -> Option<u32> {
        if step == 0 || step > APU_LIMIT - APU_BASE {
            return None;
        }
        // Fast path from `self.next`, then wrap once to `APU_BASE` to reclaim freed ranges.
        for start_hint in [self.next as u64, APU_BASE] {
            let mut cand = start_hint.max(APU_BASE);
            while let Some(end) = cand.checked_add(step).filter(|&e| e <= APU_LIMIT) {
                match self.overlaps(cand, end) {
                    None => return Some(cand as u32),
                    Some(skip_to) => {
                        let next = (skip_to + 0xFFF) & !0xFFF;
                        if next <= cand {
                            break;
                        }
                        cand = next;
                    }
                }
            }
        }
        None
    }

    /// The APU address of `va` – in a known region (for non-owned mappings covering the full span),
    /// or a new non-overlapping 32-bit APU range of `size` bytes.
    fn address(&mut self, va: usize, size: u32, owned: bool) -> Option<u32> {
        let span = size.max(1) as usize;
        if !owned {
            if let Some(r) = self.regions.iter().find(|r| {
                va >= r.va && (va - r.va).saturating_add(span) <= r.size.max(1) as usize
            }) {
                return Some(r.apu + (va - r.va) as u32);
            }
        }
        let step = (size.max(1) as u64 + 0xFFF) & !0xFFF;
        let apu = self.find_free_range(step)?;
        let next = (apu as u64) + step;
        self.next = if next < APU_LIMIT { next as u32 } else { APU_BASE as u32 };
        self.regions.push(Region { va, size, apu, owned });
        Some(apu)
    }
}

/// `size` bytes aligned to `align` (zeroed; the allocation's base is `va` rounded down to
/// 64 KiB – VirtualAlloc's granularity – so MEM_RELEASE of it frees all)
unsafe fn alloc_aligned(size: usize, align: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let align = align.max(1).checked_next_power_of_two()?;
    if align <= 0x1_0000 {
        let p = VirtualAlloc(std::ptr::null(), size.max(1), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as usize;
        return (p != 0).then_some(p);
    }
    let reserve = size.checked_add(align)?;
    // reserve enough to find an aligned start, then reserve exactly there
    for _ in 0..8 {
        let p = VirtualAlloc(std::ptr::null(), reserve, MEM_RESERVE, PAGE_READWRITE) as usize;
        if p == 0 {
            return None;
        }
        VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        let a = p.checked_add(align - 1)? & !(align - 1);
        let q = VirtualAlloc(a as *const c_void, size.max(1), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as usize;
        if q == a {
            return Some(q);
        }
        if q != 0 {
            VirtualFree(q as *mut c_void, 0, MEM_RELEASE);
        }
    }
    None
}

/// ApuCreateHeap(size, flags): the heap ApuAlloc takes from – the process's memory here.
pub unsafe extern "system" fn ApuCreateHeap(size: u32, flags: u32) -> HRESULT {
    crate::xlog_info!("acphal: ApuCreateHeap({size:#x}, {flags:#x}) – the audio processor's memory is the process's");
    S_OK
}

/// ApuAlloc(&va, &apu, size, alignment, flags)
pub unsafe extern "system" fn ApuAlloc(va: *mut *mut c_void, apu: *mut u32, size: u32, alignment: u32, _flags: u32) -> HRESULT {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    if va.is_null() {
        return E_INVALIDARG;
    }
    let Some(p) = alloc_aligned(size as usize, alignment as usize) else {
        *va = std::ptr::null_mut();
        return E_OUTOFMEMORY;
    };
    let Some(a) = APU.lock().unwrap().address(p, size, true) else {
        VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        *va = std::ptr::null_mut();
        return E_OUTOFMEMORY;
    };
    *va = p as *mut c_void;
    if !apu.is_null() {
        *apu = a;
    }
    if trace_command("ApuAlloc", 0) {
        crate::xlog_info!("apu-trace: ApuAlloc({size:#x}, align {alignment:#x}, flags {_flags:#x}) -> va {p:#x} apu {a:#x}");
    }
    S_OK
}

/// ApuFree(va)
pub unsafe extern "system" fn ApuFree(va: *mut c_void) -> HRESULT {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    if va.is_null() {
        return E_INVALIDARG;
    }
    let owned = {
        let mut g = APU.lock().unwrap();
        let i = g
            .regions
            .iter()
            .position(|r| r.va == va as usize && r.owned)
            .or_else(|| g.regions.iter().position(|r| r.va == va as usize));
        i.map(|i| g.regions.swap_remove(i).owned).unwrap_or(false)
    };
    if owned {
        VirtualFree(va, 0, MEM_RELEASE);
    }
    S_OK
}

/// ApuMapVirtualAddress(&apu, va, size): an APU address for the title's own memory
pub unsafe extern "system" fn ApuMapVirtualAddress(apu: *mut u32, va: *mut c_void, size: u32) -> HRESULT {
    if apu.is_null() || va.is_null() {
        return E_INVALIDARG;
    }
    let Some(a) = APU.lock().unwrap().address(va as usize, size, false) else {
        return E_OUTOFMEMORY;
    };
    *apu = a;
    if trace_command("ApuMapVirtualAddress", 0) {
        crate::xlog_info!("apu-trace: ApuMapVirtualAddress({va:p}, {size:#x}) -> apu {a:#x}");
    }
    S_OK
}

/// ApuMapApuAddress(apu) -> the memory an APU address stands for (null: none)
pub unsafe extern "system" fn ApuMapApuAddress(apu: u32) -> *mut c_void {
    let g = APU.lock().unwrap();
    g.regions
        .iter()
        .find(|r| apu >= r.apu && apu - r.apu < r.size.max(1))
        .map_or(std::ptr::null_mut(), |r| (r.va + (apu - r.apu) as usize) as *mut c_void)
}

/// ApuIsVirtualAddressValid: all of the process's memory is the processor's here
pub unsafe extern "system" fn ApuIsVirtualAddressValid(_va: *const c_void, _size: u32) -> i32 {
    1
}

/// ApuHeapGetState(&state, flags): free bytes, bytes in use, bytes lost, the largest free
/// block, blocks – there is always room
pub unsafe extern "system" fn ApuHeapGetState(state: *mut u32, _flags: u32) -> HRESULT {
    if state.is_null() {
        return E_INVALIDARG;
    }
    let g = APU.lock().unwrap();
    let used: u32 = g.regions.iter().filter(|r| r.owned).map(|r| r.size).fold(0, u32::wrapping_add);
    let s = std::slice::from_raw_parts_mut(state, 5);
    s.copy_from_slice(&[256 << 20, used, 0, 64 << 20, g.regions.len() as u32]);
    S_OK
}

// ---------------------------------------------------------------------------------------
// SHAPE contexts
// ---------------------------------------------------------------------------------------

/// AcpHalAllocateShapeContexts' argument: counts of the five kinds, then an array of each
#[repr(C)]
pub struct ShapeContexts {
    counts: [u32; 5],
    reserved: u32,
    contexts: [*mut c_void; 5],
}

/// bytes of a context entry (the largest seen)
const CONTEXT_BYTES: u32 = 64;
static SHAPE: Mutex<[usize; 5]> = Mutex::new([0; 5]);

unsafe fn allocate_shape_contexts_with<F>(c: *mut ShapeContexts, mut alloc_fn: F) -> HRESULT
where
    F: FnMut(usize, u32) -> Result<*mut c_void, HRESULT>,
{
    if c.is_null() {
        return E_INVALIDARG;
    }
    let c = &mut *c;
    let mut sizes = [0u32; 5];
    for i in 0..5 {
        if c.counts[i] != 0 {
            let Some(bytes) = c.counts[i].checked_mul(CONTEXT_BYTES) else {
                c.contexts = [std::ptr::null_mut(); 5];
                return E_OUTOFMEMORY;
            };
            sizes[i] = bytes;
        }
    }
    let mut new_arrays = [0usize; 5];
    let mut new_ptrs = [std::ptr::null_mut(); 5];
    for i in 0..5 {
        if sizes[i] == 0 {
            continue;
        }
        match alloc_fn(i, sizes[i]) {
            Ok(va) => {
                new_arrays[i] = va as usize;
                new_ptrs[i] = va;
            }
            Err(hr) => {
                for &a in &new_arrays[..i] {
                    if a != 0 {
                        ApuFree(a as *mut c_void);
                    }
                }
                c.contexts = [std::ptr::null_mut(); 5];
                return hr;
            }
        }
    }
    let old_arrays = {
        let mut arrays = SHAPE.lock().unwrap();
        let old = std::mem::replace(&mut *arrays, new_arrays);
        c.contexts = new_ptrs;
        old
    };
    for a in old_arrays.into_iter().filter(|&a| a != 0) {
        ApuFree(a as *mut c_void);
    }
    crate::xlog_info!("acphal: SHAPE contexts {:?} (they do not run: no audio from the console's mixer)", c.counts);
    S_OK
}

/// AcpHalAllocateShapeContexts(&contexts): an array of each kind in APU memory (none run)
pub unsafe extern "system" fn AcpHalAllocateShapeContexts(c: *mut ShapeContexts) -> HRESULT {
    allocate_shape_contexts_with(c, |_, bytes| {
        let mut va = std::ptr::null_mut();
        let hr = ApuAlloc(&mut va, std::ptr::null_mut(), bytes, 256, 0);
        if hr < 0 { Err(hr) } else { Ok(va) }
    })
}

/// AcpHalReleaseShapeContexts()
pub unsafe extern "system" fn AcpHalReleaseShapeContexts() -> HRESULT {
    let arrays = std::mem::take(&mut *SHAPE.lock().unwrap());
    for a in arrays.into_iter().filter(|&a| a != 0) {
        ApuFree(a as *mut c_void);
    }
    S_OK
}

// ---------------------------------------------------------------------------------------
// the processor's command / message channel (AcpHalCreate)
// ---------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Message {
    pub kind: u32,
    pub reserved: u32,
    pub value: u64,
    pub id: u64,
}

pub const MSG_AUDIO_FRAME: u32 = 2;
pub const MSG_COMMAND_DONE: u32 = 0x10;
pub const MSG_DISCONNECTED: u32 = 0x80;

/// What the channel holds: queued messages and when the next audio frame is due.
#[derive(Default)]
pub struct Channel {
    messages: VecDeque<Message>,
    connected: bool,
    next_frame: Option<Instant>,
}

impl Channel {
    pub fn connect(&mut self, now: Instant) {
        self.connected = true;
        self.next_frame = Some(now + FRAME);
    }

    pub fn disconnect(&mut self) {
        self.connected = false;
        self.messages.push_back(Message { kind: MSG_DISCONNECTED, ..Default::default() });
    }

    /// A command: done at once (a message for it unless its id is 0 – nobody waits).
    pub fn submit(&mut self, id: u64) -> bool {
        if !self.connected {
            return false;
        }
        if id != 0 {
            self.messages.push_back(Message { kind: MSG_COMMAND_DONE, id, ..Default::default() });
        }
        true
    }

    fn frame_due(&self, now: Instant) -> bool {
        self.connected && self.next_frame.is_some_and(|t| now >= t)
    }

    /// The next message: queued ones first, else an audio frame when one is due (a frame
    /// late by more than 8 is not made up for).
    pub fn pop(&mut self, now: Instant) -> Option<Message> {
        if let Some(m) = self.messages.pop_front() {
            return Some(m);
        }
        if self.frame_due(now) {
            let t = self.next_frame.unwrap_or(now) + FRAME;
            self.next_frame = Some(match now.checked_sub(FRAME * 8) {
                Some(floor) => t.max(floor),
                None => t,
            });
            return Some(Message { kind: MSG_AUDIO_FRAME, ..Default::default() });
        }
        None
    }

    pub fn count(&self, now: Instant) -> u32 {
        self.messages.len() as u32 + self.frame_due(now) as u32
    }
}

#[repr(C)]
struct Hal {
    vtable: *const usize,
    channel: Mutex<Channel>,
}

unsafe extern "system" fn hal_connect(h: *mut Hal, commands: u32, messages: u32) -> HRESULT {
    (*h).channel.lock().unwrap().connect(Instant::now());
    crate::xlog_info!("acphal: audio processor connected ({commands} commands, {messages} messages) – commands are acknowledged, nothing runs");
    S_OK
}

unsafe extern "system" fn hal_disconnect(h: *mut Hal) -> HRESULT {
    (*h).channel.lock().unwrap().disconnect();
    S_OK
}

unsafe extern "system" fn hal_submit(h: *mut Hal, kind: u32, id: u64, frame: u32, data: *const c_void, size: u32) -> HRESULT {
    if trace_command("acphal", kind) {
        let dump = if data.is_null() || size == 0 { "none".to_string() } else { hex(data, (size as usize).min(96)) };
        crate::xlog_info!("apu-trace: acphal command {kind:#x} id {id:#x} frame {frame} data {data:p} size {size}: {dump}");
    }
    static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    {
        let mut s = SEEN.lock().unwrap();
        if !s.contains(&kind) && s.len() < 64 {
            s.push(kind);
            crate::xlog_info!("acphal: audio processor command {kind:#x} first seen");
        }
    }
    if (*h).channel.lock().unwrap().submit(id) {
        S_OK
    } else {
        E_UNEXPECTED
    }
}

unsafe extern "system" fn hal_pop(h: *mut Hal, m: *mut Message) -> bool {
    if m.is_null() {
        return false;
    }
    match (*h).channel.lock().unwrap().pop(Instant::now()) {
        Some(x) => {
            if trace_command("acphal-pop", x.kind) {
                let (kind, value, id) = (x.kind, x.value, x.id);
                crate::xlog_info!("apu-trace: acphal PopMessage -> kind {kind:#x} value {value:#x} id {id:#x}");
            }
            *m = x;
            true
        }
        None => {
            trace_command("acphal-pop", 0xFFFF_FFFF);
            false
        }
    }
}

unsafe extern "system" fn hal_count(h: *mut Hal) -> u32 {
    trace_command("acphal-count", 0);
    (*h).channel.lock().unwrap().count(Instant::now())
}

unsafe extern "system" fn hal_release(h: *mut Hal) {
    drop(Box::from_raw(h));
}

/// the channel object's vtable (made once; function addresses are not constants)
fn hal_vtable() -> *const usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v: Box<[usize; 6]> = Box::new([hal_connect as usize, hal_disconnect as usize, hal_submit as usize, hal_pop as usize, hal_count as usize, hal_release as usize]);
        Box::leak(v).as_ptr() as usize
    }) as *const usize
}

/// AcpHalCreate(&hal)
pub unsafe extern "system" fn AcpHalCreate(out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let h = Box::new(Hal { vtable: hal_vtable(), channel: Mutex::new(Channel::default()) });
    *out = Box::into_raw(h) as *mut c_void;
    S_OK
}

// ---------------------------------------------------------------------------------------
// xapu.dll: the decoder's command queue
// ---------------------------------------------------------------------------------------

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct XApuCommand {
    pub kind: u32,
    pub stream: u32,
    pub sequence: u32,
    pub options: u32,
    pub context: *mut c_void,
}

#[repr(C, packed)]
struct XApuActivate {
    id: XApuCommand,
    channels: u32,
}

#[repr(C, packed)]
struct XApuProcess {
    id: XApuCommand,
    ramp: f32,
    pitch: f32,
    first_frame: u32,
    frames: u32,
    input_bytes: u32,
    input: *mut c_void,
    output_bytes: u32,
    output: *mut c_void,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct XApuResult {
    pub id: XApuCommand,
    pub state: u32,
    pub state_data: u64,
    pub time: u64,
    pub error: HRESULT,
    pub frames: u32,
    pub output_bytes: u32,
    pub output: *mut c_void,
    pub pitch: f32,
    pub input_bytes: u32,
    pub input: *mut c_void,
}

#[repr(C, packed)]
pub struct XApuConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    data_bytes: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C, packed)]
pub struct XApuConnectOutput {
    data: *mut c_void,
    ready: *mut c_void,
}

const XAPU_ACTIVATE: u32 = 1;
const XAPU_DEACTIVATE: u32 = 2;
const XAPU_PROCESS: u32 = 3;

/// What one decoder stream holds between commands.
struct XApuStream {
    /// 1 or 2 (what Activate asked for)
    channels: u32,
    /// the Opus decoder of the stream (made by the first Activate; none: no decoder in this process)
    decoder: Option<opus::Decoder>,
    /// between Activate and Deactivate; the decoder is kept across them and reset by the next Activate
    active: bool,
}

struct XApu {
    results: Mutex<VecDeque<XApuResult>>,
    channels: Mutex<Vec<XApuStream>>,
    data: usize,
    ready: usize,
    frames_per_output: u32,
}

unsafe fn xapu_connect_with<FA, FE>(
    input: *const XApuConnectInput,
    output: *mut XApuConnectOutput,
    handle: *mut *mut c_void,
    alloc_fn: FA,
    event_fn: FE,
) -> HRESULT
where
    FA: FnOnce(usize, usize) -> Option<usize>,
    FE: FnOnce() -> usize,
{
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    if input.is_null() || output.is_null() || handle.is_null() {
        return E_INVALIDARG;
    }
    let i = std::ptr::read_unaligned(input);
    let clear_out = || {
        std::ptr::write_unaligned(output, XApuConnectOutput { data: std::ptr::null_mut(), ready: std::ptr::null_mut() });
        *handle = std::ptr::null_mut();
    };
    let data = if i.data_bytes != 0 {
        let Some(p) = alloc_fn(i.data_bytes as usize, 0x1_0000) else {
            clear_out();
            return E_OUTOFMEMORY;
        };
        p
    } else {
        0
    };
    let ready = if i.options & 8 != 0 {
        let e = event_fn();
        if e == 0 {
            if data != 0 {
                VirtualFree(data as *mut c_void, 0, MEM_RELEASE);
            }
            clear_out();
            return E_OUTOFMEMORY;
        }
        e
    } else {
        0
    };
    let n_streams = i.streams.max(1) as usize;
    let mut channels = Vec::new();
    if channels.try_reserve_exact(n_streams).is_err() {
        if data != 0 {
            VirtualFree(data as *mut c_void, 0, MEM_RELEASE);
        }
        if ready != 0 {
            CloseHandle(ready as _);
        }
        clear_out();
        return E_OUTOFMEMORY;
    }
    channels.resize_with(n_streams, || XApuStream { channels: 2, decoder: None, active: false });
    let x = Box::new(XApu { results: Mutex::new(VecDeque::new()), channels: Mutex::new(channels), data, ready, frames_per_output: i.frames_per_output });
    std::ptr::write_unaligned(output, XApuConnectOutput { data: data as *mut c_void, ready: ready as *mut c_void });
    *handle = Box::into_raw(x) as *mut c_void;
    let (process, streams) = (i.process, i.streams);
    if trace_on() {
        let (data_bytes, per_stream, frames, options) = (i.data_bytes, i.commands_per_stream, i.frames_per_output, i.options);
        crate::xlog_info!("apu-trace: XApuConnect: data {data_bytes:#x} B at {data:#x}, {per_stream} commands per stream, {frames} frames per output, options {options:#x}");
    }
    crate::xlog_info!("xapu: decoder connected (process {process}, {streams} streams) – Opus packets are decoded (silence without an Opus decoder)");
    S_OK
}

/// The Opus decoder of a stream Activate (re)started. None when the process has no decoder (logged once).
fn new_decoder(stream: u32, channels: u32) -> Option<opus::Decoder> {
    static WARNED: std::sync::Once = std::sync::Once::new();
    match opus::Decoder::new(channels as u16, 48_000, &[]) {
        Ok(d) => Some(d),
        Err(e) => {
            WARNED.call_once(|| crate::xlog_warn!("xapu: no Opus decoder ({e}) – the decoder answers with silence"));
            crate::xlog_debug!("xapu: stream {stream} without decoder");
            None
        }
    }
}

/// Decodes a Process command's input (an Opus packet) into `out` (float32 samples, `channels` interleaved):
/// `first_frame` frames of what the decoder produces are skipped, the rest fills `out` (what is not
/// produced stays silent). Returns the frames written (None: nothing was decoded).
unsafe fn decode_into(s: &mut XApuStream, stream: u32, p: &XApuProcess, out: &mut [f32]) -> Option<usize> {
    let (input, input_bytes, first_frame) = (p.input, p.input_bytes, p.first_frame);
    if !s.active {
        return None;
    }
    let dec = s.decoder.as_mut()?;
    if input.is_null() || input_bytes == 0 {
        return None;
    }
    let packet = std::slice::from_raw_parts(input as *const u8, input_bytes as usize);
    let mut pcm: Vec<i16> = Vec::with_capacity(960 * 2);
    let ch = s.channels as usize;
    match dec.decode_packet(packet, &mut pcm) {
        Ok(frames) if frames > 0 => {
            let skip = (first_frame as usize).min(frames);
            let produced = (frames - skip).min(out.len() / ch);
            let mut nonzero = 0u64;
            for (o, &v) in out.iter_mut().zip(pcm[skip * ch..(skip + produced) * ch].iter()) {
                *o = v as f32 / 32768.0;
                nonzero += (v != 0) as u64;
            }
            xaudio2::note_console_decode(1, 0, produced as u64, nonzero);
            Some(produced)
        }
        other => {
            xaudio2::note_console_decode(0, 1, 0, 0);
            if trace_on() {
                crate::xlog_info!("apu-trace: stream {stream}: {input_bytes} B did not decode ({other:?})");
            }
            None
        }
    }
}

/// Every 5 s while the decoder works: what it decoded (the log is what a run that is cut short
/// leaves; report.json is written at the start).
fn report_progress() {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let now = Instant::now();
    {
        let mut l = LAST.lock().unwrap();
        match *l {
            Some(t) if now.duration_since(t) < Duration::from_secs(5) => return,
            _ => *l = Some(now),
        }
    }
    let a = xaudio2::stats();
    crate::xlog_info!(
        "xapu: {} Opus packets decoded ({} errors), {} frames, {} non-zero samples",
        a.xapu_packets_decoded, a.xapu_decode_errors, a.xapu_frames, a.xapu_nonzero_samples
    );
}

/// XApuConnect(&input, &output, &handle)
pub unsafe extern "system" fn XApuConnect(input: *const XApuConnectInput, output: *mut XApuConnectOutput, handle: *mut *mut c_void) -> HRESULT {
    use windows_sys::Win32::System::Threading::CreateEventW;
    xapu_connect_with(input, output, handle, |sz, al| alloc_aligned(sz, al), || {
        CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) as usize
    })
}

/// XApuDisconnect(handle)
pub unsafe extern "system" fn XApuDisconnect(handle: *mut c_void) -> HRESULT {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    if handle.is_null() {
        return E_INVALIDARG;
    }
    let x = Box::from_raw(handle as *mut XApu);
    if x.data != 0 {
        VirtualFree(x.data as *mut c_void, 0, MEM_RELEASE);
    }
    if x.ready != 0 {
        CloseHandle(x.ready as _);
    }
    S_OK
}

/// XApuEnqueueCommand(handle, &command, &results waiting): done at once – a Process command
/// decodes its Opus packet into the output buffer (silence of the length asked for where there is
/// nothing to decode)
pub unsafe extern "system" fn XApuEnqueueCommand(handle: *mut c_void, command: *const XApuCommand, count: *mut u32) -> HRESULT {
    use windows_sys::Win32::System::Threading::SetEvent;
    if handle.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    let x = &*(handle as *const XApu);
    let c = std::ptr::read_unaligned(command);
    if trace_command("xapu", c.kind) {
        let (kind, stream, seq, opts, ctx) = (c.kind, c.stream, c.sequence, c.options, c.context);
        crate::xlog_info!("apu-trace: xapu command {kind:#x} stream {stream} seq {seq} options {opts:#x} context {ctx:p}: {}", hex(command as *const c_void, command_len(kind)));
        trace_callers(kind);
        if kind == XAPU_PROCESS {
            let p = std::ptr::read_unaligned(command as *const XApuProcess);
            let (ramp, pitch, first, frames, ib, input, ob, output) = (p.ramp, p.pitch, p.first_frame, p.frames, p.input_bytes, p.input, p.output_bytes, p.output);
            crate::xlog_info!(
                "apu-trace:   process ramp {ramp} pitch {pitch} first_frame {first} frames {frames} input {ib} B at {input:p} output {ob} B at {output:p}; input starts {}",
                hex(input, (ib as usize).min(512))
            );
        }
    }
    let mut r = XApuResult { id: c, state: 0, state_data: 0, time: 0, error: S_OK, frames: 0, output_bytes: 0, output: std::ptr::null_mut(), pitch: 1.0, input_bytes: 0, input: std::ptr::null_mut() };
    {
        let mut ch = x.channels.lock().unwrap();
        // a stream the connection does not have: no decoder, no state; a Process answers with silence
        // of the length asked for, with the channel count of the first stream (if any; else stereo)
        let known = (c.stream as usize) < ch.len();
        if c.kind == XAPU_ACTIVATE {
            let n = std::ptr::read_unaligned(command as *const XApuActivate).channels;
            let channels = if n == 1 { 1 } else { 2 };
            if let Some(s) = ch.get_mut(c.stream as usize) {
                match s.decoder.as_mut() {
                    // the stream started again with the same layout: its decoder is reset, not
                    // rebuilt (on Windows a new decoder is a new Media Foundation transform)
                    Some(d) if s.channels == channels => d.flush(),
                    _ => s.decoder = new_decoder(c.stream, channels),
                }
                s.channels = channels;
                s.active = true;
            }
        } else if c.kind == XAPU_DEACTIVATE {
            if let Some(s) = ch.get_mut(c.stream as usize) {
                s.active = false;
            }
        } else if c.kind == XAPU_PROCESS {
            let p = std::ptr::read_unaligned(command as *const XApuProcess);
            let channels = if known { ch[c.stream as usize].channels } else { ch.first().map_or(2, |s| s.channels) };
            let frame_bytes = channels * 4;
            let asked = if p.frames != 0 { p.frames } else { x.frames_per_output };
            let mut frames = (if asked != 0 { asked } else { 960 }).min(p.output_bytes / frame_bytes);
            if !p.output.is_null() && frames != 0 {
                // decoded here, then copied byte by byte: the title's buffer is not known to be 4-byte aligned
                let mut out = vec![0.0f32; (frames * channels) as usize];
                // decoded: the frames after first_frame; not decoded: silence of the length asked for
                if known {
                    if let Some(n) = decode_into(&mut ch[c.stream as usize], c.stream, &p, &mut out) {
                        frames = n as u32;
                    }
                }
                std::ptr::copy_nonoverlapping(out.as_ptr() as *const u8, p.output as *mut u8, out.len() * 4);
            }
            // output ready, more input wanted
            r.state = 1 | 2;
            r.frames = frames;
            r.output_bytes = frames * frame_bytes;
            r.output = p.output;
            r.pitch = p.pitch;
            r.input_bytes = p.input_bytes;
            r.input = p.input;
            report_progress();
        }
    }
    let n = {
        let mut q = x.results.lock().unwrap();
        q.push_back(r);
        q.len() as u32
    };
    if !count.is_null() {
        *count = n;
    }
    if x.ready != 0 && c.options & 1 != 0 {
        SetEvent(x.ready as _);
    }
    S_OK
}

/// XApuDequeueResult(handle, &result)
pub unsafe extern "system" fn XApuDequeueResult(handle: *mut c_void, result: *mut XApuResult) -> HRESULT {
    if handle.is_null() || result.is_null() {
        return E_INVALIDARG;
    }
    trace_command("xapu-dequeue", 0);
    match (*(handle as *const XApu)).results.lock().unwrap().pop_front() {
        Some(r) => {
            std::ptr::write_unaligned(result, r);
            S_OK
        }
        None => XAPU_NO_RESULTS,
    }
}

/// XApuGetResultCount(handle, &count) -> count
pub unsafe extern "system" fn XApuGetResultCount(handle: *mut c_void, total: *mut u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let n = (*(handle as *const XApu)).results.lock().unwrap().len() as u32;
    if !total.is_null() {
        *total = n;
    }
    n
}

/// XApuGetUtilization(…, &utilization): streams available and the most there can be
pub unsafe extern "system" fn XApuGetUtilization(_a: u32, _b: u32, u: *mut u32) -> HRESULT {
    if u.is_null() {
        return E_INVALIDARG;
    }
    *u = 256;
    *u.add(1) = 256;
    S_OK
}

// ---------------------------------------------------------------------------------------
// the DSP (chad.dll): connected, with nothing to run
// ---------------------------------------------------------------------------------------

/// a DSP connection (what the handle points at)
struct XDsp {
    streams: u32,
}

/// XDspConnectWithMaximumStreamLimit(a, b, c, max streams, flags, &handle)
pub unsafe extern "system" fn XDspConnectWithMaximumStreamLimit(_a: usize, _b: u32, _c: u32, streams: u32, _flags: u32, handle: *mut *mut c_void) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    *handle = Box::into_raw(Box::new(XDsp { streams })) as *mut c_void;
    crate::xlog_info!("acphal: DSP connected ({streams} streams) – its commands are taken, nothing runs");
    S_OK
}

pub unsafe extern "system" fn XDspDisconnect(handle: *mut c_void) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    drop(Box::from_raw(handle as *mut XDsp));
    S_OK
}

pub unsafe extern "system" fn XDspActivate(handle: *mut c_void, a: usize, b: usize, c: usize) -> HRESULT {
    if trace_command("dsp", 0xA0) {
        crate::xlog_info!("apu-trace: XDspActivate({a:#x}, {b:#x}, {c:#x})");
    }
    if handle.is_null() {
        E_INVALIDARG
    } else {
        S_OK
    }
}

pub unsafe extern "system" fn XDspDeactivate(handle: *mut c_void, _a: usize, _b: usize, _c: usize) -> HRESULT {
    if handle.is_null() {
        E_INVALIDARG
    } else {
        S_OK
    }
}

pub unsafe extern "system" fn XDspSubmitCommand(handle: *mut c_void, a: usize, b: usize, c: usize) -> HRESULT {
    static FIRST: std::sync::Once = std::sync::Once::new();
    if trace_command("dsp", 0x5C) {
        crate::xlog_info!("apu-trace: XDspSubmitCommand({a:#x}, {b:#x}, {c:#x})");
    }
    if handle.is_null() {
        return E_INVALIDARG;
    }
    FIRST.call_once(|| crate::xlog_info!("acphal: first DSP command ({} streams) – taken, not run", (*(handle as *const XDsp)).streams));
    S_OK
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        XDspConnectWithMaximumStreamLimit,
        XDspDisconnect,
        XDspActivate,
        XDspDeactivate,
        XDspSubmitCommand,
        ApuCreateHeap,
        ApuAlloc,
        ApuFree,
        ApuMapVirtualAddress,
        ApuMapApuAddress,
        ApuIsVirtualAddressValid,
        ApuHeapGetState,
        AcpHalCreate,
        AcpHalAllocateShapeContexts,
        AcpHalReleaseShapeContexts,
        XApuConnect,
        XApuDisconnect,
        XApuEnqueueCommand,
        XApuDequeueResult,
        XApuGetResultCount,
        XApuGetUtilization,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dsp_connects_and_takes_commands() {
        unsafe {
            let mut h = std::ptr::null_mut();
            assert_eq!(XDspConnectWithMaximumStreamLimit(0, 0, 0, 0x80, 1, &mut h), S_OK);
            assert!(!h.is_null());
            assert_eq!(XDspActivate(h, 0, 0, 0), S_OK);
            assert_eq!(XDspSubmitCommand(h, 1, 2, 3), S_OK);
            assert_eq!(XDspDeactivate(h, 0, 0, 0), S_OK);
            assert_eq!(XDspDisconnect(h), S_OK);
            assert_eq!(XDspSubmitCommand(std::ptr::null_mut(), 0, 0, 0), E_INVALIDARG);
        }
    }

    #[test]
    fn apu_memory_maps_both_ways() {
        unsafe {
            let (mut va, mut apu) = (std::ptr::null_mut(), 0u32);
            assert_eq!(ApuAlloc(&mut va, &mut apu, 0x3000, 0x2_0000, 0), S_OK);
            assert_eq!(va as usize % 0x2_0000, 0, "alignment above 64 KiB");
            assert_eq!(ApuMapApuAddress(apu + 0x10) as usize, va as usize + 0x10);
            let mut again = 0u32;
            assert_eq!(ApuMapVirtualAddress(&mut again, (va as usize + 0x20) as *mut c_void, 4), S_OK);
            assert_eq!(again, apu + 0x20);
            assert_eq!(ApuFree(va), S_OK);
            assert!(ApuMapApuAddress(apu).is_null());
        }
    }

    #[test]
    fn channel_acknowledges_and_ticks() {
        let t0 = Instant::now();
        let mut c = Channel::default();
        assert!(!c.submit(7), "not connected yet");
        c.connect(t0);
        assert!(c.submit(7));
        assert!(c.submit(0), "id 0: no message");
        assert_eq!(c.count(t0), 1);
        assert_eq!(c.pop(t0), Some(Message { kind: MSG_COMMAND_DONE, id: 7, ..Default::default() }));
        assert_eq!(c.pop(t0), None, "no frame due yet");
        let later = t0 + FRAME * 3;
        assert_eq!(c.pop(later).map(|m| m.kind), Some(MSG_AUDIO_FRAME));
        c.disconnect();
        assert_eq!(c.pop(later).map(|m| m.kind), Some(MSG_DISCONNECTED));
    }

    #[test]
    fn decoder_returns_silence() {
        unsafe {
            let input = XApuConnectInput { process: 1, streams: 2, commands_per_stream: 4, data_bytes: 0, frames_per_output: 256, options: 0 };
            let mut out = XApuConnectOutput { data: std::ptr::null_mut(), ready: std::ptr::null_mut() };
            let mut h = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut out, &mut h), S_OK);
            let mut buf = vec![0xAAu8; 4096];
            let p = XApuProcess {
                id: XApuCommand { kind: XAPU_PROCESS, stream: 1, sequence: 1, options: 0, context: std::ptr::null_mut() },
                ramp: 0.0,
                pitch: 1.0,
                first_frame: 0,
                frames: 0,
                input_bytes: 0,
                input: std::ptr::null_mut(),
                output_bytes: buf.len() as u32,
                output: buf.as_mut_ptr() as *mut c_void,
            };
            let mut n = 0;
            assert_eq!(XApuEnqueueCommand(h, &p as *const _ as *const XApuCommand, &mut n), S_OK);
            assert_eq!(n, 1);
            assert_eq!(XApuGetResultCount(h, std::ptr::null_mut()), 1);
            let mut r: XApuResult = std::mem::zeroed();
            assert_eq!(XApuDequeueResult(h, &mut r), S_OK);
            let frames = r.frames;
            assert_eq!(frames, 256, "frames_per_output, stereo floats");
            assert!(buf[..256 * 8].iter().all(|&b| b == 0) && buf[256 * 8] == 0xAA);
            assert_eq!(XApuDequeueResult(h, &mut r), XAPU_NO_RESULTS);
            assert_eq!(XApuDisconnect(h), S_OK);
        }
    }

    #[test]
    fn apu_address_prevents_overflow_overlap_and_rejects_null_va() {
        // u32::MAX cannot fit within [APU_BASE, 4 GiB) without wrapping.
        let mut apu = Apu { regions: Vec::new(), next: 0x1000_0000 };
        assert_eq!(apu.address(0x1_0000_0000, u32::MAX, false), None);

        // Near the 4 GiB boundary: a region that would cross u32::MAX wraps to APU_BASE without
        // overlapping an existing live region at APU_BASE.
        let first = apu.address(0x2_0000_0000, 0x2000, false).unwrap();
        assert_eq!(first, 0x1000_0000);
        apu.next = 0xFFFF_F000;
        let second = apu.address(0x3_0000_0000, 0x2000, false).unwrap();
        assert_eq!(second, 0x1000_2000, "wrapped range must skip past live region at APU_BASE");

        // Owned allocation must register its own entry even if an unowned mapping covers the VA.
        let owned = apu.address(0x2_0000_0000, 0x1000, true).unwrap();
        assert_ne!(owned, first);
        assert!(apu.regions.iter().any(|r| r.apu == owned && r.owned));

        unsafe {
            let mut out = 0xDEAD_BEEFu32;
            assert_eq!(ApuMapVirtualAddress(&mut out, std::ptr::null_mut(), 16), E_INVALIDARG);
            assert_eq!(out, 0xDEAD_BEEF);
        }
    }

    #[test]
    fn shape_contexts_overflow_and_partial_failure_roll_back() {
        unsafe {
            // Overflow on counts[1] * CONTEXT_BYTES fails cleanly without allocating slot 0.
            let mut c = ShapeContexts {
                counts: [4, u32::MAX / CONTEXT_BYTES + 1, 0, 0, 0],
                reserved: 0,
                contexts: [0x1234 as *mut c_void; 5],
            };
            assert_eq!(AcpHalAllocateShapeContexts(&mut c), E_OUTOFMEMORY);
            assert!(c.contexts.iter().all(|p| p.is_null()));

            // Partial failure after slot 0 allocated via real ApuAlloc rolls back slot 0.
            let mut first_apu = 0u32;
            let mut c2 = ShapeContexts {
                counts: [4, 4, 0, 0, 0],
                reserved: 0,
                contexts: [0x1234 as *mut c_void; 5],
            };
            let hr = allocate_shape_contexts_with(&mut c2, |idx, bytes| {
                if idx == 0 {
                    let mut va = std::ptr::null_mut();
                    let hr = ApuAlloc(&mut va, &mut first_apu, bytes, 256, 0);
                    assert_eq!(hr, S_OK);
                    Ok(va)
                } else {
                    Err(E_OUTOFMEMORY)
                }
            });
            assert_eq!(hr, E_OUTOFMEMORY);
            assert!(c2.contexts.iter().all(|p| p.is_null()));
            assert!(ApuMapApuAddress(first_apu).is_null(), "slot 0 must be freed on rollback");

            // Success followed by a failed replacement preserves the live allocation;
            // a subsequent successful replacement frees the prior allocation cleanly.
            let mut c3 = ShapeContexts { counts: [2, 0, 0, 0, 0], reserved: 0, contexts: [std::ptr::null_mut(); 5] };
            assert_eq!(AcpHalAllocateShapeContexts(&mut c3), S_OK);
            let old_va = c3.contexts[0];
            assert!(!old_va.is_null());
            let mut old_apu = 0u32;
            assert_eq!(ApuMapVirtualAddress(&mut old_apu, old_va, 128), S_OK);

            let mut failed_replace = ShapeContexts { counts: [4, 0, 0, 0, 0], reserved: 0, contexts: [std::ptr::null_mut(); 5] };
            assert_eq!(allocate_shape_contexts_with(&mut failed_replace, |_, _| Err(E_OUTOFMEMORY)), E_OUTOFMEMORY);
            assert_eq!(ApuMapApuAddress(old_apu), old_va, "failed replacement must preserve prior live allocation");

            let mut c4 = ShapeContexts { counts: [3, 0, 0, 0, 0], reserved: 0, contexts: [std::ptr::null_mut(); 5] };
            assert_eq!(AcpHalAllocateShapeContexts(&mut c4), S_OK);
            assert!(!c4.contexts[0].is_null());
            assert!(ApuMapApuAddress(old_apu).is_null(), "prior SHAPE context allocation must be freed on replace");
            assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
        }
    }

    #[test]
    fn xapu_connect_returns_oom_and_cleans_up_on_failure() {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_FREE};
        unsafe {
            let input = XApuConnectInput { process: 1, streams: 2, commands_per_stream: 4, data_bytes: 0x1000, frames_per_output: 256, options: 8 };
            let sentinel = 0x1234 as *mut c_void;

            // Data allocation failure returns E_OUTOFMEMORY and clears output + handle.
            let mut out = XApuConnectOutput { data: sentinel, ready: sentinel };
            let mut h = sentinel;
            let hr = xapu_connect_with(&input, &mut out, &mut h, |_, _| None, || unreachable!());
            assert_eq!(hr, E_OUTOFMEMORY);
            assert!(h.is_null());
            assert!({ out.data }.is_null() && { out.ready }.is_null());

            // Event creation failure frees the allocated data buffer and returns E_OUTOFMEMORY.
            let mut out2 = XApuConnectOutput { data: sentinel, ready: sentinel };
            let mut h2 = sentinel;
            let mut allocated_ptr = 0usize;
            let mut event_attempted = false;
            let hr2 = xapu_connect_with(
                &input,
                &mut out2,
                &mut h2,
                |sz, al| {
                    let p = alloc_aligned(sz, al);
                    allocated_ptr = p.unwrap_or(0);
                    p
                },
                || {
                    event_attempted = true;
                    0
                },
            );
            assert_eq!(hr2, E_OUTOFMEMORY);
            assert!(allocated_ptr != 0 && event_attempted);
            assert!(h2.is_null());
            assert!({ out2.data }.is_null() && { out2.ready }.is_null());
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(allocated_ptr as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()), 0);
            assert_eq!(mbi.State, MEM_FREE, "data buffer must be released when event creation fails");
        }
    }
}
