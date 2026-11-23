//! `xcrdapi.dll` – the console's DirectStorage: `DStorageGetFactory` → IDStorageFactoryX.
//!
//! The interfaces are the PC DirectStorage ones in the console's vtable order and structure
//! layouts (as observed on Fortnite, which reads all its data through them):
//! * factory: QueryInterface, AddRef, Release, CreateQueue(desc, riid, out), OpenFile(path,
//!   riid, out), CreateStatusArray(capacity, name, riid, out), SetCpuAffinity(mask),
//!   SetDebugFlags(flags), SetStagingBufferSize(bytes), then 7 more,
//! * queue desc (0x18): source type u32 @0 (0 file, 1 memory), capacity u16 @8, priority i8
//!   @10 (-1 … 2; memory queues 2), name @0x10,
//! * request (0x48): options u64 @0 (`DSTORAGE_REQUEST_OPTIONS`: bit 0 `ZlibDecompress`,
//!   bits 1–4 `BcpackMode`, bits 5–20 `SwizzleMode`, bit 21 `DestinationIsPhysicalPages`,
//!   bit 22 `SourceType` (0 file / 1 memory), bit 23 `SourceIsPhysicalPages`),
//!   destination @8 (`void*` virtual/UMA address or `PULONG_PTR DestinationPageArray`),
//!   `DestinationPageOffset` u16 @0x10, destination size u32 @0x18,
//!   source @0x20 (`IDStorageFileX*`, `const void*`, or `PULONG_PTR SourcePageArray`),
//!   file offset u64 / `SourcePageOffset` u16 @0x28, source size u32 @0x30,
//!   intermediate size u32 @0x34, cancellation tag @0x38, name @0x40,
//! * queue: QueryInterface, AddRef, Release, EnqueueRequest, EnqueueStatus(array, index),
//!   EnqueueSignal(fence, value), Submit, CancelRequestsWithTag(mask, value), Close,
//!   GetErrorEvent, RetrieveErrorRecord (0x290 bytes), Query (0x1C: the desc, then empty
//!   slots u16 @0x18 and requests until the automatic submit u16 @0x1A), then 6 more,
//! * file: QueryInterface, AddRef, Release, Close, GetHandle, 2 unknown, GetId, then 4 more;
//!   status array: QueryInterface, AddRef, Release, IsComplete(index), GetHResult(index),
//!   then 5 more.
//!
//! Each queue has a worker thread: submitted requests run in order (file reads through the
//! VFS like the title's own, memory copies, zlib), a status entry gets the first failure of
//! the requests since the previous one, a signal sets the fence from the CPU. Methods not
//! listed are logged and answer E_NOTIMPL; BCPack and other console formats are refused.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::os::windows::fs::FileExt;
use std::os::windows::io::AsRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

use windows_sys::core::GUID;
use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE};
use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess, ResetEvent, SetEvent};

use crate::util::{HRESULT, S_OK};

const DLL: &str = "xcrdapi.dll";

const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002u32 as i32;
const E_POINTER: HRESULT = 0x8000_4003u32 as i32;
const E_PENDING: HRESULT = 0x8000_000Au32 as i32;
const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
const E_ABORT: HRESULT = 0x8000_4004u32 as i32;
const E_QUEUE_CAPACITY: HRESULT = 0x8924_0003u32 as i32;
const E_END_OF_FILE: HRESULT = 0x8924_0007u32 as i32;
const E_ACCESS_VIOLATION: HRESULT = 0x8924_0009u32 as i32;
const E_DESTINATION_SIZE: HRESULT = 0x8924_000Fu32 as i32;
const E_QUEUE_CLOSED: HRESULT = 0x8924_0010u32 as i32;
const E_QUEUE_PRIORITY: HRESULT = 0x8924_0013u32 as i32;
const E_FILE_HANDLE: HRESULT = 0x8924_0017u32 as i32;
const E_SOURCE_TYPE: HRESULT = 0x8924_001Bu32 as i32;
const E_DECOMPRESSION: HRESULT = 0x8924_0030u32 as i32;
const E_ZLIB_DATA: HRESULT = 0x8924_0032u32 as i32;
const E_QUEUE_TYPE: HRESULT = 0x8924_00F5u32 as i32;

const IID_IUNKNOWN: GUID = GUID::from_u128(0x00000000_0000_0000_c000_000000000046);
const IID_FACTORY: GUID = GUID::from_u128(0xfb3b54ac_bf82_463a_b777_73195837864d);
const IID_FILE: GUID = GUID::from_u128(0x43704f9f_27c6_4a9f_8815_d16527894df4);
const IID_QUEUE: GUID = GUID::from_u128(0xa3f6649a_8c2c_4b3a_8b6e_10186a04b9bc);
/// The factory and queue of a newer SDK (a retail game): the same first methods; the factory's slot 10
/// creates a queue from a 0x20-byte desc (a u64 @0x18 more) with `IID_QUEUE_X1`.
const IID_FACTORY_X1: GUID = GUID::from_u128(0x62959d11_d45f_4884_a0a3_f7862ea243ae);
const IID_QUEUE_X1: GUID = GUID::from_u128(0x3c02cb0c_c050_4195_99a1_a1ce94ba33c5);
const IID_STATUS_ARRAY: GUID = GUID::from_u128(0xcd66d752_2786_44f7_9070_e2a6fdad3f1b);

/// Console `DSTORAGE_REQUEST_OPTIONS` bitfields (Xbox Series X|S `dstorage_xs.h`):
/// - bit  0      : `ZlibDecompress` (1 bit: RFC 1950 zlib)
/// - bits 1..=4  : `BcpackMode` (4 bits: 0 = none, 1..=7 = BC1..BC7)
/// - bits 5..=20 : `SwizzleMode` (16 bits: 0 = none, 1..=4 = hardware texture swizzle)
/// - bit  21     : `DestinationIsPhysicalPages` (1 bit)
/// - bit  22     : `SourceType` (1 bit: 0 = file, 1 = memory)
/// - bit  23     : `SourceIsPhysicalPages` (1 bit)
/// - bits 24..=63: `Reserved` (40 bits)
///
/// Unlike PC DirectStorage (`dstorage.h`), Xbox Series X|S is UMA: placed D3D12 buffers and
/// textures live at virtual addresses in UMA or graphics reservations (tracked by
/// `MEM_WRITE_WATCH`), or in 64 KiB physical page arrays from `AllocateTitlePhysicalPages`.
const ZLIB_BIT: u64 = 1 << 0;
const BCPACK_SHIFT: u32 = 1;
const BCPACK_MASK: u64 = 0xF << BCPACK_SHIFT;
const SWIZZLE_SHIFT: u32 = 5;
const SWIZZLE_MASK: u64 = 0xFFFF << SWIZZLE_SHIFT;
const DEST_PHYS_PAGES_BIT: u64 = 1 << 21;
const SOURCE_MEMORY_BIT: u64 = 1 << 22;
const SOURCE_PHYS_PAGES_BIT: u64 = 1 << 23;
const KNOWN_OPTIONS_MASK: u64 = (1u64 << 24) - 1;
const REQUEST_BYTES: usize = 0x48;
const ERROR_RECORD_BYTES: usize = 0x290;
const NO_COMMAND: u32 = 0xFFFF_FFFF;
const COMMAND_REQUEST: u32 = 0;
/// ID3D12Fence::Signal
const FENCE_SIGNAL: usize = 10;

#[path = "dstorage_capture.rs"]
mod bcpack_capture;

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![DStorageGetFactory]
}

fn same(a: &GUID, b: &GUID) -> bool {
    (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4)
}

fn warn_once(what: &str, why: std::fmt::Arguments) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut s = SEEN.lock().unwrap();
    if s.len() < 64 && !s.iter().any(|x| x == what) {
        s.push(what.to_string());
        crate::xlog_warn!("DirectStorage: {what}: {why}");
    }
}

/// A vtable: the listed methods, then logged E_NOTIMPL stubs up to `total` entries.
fn vtable(iface: &str, methods: &[usize], total: usize) -> &'static [usize] {
    let mut v = methods.to_vec();
    for i in methods.len()..total {
        let (stub, _) = crate::stubs::make_stub_info(DLL, &format!("{iface}::#{i}"), E_NOTIMPL as u32 as u64);
        v.push(stub);
    }
    Box::leak(v.into_boxed_slice())
}

/// COM object header: vtable, reference count, the shared part.
#[repr(C)]
struct Object<T> {
    vtbl: *const usize,
    refs: AtomicU32,
    inner: Arc<T>,
}

unsafe fn new_object<T>(vtbl: &'static [usize], inner: Arc<T>) -> *mut c_void {
    Box::into_raw(Box::new(Object { vtbl: vtbl.as_ptr(), refs: AtomicU32::new(1), inner })) as *mut c_void
}

unsafe fn inner<'a, T>(p: *mut c_void) -> &'a Arc<T> {
    &(*(p as *const Object<T>)).inner
}

unsafe fn add_ref<T>(p: *mut c_void) -> u32 {
    (*(p as *const Object<T>)).refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe fn release<T>(p: *mut c_void) -> u32 {
    let n = (*(p as *const Object<T>)).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        drop(Box::from_raw(p as *mut Object<T>));
    }
    n
}

unsafe fn no_ref(_p: *mut c_void) -> u32 {
    1
}

unsafe fn query(p: *mut c_void, own: &GUID, riid: *const GUID, out: *mut *mut c_void, add: unsafe fn(*mut c_void) -> u32) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    if !riid.is_null() && (same(&*riid, own) || same(&*riid, &IID_IUNKNOWN)) {
        add(p);
        *out = p;
        return S_OK;
    }
    *out = std::ptr::null_mut();
    E_NOINTERFACE
}

// ---------------------------------------------------------------------------------------
// factory
// ---------------------------------------------------------------------------------------

/// `HRESULT DStorageGetFactory(REFIID riid, void** factory)`
pub unsafe extern "system" fn DStorageGetFactory(riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static FACTORY: OnceLock<usize> = OnceLock::new();
    let f = *FACTORY.get_or_init(|| {
        let v = vtable(
            "IDStorageFactoryX",
            &[
                factory_qi as usize,
                static_ref as usize,
                static_ref as usize,
                factory_create_queue as usize,
                factory_open_file as usize,
                factory_create_status_array as usize,
                factory_set_cpu_affinity as usize,
                factory_set_debug_flags as usize,
                factory_set_staging_buffer_size as usize,
                factory_unknown as usize,
                factory_create_queue1 as usize,
            ],
            16,
        );
        Box::into_raw(Box::new(v.as_ptr())) as usize
    });
    factory_qi(f as *mut c_void, riid, out)
}

unsafe extern "system" fn static_ref(_p: *mut c_void) -> u32 {
    1
}

unsafe extern "system" fn factory_qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if !riid.is_null() && same(&*riid, &IID_FACTORY_X1) {
        return query(p, &IID_FACTORY_X1, riid, out, no_ref);
    }
    query(p, &IID_FACTORY, riid, out, no_ref)
}

unsafe extern "system" fn factory_unknown(_p: *mut c_void) -> HRESULT {
    warn_once("IDStorageFactoryX::#9", format_args!("called – answering E_NOTIMPL"));
    E_NOTIMPL
}

/// Slot 10 of the newer factory: CreateQueue with a 0x20-byte desc – the 0x18 of
/// `factory_create_queue`, then a u64 (a retail game: 1).
unsafe extern "system" fn factory_create_queue1(p: *mut c_void, desc: *const u8, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if !desc.is_null() {
        let extra = (desc.add(0x18) as *const u64).read_unaligned();
        static SEEN: AtomicBool = AtomicBool::new(false);
        if !SEEN.swap(true, Ordering::Relaxed) {
            crate::xlog_info!("DirectStorage: CreateQueue (0x20 desc), u64 @0x18 = {extra:#x}");
        }
    }
    factory_create_queue(p, desc, riid, out)
}

unsafe extern "system" fn factory_set_cpu_affinity(_p: *mut c_void, _mask: u64) -> HRESULT {
    S_OK
}

unsafe extern "system" fn factory_set_debug_flags(_p: *mut c_void, _flags: u32) {}

unsafe extern "system" fn factory_set_staging_buffer_size(_p: *mut c_void, _bytes: u32) -> HRESULT {
    S_OK
}

// ---------------------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------------------

struct FileInner {
    id: u32,
    path: Vec<u16>,
    file: Mutex<Option<Arc<std::fs::File>>>,
}

impl FileInner {
    fn handle(&self) -> Option<Arc<std::fs::File>> {
        self.file.lock().unwrap().clone()
    }
}

fn file_vtbl() -> &'static [usize] {
    static V: OnceLock<&'static [usize]> = OnceLock::new();
    V.get_or_init(|| {
        vtable(
            "IDStorageFileX",
            &[file_qi as usize, file_add_ref as usize, file_release as usize, file_close as usize, file_get_handle as usize, file_unknown as usize, file_unknown as usize, file_get_id as usize],
            12,
        )
    })
}

unsafe extern "system" fn factory_open_file(_p: *mut c_void, path: *const u16, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static NEXT_ID: AtomicU32 = AtomicU32::new(1);
    if out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    let Some(name) = crate::util::from_wide_ptr(path) else { return E_INVALIDARG };
    // through the VFS hooks, as the title's own CreateFileW
    let file = match std::fs::File::open(&name) {
        Ok(f) => f,
        Err(e) => {
            crate::xlog_warn!("DirectStorage: OpenFile '{name}': {e}");
            return e.raw_os_error().map_or(E_INVALIDARG, |c| (0x8007_0000u32 | (c as u32 & 0xFFFF)) as i32);
        }
    };
    let inner = Arc::new(FileInner { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), path: name.encode_utf16().collect(), file: Mutex::new(Some(Arc::new(file))) });
    let o = new_object(file_vtbl(), inner);
    let hr = file_qi(o, riid, out);
    file_release(o);
    hr
}

unsafe extern "system" fn file_qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    query(p, &IID_FILE, riid, out, add_ref::<FileInner>)
}

unsafe extern "system" fn file_add_ref(p: *mut c_void) -> u32 {
    add_ref::<FileInner>(p)
}

unsafe extern "system" fn file_release(p: *mut c_void) -> u32 {
    release::<FileInner>(p)
}

unsafe extern "system" fn file_close(p: *mut c_void) {
    inner::<FileInner>(p).file.lock().unwrap().take();
}

/// a handle of the title's own (it closes it)
unsafe extern "system" fn file_get_handle(p: *mut c_void) -> HANDLE {
    let Some(f) = inner::<FileInner>(p).handle() else { return std::ptr::null_mut() };
    let mut h: HANDLE = std::ptr::null_mut();
    let me = GetCurrentProcess();
    if DuplicateHandle(me, f.as_raw_handle() as HANDLE, me, &mut h, 0, 0, DUPLICATE_SAME_ACCESS) == 0 {
        return std::ptr::null_mut();
    }
    h
}

unsafe extern "system" fn file_unknown(_p: *mut c_void) -> HRESULT {
    warn_once("IDStorageFileX #5 / #6", format_args!("not modelled"));
    E_NOTIMPL
}

unsafe extern "system" fn file_get_id(p: *mut c_void) -> u32 {
    let f = inner::<FileInner>(p);
    if f.handle().is_some() {
        f.id
    } else {
        u32::MAX
    }
}

// ---------------------------------------------------------------------------------------
// status arrays
// ---------------------------------------------------------------------------------------

struct StatusInner {
    slots: Box<[AtomicI32]>,
}

fn status_vtbl() -> &'static [usize] {
    static V: OnceLock<&'static [usize]> = OnceLock::new();
    V.get_or_init(|| {
        vtable(
            "IDStorageStatusArrayX",
            &[status_qi as usize, status_add_ref as usize, status_release as usize, status_is_complete as usize, status_get_hresult as usize],
            10,
        )
    })
}

unsafe extern "system" fn factory_create_status_array(_p: *mut c_void, capacity: u32, _name: *const u8, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    if capacity == 0 || capacity > 1 << 20 {
        return E_INVALIDARG;
    }
    let inner = Arc::new(StatusInner { slots: (0..capacity).map(|_| AtomicI32::new(E_PENDING)).collect() });
    let o = new_object(status_vtbl(), inner);
    let hr = status_qi(o, riid, out);
    status_release(o);
    hr
}

unsafe extern "system" fn status_qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    query(p, &IID_STATUS_ARRAY, riid, out, add_ref::<StatusInner>)
}

unsafe extern "system" fn status_add_ref(p: *mut c_void) -> u32 {
    add_ref::<StatusInner>(p)
}

unsafe extern "system" fn status_release(p: *mut c_void) -> u32 {
    release::<StatusInner>(p)
}

unsafe extern "system" fn status_is_complete(p: *mut c_void, index: u32) -> bool {
    inner::<StatusInner>(p).slots.get(index as usize).is_some_and(|s| s.load(Ordering::Acquire) != E_PENDING)
}

unsafe extern "system" fn status_get_hresult(p: *mut c_void, index: u32) -> HRESULT {
    inner::<StatusInner>(p).slots.get(index as usize).map_or(E_INVALIDARG, |s| s.load(Ordering::Acquire))
}

// ---------------------------------------------------------------------------------------
// requests
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Request {
    options: u64,
    destination: u64,
    destination_page_offset: u16,
    destination_size: u32,
    source: u64,
    offset: u64,
    source_page_offset: u16,
    source_size: u32,
    intermediate_size: u32,
    tag: u64,
}

impl Request {
    fn parse(b: &[u8; REQUEST_BYTES]) -> Request {
        let q = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let d = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let w = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap());
        let options = q(0);
        let memory = options & SOURCE_MEMORY_BIT != 0;
        let dest_phys = options & DEST_PHYS_PAGES_BIT != 0;
        let src_phys = memory && (options & SOURCE_PHYS_PAGES_BIT != 0);
        Request {
            options,
            destination: q(8),
            destination_page_offset: if dest_phys { w(0x10) } else { 0 },
            destination_size: d(0x18),
            source: q(0x20),
            offset: if memory { 0 } else { q(0x28) },
            source_page_offset: if src_phys { w(0x28) } else { 0 },
            source_size: d(0x30),
            intermediate_size: d(0x34),
            tag: q(0x38),
        }
    }

    fn zlib(&self) -> bool {
        self.options & ZLIB_BIT != 0
    }

    fn bcpack(&self) -> u8 {
        ((self.options & BCPACK_MASK) >> BCPACK_SHIFT) as u8
    }

    fn swizzle(&self) -> u16 {
        ((self.options & SWIZZLE_MASK) >> SWIZZLE_SHIFT) as u16
    }

    fn dest_physical_pages(&self) -> bool {
        self.options & DEST_PHYS_PAGES_BIT != 0
    }

    fn memory_source(&self) -> bool {
        self.options & SOURCE_MEMORY_BIT != 0
    }

    fn source_physical_pages(&self) -> bool {
        self.options & SOURCE_PHYS_PAGES_BIT != 0
    }

    /// What a queue of `source_memory` refuses at enqueue.
    fn check(&self, source_memory: bool) -> Result<(), HRESULT> {
        if self.memory_source() != source_memory || (self.source_physical_pages() && !self.memory_source()) {
            return Err(E_SOURCE_TYPE);
        }
        if self.destination == 0 || (self.memory_source() && self.source == 0) {
            return Err(E_ACCESS_VIOLATION);
        }
        if self.options & !KNOWN_OPTIONS_MASK != 0 {
            warn_once("DSTORAGE_REQUEST_OPTIONS", format_args!("unknown high option bits {:#x}", self.options & !KNOWN_OPTIONS_MASK));
        }
        if self.swizzle() != 0 {
            warn_once("SwizzleMode", format_args!("hardware swizzle mode {} unsupported", self.swizzle()));
            return Err(E_DECOMPRESSION);
        }
        match (self.zlib(), self.bcpack()) {
            // Uncompressed copy: source and destination sizes must match.
            (false, 0) => {
                if self.destination_size != self.source_size || (self.intermediate_size != 0 && self.intermediate_size != self.destination_size) {
                    return Err(E_DESTINATION_SIZE);
                }
            }
            // Zlib decompression: if `intermediate_size > 0`, inflate into an intermediate buffer
            // of that size (`source_size < intermediate_size <= destination_size`) before copying.
            (true, 0) => {
                if self.intermediate_size == 0 {
                    if self.destination_size <= self.source_size {
                        return Err(E_DESTINATION_SIZE);
                    }
                } else if self.intermediate_size <= self.source_size || self.intermediate_size > self.destination_size {
                    return Err(E_DESTINATION_SIZE);
                }
            }
            // Two-stage zlib -> BCPack: validate intermediate bounds when supplied, then refuse BCPack.
            (true, bc) => {
                if self.intermediate_size != 0 && (self.intermediate_size <= self.source_size || self.destination_size <= self.intermediate_size) {
                    return Err(E_DESTINATION_SIZE);
                }
                warn_once(
                    "BCPack",
                    format_args!("zlib + BCPack mode {bc} unsupported (src={}, intermediate={}, dst={})", self.source_size, self.intermediate_size, self.destination_size),
                );
                return Err(E_DECOMPRESSION);
            }
            // Single-stage BCPack: `intermediate_size` must be 0, then refuse BCPack.
            (false, bc) => {
                if self.intermediate_size != 0 {
                    return Err(E_DESTINATION_SIZE);
                }
                warn_once("BCPack", format_args!("BCPack mode {bc} unsupported (src={}, dst={})", self.source_size, self.destination_size));
                return Err(E_DECOMPRESSION);
            }
        }
        Ok(())
    }
}

/// Snapshot a `PULONG_PTR` physical page array at enqueue time (callers often stack-allocate it).
unsafe fn snapshot_page_array(ptr: u64, offset: u16, byte_len: u32) -> Result<Vec<usize>, HRESULT> {
    if byte_len == 0 {
        return Ok(Vec::new());
    }
    let Some(span) = (offset as usize).checked_add(byte_len as usize) else {
        return Err(E_DESTINATION_SIZE);
    };
    let count = span.div_ceil(crate::hle::kernelx::PHYS_PAGE_SIZE);
    let Some(bytes) = count.checked_mul(std::mem::size_of::<usize>()) else {
        return Err(E_ACCESS_VIOLATION);
    };
    if !crate::gpu::shader::readable(ptr as usize, bytes) {
        return Err(E_ACCESS_VIOLATION);
    }
    Ok(std::slice::from_raw_parts(ptr as *const usize, count).to_vec())
}

/// Whether `n` bytes at `p` are committed, writable memory.
fn writable(p: usize, n: usize) -> bool {
    let Some(end) = p.checked_add(n) else { return false };
    crate::hle::kernelx::lazy::ensure(p, n);
    let mut at = p;
    while at < end {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 {
            return false;
        }
        let rw = 0x04 | 0x08 | 0x40 | 0x80; // PAGE_READWRITE / WRITECOPY / EXECUTE_READWRITE / EXECUTE_WRITECOPY
        if m.State != MEM_COMMIT || m.Protect & rw == 0 || m.Protect & 0x100 != 0 {
            return false;
        }
        at = m.BaseAddress as usize + m.RegionSize;
    }
    true
}

fn read_exact_at(f: &std::fs::File, offset: u64, buf: &mut [u8]) -> Result<(), HRESULT> {
    let mut done = 0;
    while done < buf.len() {
        match f.seek_read(&mut buf[done..], offset + done as u64) {
            Ok(0) => return Err(E_END_OF_FILE),
            Ok(n) => done += n,
            Err(e) if e.raw_os_error() == Some(38) => return Err(E_END_OF_FILE), // ERROR_HANDLE_EOF
            Err(e) => return Err(e.raw_os_error().map_or(E_ABORT, |c| (0x8007_0000u32 | (c as u32 & 0xFFFF)) as i32)),
        }
    }
    Ok(())
}

/// Run a request; the destination gets the bytes or nothing.
unsafe fn execute(r: &Request, file: Option<&FileInner>, dest_pages: Option<&[usize]>, src_pages: Option<&[usize]>) -> Result<(), HRESULT> {
    let n = r.destination_size as usize;
    {
        // the first requests: where a title's data goes and how it is packed
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        if k < 16 || k.is_power_of_two() {
            crate::xlog_info!(
                "DirectStorage: request #{k}: options {:#x} ({} source{}), destination {:#x}{} ({n} bytes, page offset {}), source size {}, intermediate {}",
                r.options,
                if r.memory_source() { "memory" } else { "file" },
                if r.zlib() { ", zlib" } else { "" },
                r.destination,
                if r.dest_physical_pages() { " (physical pages)" } else { "" },
                r.destination_page_offset,
                r.source_size,
                r.intermediate_size
            );
        }
    }
    if !r.dest_physical_pages() && !writable(r.destination as usize, n) {
        return Err(E_ACCESS_VIOLATION);
    }
    let src: Vec<u8> = if r.memory_source() {
        let mut b = vec![0u8; r.source_size as usize];
        if r.source_physical_pages() {
            let pages = src_pages.ok_or(E_ACCESS_VIOLATION)?;
            if !crate::hle::kernelx::read_physical_pages(pages, r.source_page_offset as usize, &mut b) {
                return Err(E_ACCESS_VIOLATION);
            }
        } else {
            if !crate::gpu::shader::readable(r.source as usize, r.source_size as usize) {
                return Err(E_ACCESS_VIOLATION);
            }
            if !b.is_empty() {
                std::ptr::copy_nonoverlapping(r.source as *const u8, b.as_mut_ptr(), b.len());
            }
        }
        b
    } else {
        let f = file.and_then(|f| f.handle()).ok_or(E_FILE_HANDLE)?;
        let mut b = vec![0u8; r.source_size as usize];
        read_exact_at(&f, r.offset, &mut b)?;
        b
    };

    if r.swizzle() != 0 || r.bcpack() != 0 {
        return Err(E_DECOMPRESSION);
    }

    let payload: Vec<u8> = if r.zlib() {
        let stage1_len = if r.intermediate_size != 0 { r.intermediate_size as usize } else { n };
        let mut d = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&src, stage1_len).map_err(|_| E_ZLIB_DATA)?;
        if d.len() != stage1_len {
            return Err(E_ZLIB_DATA);
        }
        if d.len() < n {
            d.resize(n, 0);
        }
        d
    } else {
        src
    };

    if r.dest_physical_pages() {
        let pages = dest_pages.ok_or(E_ACCESS_VIOLATION)?;
        if !crate::hle::kernelx::write_physical_pages(pages, r.destination_page_offset as usize, &payload[..n]) {
            return Err(E_ACCESS_VIOLATION);
        }
    } else if n > 0 {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), r.destination as *mut u8, n);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// queues
// ---------------------------------------------------------------------------------------

/// RAII wrapper holding one COM reference to an `ID3D12Fence` so it is never leaked on
/// queue close, cancellation, or drop.
struct FenceRef(usize);

impl FenceRef {
    unsafe fn signal(&self, value: u64) {
        let vt = *(self.0 as *const *const usize);
        let signal: unsafe extern "system" fn(usize, u64) -> HRESULT = std::mem::transmute(*vt.add(FENCE_SIGNAL));
        signal(self.0, value);
    }
}

impl Drop for FenceRef {
    fn drop(&mut self) {
        unsafe {
            let vt = *(self.0 as *const *const usize);
            let release: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*vt.add(2));
            release(self.0);
        }
    }
}

enum Entry {
    Request {
        seq: u64,
        req: Request,
        file: Option<Arc<FileInner>>,
        dest_pages: Option<Vec<usize>>,
        src_pages: Option<Vec<usize>>,
    },
    /// refused at enqueue: reported in queue order by the worker
    Refused {
        hr: HRESULT,
        req: Request,
        file: Option<Arc<FileInner>>,
    },
    Status(Arc<StatusInner>, u32),
    /// a fence (one reference held) and the value it gets
    Signal(FenceRef, u64),
    #[cfg(test)]
    /// Test-only barrier to hold the worker deterministically while testing submitted-batch cancellation.
    Gate(Arc<std::sync::Barrier>),
}

struct Pending {
    entries: Vec<Entry>,
    next_seq: u64,
}

struct CancelRule {
    /// Cancels requests with `seq <= up_to_seq` (enqueued before this `CancelRequestsWithTag` call).
    up_to_seq: u64,
    mask: u64,
    value: u64,
}

struct FirstError {
    hr: HRESULT,
    request: Request,
    file: Vec<u16>,
}

struct QueueInner {
    desc: [u8; 0x18],
    source_memory: bool,
    capacity: usize,
    pending: Mutex<Pending>,
    cancels: Mutex<Vec<CancelRule>>,
    worker: Mutex<Option<mpsc::Sender<Vec<Entry>>>>,
    worker_active: (Mutex<bool>, Condvar),
    closed: AtomicBool,
    error_event: usize,
    errors: Mutex<(u32, Option<FirstError>)>,
}

impl QueueInner {
    fn fail(&self, hr: HRESULT, request: Request, file: Option<&FileInner>) {
        let mut e = self.errors.lock().unwrap();
        e.0 += 1;
        if e.1.is_none() {
            e.1 = Some(FirstError { hr, request, file: file.map(|f| f.path.clone()).unwrap_or_default() });
        }
        unsafe { SetEvent(self.error_event as HANDLE) };
    }

    fn is_cancelled(&self, seq: u64, tag: u64) -> bool {
        let mut rules = self.cancels.lock().unwrap();
        // Worker processes requests in strictly increasing `seq` order, so rules with
        // `up_to_seq < seq` can never match this or any later request.
        rules.retain(|r| r.up_to_seq >= seq);
        rules.iter().any(|r| tag & r.mask == r.value)
    }

    fn submit(self: &Arc<Self>) {
        // Lock `worker` before `pending` so concurrent `Submit()` calls always enqueue
        // batches to the channel in strict `seq` order.
        let mut w = self.worker.lock().unwrap();
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let batch = std::mem::take(&mut self.pending.lock().unwrap().entries);
        if batch.is_empty() {
            return;
        }
        if w.is_none() {
            let (tx, rx) = mpsc::channel::<Vec<Entry>>();
            let q = Arc::clone(self);
            *self.worker_active.0.lock().unwrap() = true;
            let started = std::thread::Builder::new().name("xcl-dstorage".into()).spawn(move || q.run(rx));
            if started.is_err() {
                *self.worker_active.0.lock().unwrap() = false;
                self.worker_active.1.notify_all();
                // Release `worker` before dropping `batch` so fence `Release` cannot re-enter under lock.
                drop(w);
                drop(batch);
                return;
            }
            *w = Some(tx);
        }
        let sent = w.as_ref().unwrap().send(batch);
        // Release `worker` before dropping any unsent `batch` returned in `Err`.
        drop(w);
        drop(sent);
    }

    fn run(&self, rx: mpsc::Receiver<Vec<Entry>>) {
        struct ActiveGuard<'a>(&'a QueueInner);
        impl Drop for ActiveGuard<'_> {
            fn drop(&mut self) {
                *self.0.worker_active.0.lock().unwrap() = false;
                self.0.worker_active.1.notify_all();
            }
        }
        let _active = ActiveGuard(self);
        // the first failure of the requests since the last status entry
        let mut since_status = S_OK;
        for batch in rx {
            for e in batch {
                if self.closed.load(Ordering::Acquire) {
                    if let Entry::Status(s, i) = e {
                        if let Some(slot) = s.slots.get(i as usize) {
                            slot.store(E_QUEUE_CLOSED, Ordering::Release);
                        }
                    }
                    continue;
                }
                match e {
                    Entry::Request { seq, req, file, dest_pages, src_pages } => {
                        if self.is_cancelled(seq, req.tag) {
                            // Cancelled requests are skipped without affecting status or error record.
                            continue;
                        }
                        if let Err(hr) = unsafe { execute(&req, file.as_deref(), dest_pages.as_deref(), src_pages.as_deref()) } {
                            self.fail(hr, req, file.as_deref());
                            if since_status >= 0 {
                                since_status = hr;
                            }
                        }
                    }
                    Entry::Refused { hr, req, file } => {
                        self.fail(hr, req, file.as_deref());
                        if since_status >= 0 {
                            since_status = hr;
                        }
                    }
                    Entry::Status(s, i) => {
                        if let Some(slot) = s.slots.get(i as usize) {
                            slot.store(since_status, Ordering::Release);
                        }
                        since_status = S_OK;
                    }
                    Entry::Signal(fence, value) => unsafe {
                        fence.signal(value);
                    },
                    #[cfg(test)]
                    Entry::Gate(b) => {
                        b.wait();
                    }
                }
            }
        }
    }

    fn enqueue(self: &Arc<Self>, e: Entry) {
        let n = {
            let mut p = self.pending.lock().unwrap();
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            if let Entry::Status(ref s, i) = e {
                if let Some(slot) = s.slots.get(i as usize) {
                    slot.store(E_PENDING, Ordering::Release);
                }
            }
            p.entries.push(e);
            p.entries.len()
        };
        // as on PC: a half-full queue submits by itself
        if n >= (self.capacity / 2).max(1) {
            self.submit();
        }
    }

    fn enqueue_request(self: &Arc<Self>, req: Request, file: Option<Arc<FileInner>>, dest_pages: Option<Vec<usize>>, src_pages: Option<Vec<usize>>) {
        let n = {
            let mut p = self.pending.lock().unwrap();
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            let seq = p.next_seq;
            p.next_seq += 1;
            p.entries.push(Entry::Request { seq, req, file, dest_pages, src_pages });
            p.entries.len()
        };
        if n >= (self.capacity / 2).max(1) {
            self.submit();
        }
    }
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.error_event as HANDLE) };
    }
}

fn queue_vtbl() -> &'static [usize] {
    static V: OnceLock<&'static [usize]> = OnceLock::new();
    V.get_or_init(|| {
        vtable(
            "IDStorageQueueX",
            &[
                queue_qi as usize,
                queue_add_ref as usize,
                queue_release as usize,
                queue_enqueue_request as usize,
                queue_enqueue_status as usize,
                queue_enqueue_signal as usize,
                queue_submit as usize,
                queue_cancel as usize,
                queue_close as usize,
                queue_get_error_event as usize,
                queue_retrieve_error_record as usize,
                queue_query as usize,
            ],
            18,
        )
    })
}

unsafe extern "system" fn factory_create_queue(_p: *mut c_void, desc: *const u8, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let raw: [u8; 0x18] = (desc as *const [u8; 0x18]).read_unaligned();
    let source = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    let capacity = u16::from_le_bytes(raw[8..10].try_into().unwrap()) as usize;
    let priority = raw[10] as i8;
    if source > 1 {
        return E_QUEUE_TYPE;
    }
    if !(-1..=2).contains(&priority) || (source == 1 && priority != 2) {
        return E_QUEUE_PRIORITY;
    }
    if capacity < 2 {
        return E_QUEUE_CAPACITY;
    }
    let ev = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
    if ev.is_null() {
        return E_ABORT;
    }
    let inner = Arc::new(QueueInner {
        desc: raw,
        source_memory: source == 1,
        capacity,
        pending: Mutex::new(Pending { entries: Vec::new(), next_seq: 1 }),
        cancels: Mutex::new(Vec::new()),
        worker: Mutex::new(None),
        worker_active: (Mutex::new(false), Condvar::new()),
        closed: AtomicBool::new(false),
        error_event: ev as usize,
        errors: Mutex::new((0, None)),
    });
    let o = new_object(queue_vtbl(), inner);
    let hr = queue_qi(o, riid, out);
    queue_release(o);
    hr
}

unsafe fn queue<'a>(p: *mut c_void) -> &'a Arc<QueueInner> {
    inner::<QueueInner>(p)
}

unsafe extern "system" fn queue_qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if !riid.is_null() && same(&*riid, &IID_QUEUE_X1) {
        return query(p, &IID_QUEUE_X1, riid, out, add_ref::<QueueInner>);
    }
    query(p, &IID_QUEUE, riid, out, add_ref::<QueueInner>)
}

unsafe extern "system" fn queue_add_ref(p: *mut c_void) -> u32 {
    add_ref::<QueueInner>(p)
}

unsafe extern "system" fn queue_release(p: *mut c_void) -> u32 {
    let obj = p as *mut Object<QueueInner>;
    let n = (*obj).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        let obj = Box::from_raw(obj);
        // Drop the channel sender so the worker thread exits and drops its
        // Arc<QueueInner> (breaking the worker <-> queue cycle).
        obj.inner.worker.lock().unwrap().take();
        drop(obj);
    }
    n
}

unsafe extern "system" fn queue_enqueue_request(p: *mut c_void, req: *const u8) {
    let q = queue(p);
    if q.closed.load(Ordering::Acquire) || req.is_null() {
        return;
    }
    let r = Request::parse(&*(req as *const [u8; REQUEST_BYTES]));
    if let Err(hr) = r.check(q.source_memory) {
        if hr == E_DECOMPRESSION && r.bcpack() == 7 {
            bcpack_capture::capture_refused(&r);
        }
        warn_once("EnqueueRequest", format_args!("refused ({hr:#x}): {r:x?}"));
        return q.enqueue(Entry::Refused { hr, req: r, file: None });
    }
    let file = if r.memory_source() {
        None
    } else {
        // only a file this factory opened
        let f = r.source as *mut c_void;
        if !crate::gpu::shader::readable(f as usize, 8) || *(f as *const *const usize) != file_vtbl().as_ptr() {
            return q.enqueue(Entry::Refused { hr: E_FILE_HANDLE, req: r, file: None });
        }
        Some(Arc::clone(inner::<FileInner>(f)))
    };
    let dest_pages = if r.dest_physical_pages() {
        match snapshot_page_array(r.destination, r.destination_page_offset, r.destination_size) {
            Ok(pages) => Some(pages),
            Err(hr) => {
                return q.enqueue(Entry::Refused { hr, req: r, file });
            }
        }
    } else {
        None
    };
    let src_pages = if r.memory_source() && r.source_physical_pages() {
        match snapshot_page_array(r.source, r.source_page_offset, r.source_size) {
            Ok(pages) => Some(pages),
            Err(hr) => {
                return q.enqueue(Entry::Refused { hr, req: r, file });
            }
        }
    } else {
        None
    };
    q.enqueue_request(r, file, dest_pages, src_pages);
}

unsafe extern "system" fn queue_enqueue_status(p: *mut c_void, array: *mut c_void, index: u32) {
    let q = queue(p);
    if q.closed.load(Ordering::Acquire)
        || array.is_null()
        || !crate::gpu::shader::readable(array as usize, std::mem::size_of::<usize>())
        || *(array as *const *const usize) != status_vtbl().as_ptr()
    {
        return;
    }
    let s = Arc::clone(inner::<StatusInner>(array));
    q.enqueue(Entry::Status(s, index));
}

unsafe extern "system" fn queue_enqueue_signal(p: *mut c_void, fence: *mut c_void, value: u64) {
    let q = queue(p);
    if q.closed.load(Ordering::Acquire) || fence.is_null() || !crate::gpu::shader::readable(fence as usize, std::mem::size_of::<usize>()) {
        return;
    }
    let vt = *(fence as *const *const usize);
    if vt.is_null() || !crate::gpu::shader::readable(vt as usize, (FENCE_SIGNAL + 1) * std::mem::size_of::<usize>()) {
        return;
    }
    let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(1));
    add(fence);
    q.enqueue(Entry::Signal(FenceRef(fence as usize), value));
}

unsafe extern "system" fn queue_submit(p: *mut c_void) {
    let q = queue(p);
    if !q.closed.load(Ordering::Acquire) {
        q.submit();
    }
}

/// Requests enqueued before this call (both pending and already submitted but not yet
/// executed) whose tag matches (`tag & mask == value`) are skipped without error.
unsafe extern "system" fn queue_cancel(p: *mut c_void, mask: u64, value: u64) {
    let q = queue(p);
    let cutoff = {
        let mut pend = q.pending.lock().unwrap();
        pend.entries.retain(|e| !matches!(e, Entry::Request { req, .. } if req.tag & mask == value));
        pend.next_seq.saturating_sub(1)
    };
    if cutoff == 0 {
        return;
    }
    let mut rules = q.cancels.lock().unwrap();
    if let Some(existing) = rules.iter_mut().find(|r| r.mask == mask && r.value == value) {
        existing.up_to_seq = existing.up_to_seq.max(cutoff);
    } else {
        rules.push(CancelRule { up_to_seq: cutoff, mask, value });
    }
}

unsafe extern "system" fn queue_close(p: *mut c_void) {
    let q = queue(p);
    // Lock `worker` then `pending` (same order as `submit`) so no racing enqueue or submit
    // can push new pending entries or spawn a new worker after `closed` is set.
    let drained = {
        let mut w = q.worker.lock().unwrap();
        let mut pend = q.pending.lock().unwrap();
        if q.closed.swap(true, Ordering::AcqRel) {
            Vec::new()
        } else {
            w.take();
            std::mem::take(&mut pend.entries)
        }
    };
    for e in drained {
        if let Entry::Status(s, i) = e {
            if let Some(slot) = s.slots.get(i as usize) {
                slot.store(E_QUEUE_CLOSED, Ordering::Release);
            }
        }
        // `Entry::Signal` drops `FenceRef` (calling Release); `Entry::Request` drops `FileInner`.
    }
    // Wait outside queue locks so no in-flight worker request can still touch caller buffers
    // after `Close()` returns (including for concurrent `Close()` callers).
    let (lock, cvar) = &q.worker_active;
    let mut active = lock.lock().unwrap();
    while *active {
        active = cvar.wait(active).unwrap();
    }
}

/// Returns a duplicated event handle owned by the caller (who closes it with `CloseHandle`).
unsafe extern "system" fn queue_get_error_event(p: *mut c_void) -> HANDLE {
    let mut h: HANDLE = std::ptr::null_mut();
    let me = GetCurrentProcess();
    if DuplicateHandle(me, queue(p).error_event as HANDLE, me, &mut h, 0, 0, DUPLICATE_SAME_ACCESS) == 0 {
        return std::ptr::null_mut();
    }
    h
}

/// DSTORAGE_ERROR_RECORD (0x290): failure count u32 @0, then the first failure: HRESULT @8,
/// command type @0xC (none: 0xFFFFFFFF), the request's options @0x10, destination @0x18 and
/// size @0x20, its file name @0x24 (0x104 UTF-16) or memory source @0x238
unsafe extern "system" fn queue_retrieve_error_record(p: *mut c_void, out: *mut u8) {
    if out.is_null() {
        return;
    }
    let q = queue(p);
    let mut rec = [0u8; ERROR_RECORD_BYTES];
    {
        let mut e = q.errors.lock().unwrap();
        match &e.1 {
            Some(f) => {
                rec[0..4].copy_from_slice(&e.0.to_le_bytes());
                rec[8..12].copy_from_slice(&f.hr.to_le_bytes());
                rec[0xC..0x10].copy_from_slice(&COMMAND_REQUEST.to_le_bytes());
                rec[0x10..0x18].copy_from_slice(&f.request.options.to_le_bytes());
                rec[0x18..0x20].copy_from_slice(&f.request.destination.to_le_bytes());
                rec[0x20..0x24].copy_from_slice(&f.request.destination_size.to_le_bytes());
                if f.request.memory_source() {
                    rec[0x238..0x240].copy_from_slice(&f.request.source.to_le_bytes());
                } else {
                    for (i, c) in f.file.iter().take(0x103).enumerate() {
                        rec[0x24 + 2 * i..0x26 + 2 * i].copy_from_slice(&c.to_le_bytes());
                    }
                }
            }
            None => rec[0xC..0x10].copy_from_slice(&NO_COMMAND.to_le_bytes()),
        }
        e.0 = 0;
        e.1 = None;
        ResetEvent(q.error_event as HANDLE);
    }
    std::ptr::copy_nonoverlapping(rec.as_ptr(), out, ERROR_RECORD_BYTES);
}

/// DSTORAGE_QUEUE_INFO (0x1C): the queue's desc, empty slots, requests until it submits by itself
unsafe extern "system" fn queue_query(p: *mut c_void, out: *mut u8) {
    if out.is_null() {
        return;
    }
    let q = queue(p);
    let pending = q.pending.lock().unwrap().entries.len();
    let empty = q.capacity.saturating_sub(pending) as u16;
    let until = (q.capacity / 2).max(1).saturating_sub(pending).max(1) as u16;
    let mut info = [0u8; 0x1C];
    info[..0x18].copy_from_slice(&q.desc);
    info[0x18..0x1A].copy_from_slice(&empty.to_le_bytes());
    info[0x1A..0x1C].copy_from_slice(&until.to_le_bytes());
    std::ptr::copy_nonoverlapping(info.as_ptr(), out, info.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(options: u64, dst: u64, dst_size: u32, src: u64, offset: u64, src_size: u32) -> [u8; REQUEST_BYTES] {
        let mut b = [0u8; REQUEST_BYTES];
        b[0..8].copy_from_slice(&options.to_le_bytes());
        b[8..16].copy_from_slice(&dst.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&dst_size.to_le_bytes());
        b[0x20..0x28].copy_from_slice(&src.to_le_bytes());
        b[0x28..0x30].copy_from_slice(&offset.to_le_bytes());
        b[0x30..0x34].copy_from_slice(&src_size.to_le_bytes());
        b
    }

    #[test]
    fn requests_are_checked_like_the_console() {
        let ok = Request::parse(&request(0, 0x1000, 64, 0x2000, 0, 64));
        assert_eq!(ok.check(false), Ok(()));
        assert_eq!(ok.check(true), Err(E_SOURCE_TYPE));
        let mem = Request::parse(&request(SOURCE_MEMORY_BIT | 1, 0x1000, 64, 0x2000, 0x99, 20));
        assert_eq!((mem.offset, mem.check(true)), (0, Ok(())), "a memory source has no offset");
        assert_eq!(Request::parse(&request(1, 0x1000, 20, 0x2000, 0, 20)).check(false), Err(E_DESTINATION_SIZE));
        assert_eq!(Request::parse(&request(3, 0x1000, 64, 0x2000, 0, 20)).check(false), Err(E_DECOMPRESSION), "BCPack");
        assert_eq!(Request::parse(&request(0, 0, 64, 0x2000, 0, 64)).check(false), Err(E_ACCESS_VIOLATION));
    }

    #[test]
    fn a_memory_queue_inflates_zlib_and_reports_through_status_and_errors() {
        unsafe {
            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let call = |i: usize| *vt.add(i);
            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&8u16.to_le_bytes());
            desc[10] = 2;
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(call(3));
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            desc[10] = 0;
            let mut bad = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut bad), E_QUEUE_PRIORITY, "a memory queue is realtime");
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(call(5));
            let mut s = std::ptr::null_mut();
            assert_eq!(create_status(f, 2, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);

            let plain = b"XCL DirectStorage zlib payload, XCL DirectStorage zlib payload".to_vec();
            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
            let mut dst = vec![0u8; plain.len()];
            let qv = *(q as *const *const usize);
            let enqueue: unsafe extern "system" fn(*mut c_void, *const u8) = std::mem::transmute(*qv.add(3));
            let status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = std::mem::transmute(*qv.add(4));
            let submit: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(*qv.add(6));
            enqueue(q, request(SOURCE_MEMORY_BIT | 1, dst.as_mut_ptr() as u64, dst.len() as u32, packed.as_ptr() as u64, 0, packed.len() as u32).as_ptr());
            status(q, s, 0);
            // a refused request: the next status reports it, the error record names it
            enqueue(q, request(SOURCE_MEMORY_BIT, dst.as_mut_ptr() as u64, 8, packed.as_ptr() as u64, 0, 4).as_ptr());
            status(q, s, 1);
            submit(q);
            let sv = *(s as *const *const usize);
            let complete: unsafe extern "system" fn(*mut c_void, u32) -> bool = std::mem::transmute(*sv.add(3));
            let hresult: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT = std::mem::transmute(*sv.add(4));
            let t = std::time::Instant::now();
            while !complete(s, 1) && t.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            assert_eq!((hresult(s, 0), hresult(s, 1)), (S_OK, E_DESTINATION_SIZE));
            assert_eq!(dst, plain);
            let record: unsafe extern "system" fn(*mut c_void, *mut u8) = std::mem::transmute(*qv.add(10));
            let mut rec = [0u8; ERROR_RECORD_BYTES];
            record(q, rec.as_mut_ptr());
            assert_eq!((u32::from_le_bytes(rec[0..4].try_into().unwrap()), i32::from_le_bytes(rec[8..12].try_into().unwrap())), (1, E_DESTINATION_SIZE));
            let qinfo: unsafe extern "system" fn(*mut c_void, *mut u8) = std::mem::transmute(*qv.add(11));
            let mut info = [0u8; 0x1C];
            qinfo(q, info.as_mut_ptr());
            assert_eq!((info[10], u16::from_le_bytes([info[0x18], info[0x19]])), (2, 8));
            queue_close(q);
            queue_release(q);
            status_release(s);
        }
    }

    fn request_ex(
        options: u64,
        dst: u64,
        dst_page_offset: u16,
        dst_size: u32,
        src: u64,
        offset_or_page_off: u64,
        src_size: u32,
        intermediate_size: u32,
        tag: u64,
    ) -> [u8; REQUEST_BYTES] {
        let mut b = request(options, dst, dst_size, src, offset_or_page_off, src_size);
        b[0x10..0x12].copy_from_slice(&dst_page_offset.to_le_bytes());
        b[0x34..0x38].copy_from_slice(&intermediate_size.to_le_bytes());
        b[0x38..0x40].copy_from_slice(&tag.to_le_bytes());
        b
    }

    #[repr(C)]
    struct MockFence {
        vtbl: *const usize,
        refs: AtomicU32,
        signalled: std::sync::atomic::AtomicU64,
    }

    fn mock_fence_vtbl() -> &'static [usize] {
        unsafe extern "system" fn qi(_p: *mut c_void, _riid: *const GUID, _out: *mut *mut c_void) -> HRESULT {
            E_NOINTERFACE
        }
        unsafe extern "system" fn add(p: *mut c_void) -> u32 {
            (*(p as *const MockFence)).refs.fetch_add(1, Ordering::Relaxed) + 1
        }
        unsafe extern "system" fn rel(p: *mut c_void) -> u32 {
            (*(p as *const MockFence)).refs.fetch_sub(1, Ordering::AcqRel) - 1
        }
        unsafe extern "system" fn sig(p: *mut c_void, val: u64) -> HRESULT {
            (*(p as *const MockFence)).signalled.store(val, Ordering::Release);
            S_OK
        }
        static V: OnceLock<&'static [usize]> = OnceLock::new();
        V.get_or_init(|| {
            let mut v = vec![0usize; FENCE_SIGNAL + 1];
            v[0] = qi as usize;
            v[1] = add as usize;
            v[2] = rel as usize;
            v[FENCE_SIGNAL] = sig as usize;
            Box::leak(v.into_boxed_slice())
        })
    }

    #[test]
    fn cancel_pending_and_submitted_requests_by_tag() {
        unsafe {
            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(3));
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(5));

            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes()); // memory queue
            desc[8..10].copy_from_slice(&32u16.to_le_bytes());
            desc[10] = 2;
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(create_status(f, 3, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);

            let qv = *(q as *const *const usize);
            let enqueue: unsafe extern "system" fn(*mut c_void, *const u8) = std::mem::transmute(*qv.add(3));
            let status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = std::mem::transmute(*qv.add(4));
            let signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) = std::mem::transmute(*qv.add(5));
            let submit: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(*qv.add(6));
            let cancel: unsafe extern "system" fn(*mut c_void, u64, u64) = std::mem::transmute(*qv.add(7));
            let record: unsafe extern "system" fn(*mut c_void, *mut u8) = std::mem::transmute(*qv.add(10));
            let sv = *(s as *const *const usize);
            let complete: unsafe extern "system" fn(*mut c_void, u32) -> bool = std::mem::transmute(*sv.add(3));
            let hresult: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT = std::mem::transmute(*sv.add(4));

            let src_a = b"SUBMITTED_CANCELLED";
            let src_b = b"SUBMITTED_KEPT_____";
            let src_c = b"POST_CANCEL_MATCH__";
            let src_d = b"PENDING_CANCELLED__";
            let mut dst_a = vec![0u8; src_a.len()];
            let mut dst_b = vec![0u8; src_b.len()];
            let mut dst_c = vec![0u8; src_c.len()];
            let mut dst_d = vec![0u8; src_d.len()];

            let fence = MockFence {
                vtbl: mock_fence_vtbl().as_ptr(),
                refs: AtomicU32::new(1),
                signalled: std::sync::atomic::AtomicU64::new(0),
            };
            let fence_ptr = &fence as *const MockFence as *mut c_void;

            // Block the worker deterministically so the submitted batch sits in the channel.
            let gate = Arc::new(std::sync::Barrier::new(2));
            queue(q).enqueue(Entry::Gate(Arc::clone(&gate)));
            // Submitted + matching tag (0x1001 & 0xF000 == 0x1000) -> cancelled.
            enqueue(q, request_ex(SOURCE_MEMORY_BIT, dst_a.as_mut_ptr() as u64, 0, dst_a.len() as u32, src_a.as_ptr() as u64, 0, src_a.len() as u32, 0, 0x1001).as_ptr());
            // Submitted + matching tag pointing to unwritable address (0x10) -> cancelled before execute, so S_OK & 0 errors!
            enqueue(q, request_ex(SOURCE_MEMORY_BIT, 0x10, 0, 4, src_a.as_ptr() as u64, 0, 4, 0, 0x1002).as_ptr());
            // Submitted + non-matching tag (0x2001) -> executes.
            enqueue(q, request_ex(SOURCE_MEMORY_BIT, dst_b.as_mut_ptr() as u64, 0, dst_b.len() as u32, src_b.as_ptr() as u64, 0, src_b.len() as u32, 0, 0x2001).as_ptr());
            status(q, s, 0);
            signal(q, fence_ptr, 42);
            submit(q);
            assert_eq!(fence.refs.load(Ordering::Acquire), 2, "queue holds one fence reference");

            // Also enqueue an unsubmitted request with a matching tag before cancelling.
            enqueue(q, request_ex(SOURCE_MEMORY_BIT, dst_d.as_mut_ptr() as u64, 0, dst_d.len() as u32, src_d.as_ptr() as u64, 0, src_d.len() as u32, 0, 0x1003).as_ptr());
            cancel(q, 0xF000, 0x1000);

            // Enqueued AFTER CancelRequestsWithTag with a matching tag -> must NOT be cancelled.
            enqueue(q, request_ex(SOURCE_MEMORY_BIT, dst_c.as_mut_ptr() as u64, 0, dst_c.len() as u32, src_c.as_ptr() as u64, 0, src_c.len() as u32, 0, 0x1004).as_ptr());
            status(q, s, 1);
            submit(q);

            // Release the worker and wait for completion.
            gate.wait();
            let t = std::time::Instant::now();
            while !complete(s, 1) && t.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            assert!(complete(s, 0) && complete(s, 1));
            assert_eq!((hresult(s, 0), hresult(s, 1)), (S_OK, S_OK));
            assert!(dst_a.iter().all(|&b| b == 0), "submitted request should be cancelled");
            assert!(dst_d.iter().all(|&b| b == 0), "pending request should be cancelled");
            assert_eq!(dst_b.as_slice(), src_b);
            assert_eq!(dst_c.as_slice(), src_c, "request enqueued after cancel must run");
            assert_eq!(fence.signalled.load(Ordering::Acquire), 42);
            assert_eq!(fence.refs.load(Ordering::Acquire), 1, "fence reference released after signal");
            assert!(queue(q).cancels.lock().unwrap().is_empty(), "expired cancel rules are pruned");

            let mut rec = [0u8; ERROR_RECORD_BYTES];
            record(q, rec.as_mut_ptr());
            assert_eq!(u32::from_le_bytes(rec[0..4].try_into().unwrap()), 0, "cancelled requests do not record errors");

            // Closing a queue with an unsubmitted signal must release its fence ref cleanly.
            signal(q, fence_ptr, 99);
            status(q, s, 2);
            assert_eq!(fence.refs.load(Ordering::Acquire), 2);
            queue_close(q);
            assert_eq!(fence.refs.load(Ordering::Acquire), 1, "queue_close releases pending fence");
            assert_eq!(hresult(s, 2), E_QUEUE_CLOSED);

            queue_release(q);
            status_release(s);
        }
    }

    #[test]
    fn multi_stage_intermediate_size_validation_and_zlib() {
        // Validation matrix for Request::check
        let bc1 = 1u64 << BCPACK_SHIFT;
        let bc7 = 7u64 << BCPACK_SHIFT;
        let swiz = 1u64 << SWIZZLE_SHIFT;

        // Zlib only with valid intermediate_size (== dst and < dst)
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT, 0x1000, 0, 64, 0x2000, 0, 20, 64, 0)).check(false), Ok(()));
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT, 0x1000, 0, 128, 0x2000, 0, 20, 64, 0)).check(false), Ok(()));
        // Zlib only with invalid intermediate_size (<= src or > dst)
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT, 0x1000, 0, 128, 0x2000, 0, 20, 20, 0)).check(false), Err(E_DESTINATION_SIZE));
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT, 0x1000, 0, 64, 0x2000, 0, 20, 96, 0)).check(false), Err(E_DESTINATION_SIZE));
        // Two-stage zlib + BCPack: valid sizes -> E_DECOMPRESSION, invalid intermediate -> E_DESTINATION_SIZE
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT | bc1, 0x1000, 0, 256, 0x2000, 0, 20, 64, 0)).check(false), Err(E_DECOMPRESSION));
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT | bc7, 0x1000, 0, 256, 0x2000, 0, 20, 20, 0)).check(false), Err(E_DESTINATION_SIZE));
        assert_eq!(Request::parse(&request_ex(ZLIB_BIT | bc7, 0x1000, 0, 64, 0x2000, 0, 20, 64, 0)).check(false), Err(E_DESTINATION_SIZE));
        // Single-stage BCPack: intermediate must be 0, then E_DECOMPRESSION
        assert_eq!(Request::parse(&request_ex(bc1, 0x1000, 0, 256, 0x2000, 0, 20, 0, 0)).check(false), Err(E_DECOMPRESSION));
        assert_eq!(Request::parse(&request_ex(bc1, 0x1000, 0, 256, 0x2000, 0, 20, 64, 0)).check(false), Err(E_DESTINATION_SIZE));
        // Hardware swizzle mode unsupported
        assert_eq!(Request::parse(&request_ex(swiz, 0x1000, 0, 64, 0x2000, 0, 64, 0, 0)).check(false), Err(E_DECOMPRESSION));
        // SourceIsPhysicalPages requires memory source
        assert_eq!(Request::parse(&request_ex(SOURCE_PHYS_PAGES_BIT, 0x1000, 0, 64, 0x2000, 0, 64, 0, 0)).check(false), Err(E_SOURCE_TYPE));

        // End-to-end execution: zlib with intermediate_size < destination_size (padded destination)
        unsafe {
            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(3));
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(5));

            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&8u16.to_le_bytes());
            desc[10] = 2;
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(create_status(f, 2, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);

            let plain = b"XCL intermediate-buffer zlib test payload! XCL intermediate-buffer zlib test payload!".to_vec();
            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
            assert!(packed.len() < plain.len());
            let mut dst = vec![0xCCu8; plain.len() + 32];

            let qv = *(q as *const *const usize);
            let enqueue: unsafe extern "system" fn(*mut c_void, *const u8) = std::mem::transmute(*qv.add(3));
            let status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = std::mem::transmute(*qv.add(4));
            let submit: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(*qv.add(6));
            let sv = *(s as *const *const usize);
            let complete: unsafe extern "system" fn(*mut c_void, u32) -> bool = std::mem::transmute(*sv.add(3));
            let hresult: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT = std::mem::transmute(*sv.add(4));

            // Valid intermediate_size == plain.len() < dst.len()
            enqueue(
                q,
                request_ex(SOURCE_MEMORY_BIT | ZLIB_BIT, dst.as_mut_ptr() as u64, 0, dst.len() as u32, packed.as_ptr() as u64, 0, packed.len() as u32, plain.len() as u32, 0).as_ptr(),
            );
            status(q, s, 0);
            // Wrong intermediate_size (1 byte off) -> E_ZLIB_DATA
            let mut dst_bad = vec![0u8; plain.len() + 32];
            enqueue(
                q,
                request_ex(SOURCE_MEMORY_BIT | ZLIB_BIT, dst_bad.as_mut_ptr() as u64, 0, dst_bad.len() as u32, packed.as_ptr() as u64, 0, packed.len() as u32, (plain.len() + 1) as u32, 0).as_ptr(),
            );
            status(q, s, 1);
            submit(q);

            let t = std::time::Instant::now();
            while !complete(s, 1) && t.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            assert_eq!((hresult(s, 0), hresult(s, 1)), (S_OK, E_ZLIB_DATA));
            assert_eq!(&dst[..plain.len()], plain.as_slice());
            assert!(dst[plain.len()..].iter().all(|&b| b == 0), "trailing padding zeroed");

            queue_close(q);
            queue_release(q);
            status_release(s);
        }
    }

    #[test]
    fn error_record_resets_after_retrieval_and_event_is_duplicated() {
        use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
        use windows_sys::Win32::System::Threading::WaitForSingleObject;

        unsafe {
            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(3));
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(5));
            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&8u16.to_le_bytes());
            desc[10] = 2;
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(create_status(f, 1, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);

            // Caller owns and closes the returned error event handle without breaking the queue.
            let ev1 = queue_get_error_event(q);
            assert!(!ev1.is_null());
            assert_ne!(ev1 as usize, queue(q).error_event, "GetErrorEvent must return a duplicated handle");
            assert_eq!(WaitForSingleObject(ev1, 0), WAIT_TIMEOUT);
            assert_ne!(CloseHandle(ev1), 0);

            let ev2 = queue_get_error_event(q);
            assert!(!ev2.is_null());
            assert_eq!(WaitForSingleObject(ev2, 0), WAIT_TIMEOUT);

            // Trigger a refused request -> reported in queue order on Submit().
            let mut dst = [0u8; 8];
            let src = [0u8; 4];
            queue_enqueue_request(q, request(SOURCE_MEMORY_BIT, dst.as_mut_ptr() as u64, 8, src.as_ptr() as u64, 0, 4).as_ptr());
            queue_enqueue_status(q, s, 0);
            assert_eq!(WaitForSingleObject(ev2, 0), WAIT_TIMEOUT, "error event must not signal before Submit");
            queue_submit(q);
            let start = std::time::Instant::now();
            while !status_is_complete(s, 0) && start.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            assert_eq!(status_get_hresult(s, 0), E_DESTINATION_SIZE);
            assert_eq!(WaitForSingleObject(ev2, 0), WAIT_OBJECT_0, "error event must signal on failure");

            let mut rec = [0u8; ERROR_RECORD_BYTES];
            queue_retrieve_error_record(q, rec.as_mut_ptr());
            assert_eq!(u32::from_le_bytes(rec[0..4].try_into().unwrap()), 1);
            assert_eq!(i32::from_le_bytes(rec[8..12].try_into().unwrap()), E_DESTINATION_SIZE);
            assert_eq!(u32::from_le_bytes(rec[0xC..0x10].try_into().unwrap()), COMMAND_REQUEST);
            assert_eq!(WaitForSingleObject(ev2, 0), WAIT_TIMEOUT, "RetrieveErrorRecord must reset error event");

            // Second retrieval sees cleared state.
            rec.fill(0xAA);
            queue_retrieve_error_record(q, rec.as_mut_ptr());
            assert_eq!(u32::from_le_bytes(rec[0..4].try_into().unwrap()), 0);
            assert_eq!(u32::from_le_bytes(rec[0xC..0x10].try_into().unwrap()), NO_COMMAND);

            assert_ne!(CloseHandle(ev2), 0);
            queue_close(q);
            queue_release(q);
            status_release(s);
        }
    }

    #[test]
    fn queue_release_concurrent_breaks_worker_cycle() {
        unsafe {
            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(3));
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(5));
            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&8u16.to_le_bytes());
            desc[10] = 2;

            for _ in 0..32 {
                let mut q = std::ptr::null_mut();
                assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
                let mut s = std::ptr::null_mut();
                assert_eq!(create_status(f, 1, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);
                let weak = Arc::downgrade(queue(q));

                // Start the worker thread with a buffer-free status entry.
                queue_enqueue_status(q, s, 0);
                queue_submit(q);

                // Race two Release() calls from refs == 2.
                assert_eq!(queue_add_ref(q), 2);
                let barrier = Arc::new(std::sync::Barrier::new(2));
                let q_addr = q as usize;
                let b2 = Arc::clone(&barrier);
                let t = std::thread::spawn(move || {
                    b2.wait();
                    queue_release(q_addr as *mut c_void);
                });
                barrier.wait();
                queue_release(q);
                t.join().unwrap();

                // Worker must exit and drop the final Arc<QueueInner>.
                let start = std::time::Instant::now();
                while weak.upgrade().is_some() && start.elapsed().as_secs() < 5 {
                    std::thread::yield_now();
                }
                assert!(weak.upgrade().is_none(), "concurrent Release leaked QueueInner / worker thread");
                status_release(s);
            }
        }
    }

    #[test]
    fn queue_validates_status_and_fence_pointers_and_closed_rejects_work() {
        unsafe {
            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(3));
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(5));
            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&8u16.to_le_bytes());
            desc[10] = 2;
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(create_status(f, 2, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);

            // Foreign object pointer (wrong vtable) must be ignored safely by EnqueueStatus.
            let bogus_vtbl: [usize; 4] = [0; 4];
            let mut bogus_obj: [*const usize; 2] = [bogus_vtbl.as_ptr(), std::ptr::null()];
            queue_enqueue_status(q, bogus_obj.as_mut_ptr() as *mut c_void, 0);
            // Null-vtable fence must be ignored safely by EnqueueSignal.
            let mut null_vtbl_fence: [*const usize; 1] = [std::ptr::null()];
            queue_enqueue_signal(q, null_vtbl_fence.as_mut_ptr() as *mut c_void, 1);
            assert!(queue(q).pending.lock().unwrap().entries.is_empty());

            // Pending status fails with E_QUEUE_CLOSED on Close().
            queue_enqueue_status(q, s, 0);
            queue_close(q);
            assert_eq!(status_get_hresult(s, 0), E_QUEUE_CLOSED);

            // Subsequent EnqueueStatus / Submit on closed queue do not mutate status or spawn a worker.
            inner::<StatusInner>(s).slots[1].store(S_OK, Ordering::Release);
            queue_enqueue_status(q, s, 1);
            queue_submit(q);
            assert_eq!(status_get_hresult(s, 1), S_OK);
            assert!(queue(q).pending.lock().unwrap().entries.is_empty());
            assert!(queue(q).worker.lock().unwrap().is_none());

            queue_release(q);
            status_release(s);
        }
    }

    #[test]
    fn physical_page_destinations_and_sources() {
        use crate::hle::kernelx::{AllocateTitlePhysicalPages, FreeTitlePhysicalPages, MapTitlePhysicalPages, PHYS_PAGE_SIZE};
        if crate::gpu::uma::get().is_none() {
            return;
        }
        unsafe {
            let mut count = 3usize;
            let mut pages = [0usize; 3];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, pages.as_mut_ptr()), 1);

            let mut f = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
            let vt = *(f as *const *const usize);
            let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(3));
            let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(*vt.add(5));

            let mut desc = [0u8; 0x18];
            desc[0..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&16u16.to_le_bytes());
            desc[10] = 2;
            let mut q = std::ptr::null_mut();
            assert_eq!(create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(create_status(f, 4, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);

            let qv = *(q as *const *const usize);
            let enqueue: unsafe extern "system" fn(*mut c_void, *const u8) = std::mem::transmute(*qv.add(3));
            let status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = std::mem::transmute(*qv.add(4));
            let submit: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(*qv.add(6));
            let sv = *(s as *const *const usize);
            let complete: unsafe extern "system" fn(*mut c_void, u32) -> bool = std::mem::transmute(*sv.add(3));
            let hresult: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT = std::mem::transmute(*sv.add(4));

            // 1. Zlib inflate into scattered physical pages [pages[2], pages[0]] crossing a 64 KiB boundary.
            let plain: Vec<u8> = (0..0x1000u32).map(|i| ((i * 37) ^ (i >> 3)) as u8).collect();
            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
            let page_off = (PHYS_PAGE_SIZE - 0x800) as u16;
            let mut scattered = [pages[2], pages[0]];
            enqueue(
                q,
                request_ex(
                    SOURCE_MEMORY_BIT | ZLIB_BIT | DEST_PHYS_PAGES_BIT,
                    scattered.as_ptr() as u64,
                    page_off,
                    plain.len() as u32,
                    packed.as_ptr() as u64,
                    0,
                    packed.len() as u32,
                    0,
                    0,
                )
                .as_ptr(),
            );
            // Clobber the stack array before Submit() to verify the page list was snapshotted at enqueue.
            std::ptr::write_volatile(&mut scattered, [0, 0]);
            status(q, s, 0);

            // 2. Read back from those scattered physical pages via SourceIsPhysicalPages into plain memory.
            let mut readback = vec![0u8; plain.len()];
            let src_scattered = [pages[2], pages[0]];
            enqueue(
                q,
                request_ex(
                    SOURCE_MEMORY_BIT | SOURCE_PHYS_PAGES_BIT,
                    readback.as_mut_ptr() as u64,
                    0,
                    readback.len() as u32,
                    src_scattered.as_ptr() as u64,
                    page_off as u64,
                    readback.len() as u32,
                    0,
                    0,
                )
                .as_ptr(),
            );
            status(q, s, 1);

            // 3. Map pages[1] into a reserved virtual address (like a streamed texture) and write via physical page!
            let reserved = windows_sys::Win32::System::Memory::VirtualAlloc(
                std::ptr::null(),
                PHYS_PAGE_SIZE,
                windows_sys::Win32::System::Memory::MEM_RESERVE,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
            );
            assert!(!reserved.is_null());
            assert_eq!(MapTitlePhysicalPages(reserved, 1, 0, 4, &mut pages[1]), reserved);
            let streamed = b"DIRECTSTORAGE_TO_MAPPED_PAGE";
            let mapped_list = [pages[1]];
            enqueue(
                q,
                request_ex(
                    SOURCE_MEMORY_BIT | DEST_PHYS_PAGES_BIT,
                    mapped_list.as_ptr() as u64,
                    0x40,
                    streamed.len() as u32,
                    streamed.as_ptr() as u64,
                    0,
                    streamed.len() as u32,
                    0,
                    0,
                )
                .as_ptr(),
            );
            status(q, s, 2);

            // 4. Invalid physical page number -> E_ACCESS_VIOLATION.
            let bogus = [usize::MAX];
            enqueue(
                q,
                request_ex(
                    SOURCE_MEMORY_BIT | DEST_PHYS_PAGES_BIT,
                    bogus.as_ptr() as u64,
                    0,
                    streamed.len() as u32,
                    streamed.as_ptr() as u64,
                    0,
                    streamed.len() as u32,
                    0,
                    0,
                )
                .as_ptr(),
            );
            status(q, s, 3);

            submit(q);
            let t = std::time::Instant::now();
            while !complete(s, 3) && t.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            assert_eq!((hresult(s, 0), hresult(s, 1), hresult(s, 2), hresult(s, 3)), (S_OK, S_OK, S_OK, E_ACCESS_VIOLATION));
            assert_eq!(readback, plain);
            let mapped_slice = std::slice::from_raw_parts((reserved as *const u8).add(0x40), streamed.len());
            assert_eq!(mapped_slice, streamed, "DirectStorage physical-page write lands in mapped reservation");

            // Unmap and free cleanly.
            MapTitlePhysicalPages(reserved, 1, 0, 4, std::ptr::null_mut());
            windows_sys::Win32::System::Memory::VirtualFree(reserved, 0, windows_sys::Win32::System::Memory::MEM_RELEASE);
            assert_eq!(FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            queue_close(q);
            queue_release(q);
            status_release(s);
        }
    }
}
