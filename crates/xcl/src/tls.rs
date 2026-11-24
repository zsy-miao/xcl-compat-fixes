//! Static TLS (`__declspec(thread)` / Rust `thread_local!`) for manually mapped images.
//!
//! Problem: the Windows loader only allocates static-TLS blocks for modules it loaded
//! itself. A manually mapped game image would read `ThreadLocalStoragePointer[_tls_index]`
//! and crash.
//!
//! Solution ("TLS carrier"): for every mapped image that has a TLS directory we synthesise
//! a tiny DLL whose own TLS template is at least as large as the game's and load it with
//! `LoadLibraryW`. Windows then owns a real TLS slot for it – allocated for every existing
//! and future thread, re-allocated when other TLS modules load, freed at thread exit. We
//! write that slot's index into the game's `_tls_index` and copy the game's TLS template
//! into the block of each thread:
//! * existing threads – right after the carrier is loaded (via their TEB),
//! * new threads – from `xcl.exe`'s own TLS callback (`on_thread_attach`), which ntdll
//!   calls after allocating all TLS blocks and before the thread start routine runs.
//!
//! Two empirical findings on Windows 11 (verified by `carrier_variants_probe`):
//! 1. ntdll only processes the TLS directory of a dynamically loaded DLL that has an
//!    import table (it happens during import snapping) – so the carrier imports
//!    `KERNEL32!GetTickCount`.
//! 2. `SizeOfZeroFill` is ignored for dynamically loaded DLLs – the block is only
//!    `End - Start` bytes. So the whole size lives in the "initialised" range, which is
//!    simply zero-filled virtual memory past the end of the section's raw data.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use crate::util::wide;

pub const NO_INDEX: u32 = u32::MAX;

// ---------------------------------------------------------------------------------------
// Carrier DLL synthesis (pure, unit-tested)
// ---------------------------------------------------------------------------------------

const CARRIER_BASE: u64 = 0x0000_0001_7F00_0000;
const FILE_ALIGN: usize = 0x200;
const TEXT_RVA: u32 = 0x1000;
const RELOC_RVA: u32 = 0x2000;
pub const DATA_RVA: u32 = 0x3000;
// Offsets inside .data
const OFF_TLS_DIR: u32 = 0x00;
pub const OFF_INDEX: u32 = 0x30;
const OFF_CALLBACKS: u32 = 0x38;
const OFF_IMPORT_DESC: u32 = 0x60;
const OFF_INT: u32 = 0x88;
const OFF_IAT: u32 = 0x98;
const OFF_HINTNAME: u32 = 0xA8;
const OFF_DLLNAME: u32 = 0xB8;
const OFF_TEMPLATE: u32 = 0x100;

#[derive(Clone, Copy, Debug)]
pub struct CarrierOpts {
    pub entry_point: bool,
    pub with_import: bool,
}

impl Default for CarrierOpts {
    fn default() -> Self {
        CarrierOpts { entry_point: true, with_import: true }
    }
}

fn put16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

/// TLS block size the carrier must provide for a game template of `total_len` bytes.
pub fn carrier_template_size_for(total_len: u64) -> u32 {
    let n = total_len.max(16);
    (((n + 15) & !15).min(u32::MAX as u64 - 0x10000)) as u32
}

pub fn build_carrier_dll(template_size: u32) -> Vec<u8> {
    build_carrier_dll_with(template_size, CarrierOpts::default())
}

/// Build a minimal x64 DLL whose TLS block is exactly `template_size` bytes.
pub fn build_carrier_dll_with(template_size: u32, opts: CarrierOpts) -> Vec<u8> {
    let template_size = template_size.max(16);
    let data_vsize = OFF_TEMPLATE + template_size;
    let size_of_image = (DATA_RVA + data_vsize + 0xFFF) & !0xFFF;
    let mut f = vec![0u8; FILE_ALIGN * 4];
    // DOS header
    f[0] = b'M';
    f[1] = b'Z';
    put32(&mut f, 0x3C, 0x40);
    // PE signature + COFF
    let nt = 0x40;
    f[nt..nt + 4].copy_from_slice(b"PE\0\0");
    let coff = nt + 4;
    put16(&mut f, coff, 0x8664);
    put16(&mut f, coff + 2, 3); // NumberOfSections
    put16(&mut f, coff + 16, 240); // SizeOfOptionalHeader
    put16(&mut f, coff + 18, 0x0002 | 0x0020 | 0x2000); // EXECUTABLE | LARGE_ADDRESS_AWARE | DLL
    // Optional header (PE32+)
    let opt = coff + 20;
    put16(&mut f, opt, 0x20B);
    f[opt + 2] = 14;
    put32(&mut f, opt + 4, FILE_ALIGN as u32); // SizeOfCode
    put32(&mut f, opt + 8, (FILE_ALIGN * 2) as u32); // SizeOfInitializedData
    put32(&mut f, opt + 16, if opts.entry_point { TEXT_RVA } else { 0 });
    put32(&mut f, opt + 20, TEXT_RVA); // BaseOfCode
    put64(&mut f, opt + 24, CARRIER_BASE);
    put32(&mut f, opt + 32, 0x1000); // SectionAlignment
    put32(&mut f, opt + 36, FILE_ALIGN as u32);
    put16(&mut f, opt + 40, 6); // MajorOperatingSystemVersion
    put16(&mut f, opt + 48, 6); // MajorSubsystemVersion
    put32(&mut f, opt + 56, size_of_image);
    put32(&mut f, opt + 60, FILE_ALIGN as u32); // SizeOfHeaders
    put16(&mut f, opt + 68, 2); // GUI
    put16(&mut f, opt + 70, 0x0020 | 0x0040 | 0x0100); // HIGH_ENTROPY_VA | DYNAMIC_BASE | NX_COMPAT
    put64(&mut f, opt + 72, 0x10_0000);
    put64(&mut f, opt + 80, 0x1000);
    put64(&mut f, opt + 88, 0x10_0000);
    put64(&mut f, opt + 96, 0x1000);
    put32(&mut f, opt + 108, 16);
    let dirs = opt + 112;
    let reloc_size: u32 = 8 + 4 * 2;
    if opts.with_import {
        put32(&mut f, dirs + 8, DATA_RVA + OFF_IMPORT_DESC);
        put32(&mut f, dirs + 8 + 4, 40);
        put32(&mut f, dirs + 12 * 8, DATA_RVA + OFF_IAT);
        put32(&mut f, dirs + 12 * 8 + 4, 16);
    }
    put32(&mut f, dirs + 5 * 8, RELOC_RVA);
    put32(&mut f, dirs + 5 * 8 + 4, reloc_size);
    put32(&mut f, dirs + 9 * 8, DATA_RVA + OFF_TLS_DIR);
    put32(&mut f, dirs + 9 * 8 + 4, 40);
    // Section table: .text, .reloc, .data (last – it grows with the template)
    let mut sec = opt + 240;
    let mut section = |f: &mut Vec<u8>, name: &[u8], vsize: u32, rva: u32, raw: usize, ch: u32| {
        f[sec..sec + name.len()].copy_from_slice(name);
        put32(f, sec + 8, vsize);
        put32(f, sec + 12, rva);
        put32(f, sec + 16, FILE_ALIGN as u32);
        put32(f, sec + 20, raw as u32);
        put32(f, sec + 36, ch);
        sec += 40;
    };
    section(&mut f, b".text", 6, TEXT_RVA, FILE_ALIGN, 0x0000_0020 | 0x2000_0000 | 0x4000_0000);
    section(&mut f, b".reloc", reloc_size, RELOC_RVA, FILE_ALIGN * 2, 0x0000_0040 | 0x0200_0000 | 0x4000_0000);
    section(&mut f, b".data", data_vsize, DATA_RVA, FILE_ALIGN * 3, 0x0000_0040 | 0x4000_0000 | 0x8000_0000);
    // .text: DllMain = mov eax, 1 ; ret
    f[FILE_ALIGN..FILE_ALIGN + 6].copy_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3]);
    // .reloc: four DIR64 entries for the TLS directory's VA fields
    let r = FILE_ALIGN * 2;
    put32(&mut f, r, DATA_RVA);
    put32(&mut f, r + 4, reloc_size);
    for i in 0..4u16 {
        put16(&mut f, r + 8 + i as usize * 2, (10 << 12) | (OFF_TLS_DIR as u16 + i * 8));
    }
    // .data: IMAGE_TLS_DIRECTORY64
    let d = FILE_ALIGN * 3;
    let va = |off: u32| CARRIER_BASE + (DATA_RVA + off) as u64;
    put64(&mut f, d + OFF_TLS_DIR as usize, va(OFF_TEMPLATE));
    put64(&mut f, d + OFF_TLS_DIR as usize + 8, va(OFF_TEMPLATE + template_size));
    put64(&mut f, d + OFF_TLS_DIR as usize + 16, va(OFF_INDEX));
    put64(&mut f, d + OFF_TLS_DIR as usize + 24, va(OFF_CALLBACKS));
    // SizeOfZeroFill = 0, Characteristics = 0
    put32(&mut f, d + OFF_INDEX as usize, NO_INDEX); // sentinel: was a slot assigned?
    if opts.with_import {
        let desc = d + OFF_IMPORT_DESC as usize;
        put32(&mut f, desc, DATA_RVA + OFF_INT);
        put32(&mut f, desc + 12, DATA_RVA + OFF_DLLNAME);
        put32(&mut f, desc + 16, DATA_RVA + OFF_IAT);
        put64(&mut f, d + OFF_INT as usize, (DATA_RVA + OFF_HINTNAME) as u64);
        put64(&mut f, d + OFF_IAT as usize, (DATA_RVA + OFF_HINTNAME) as u64);
        let hn = d + OFF_HINTNAME as usize + 2;
        f[hn..hn + 12].copy_from_slice(b"GetTickCount");
        let dn = d + OFF_DLLNAME as usize;
        f[dn..dn + 12].copy_from_slice(b"KERNEL32.dll");
    }
    f
}

// ---------------------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------------------

pub struct TlsImage {
    pub name: String,
    pub template: usize,
    pub template_len: usize,
    pub index: AtomicU32,
}

static IMAGES: RwLock<Vec<Arc<TlsImage>>> = RwLock::new(Vec::new());

#[repr(C)]
struct ThreadBasicInformation {
    exit_status: i32,
    teb: *mut u8,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn NtQueryInformationThread(
        thread: *mut core::ffi::c_void,
        class: u32,
        info: *mut core::ffi::c_void,
        len: u32,
        ret: *mut u32,
    ) -> i32;
}

#[inline(always)]
pub unsafe fn current_teb() -> *mut u8 {
    let t: *mut u8;
    core::arch::asm!("mov {}, gs:[0x30]", out(reg) t, options(nostack, readonly, preserves_flags));
    t
}

/// TEB.ThreadLocalStoragePointer (offset 0x58 on x64) [index].
unsafe fn tls_block_of(teb: *mut u8, index: u32) -> *mut u8 {
    if teb.is_null() {
        return std::ptr::null_mut();
    }
    let arr = *(teb.add(0x58) as *const *mut *mut u8);
    if arr.is_null() {
        return std::ptr::null_mut();
    }
    *arr.add(index as usize)
}

unsafe fn fill_block(block: *mut u8, img: &TlsImage) {
    if block.is_null() || img.template_len == 0 {
        return;
    }
    std::ptr::copy_nonoverlapping(img.template as *const u8, block, img.template_len);
}

// ---------------------------------------------------------------------------------------
// slot 0 for the title's executable
// ---------------------------------------------------------------------------------------

/// The slot xcl.exe's own TLS moved to when the title's executable took slot 0 (NO_INDEX: not
/// moved).
static XCL_SLOT: AtomicU32 = AtomicU32::new(NO_INDEX);
/// TEBs of the threads whose slots were exchanged by `take_slot_zero` itself (their TLS
/// callback, if it still comes, leaves them as they are); held during the exchange
static MOVED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// xcl.exe's `_tls_index` (its TLS directory's AddressOfIndex)
unsafe fn own_tls_index() -> Option<*mut u32> {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    let base = GetModuleHandleW(std::ptr::null()) as usize;
    if base == 0 {
        return None;
    }
    let nt = base + *((base + 0x3C) as *const u32) as usize;
    // PE32+: the optional header at +24, its data directories at +112; TLS is #9
    let rva = *((nt + 24 + 112 + 9 * 8) as *const u32) as usize;
    if rva == 0 {
        return None;
    }
    let index = *((base + rva + 16) as *const u64) as usize;
    (index != 0).then_some(index as *mut u32)
}

/// The TEBs of the process's threads (the current one first).
unsafe fn thread_tebs() -> Vec<*mut u8> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId, OpenThread, THREAD_QUERY_INFORMATION};
    let mut out = vec![current_teb()];
    let (pid, tid) = (GetCurrentProcessId(), GetCurrentThreadId());
    let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if snap == INVALID_HANDLE_VALUE {
        return out;
    }
    let mut te: THREADENTRY32 = std::mem::zeroed();
    te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
    let mut ok = Thread32First(snap, &mut te) != 0;
    while ok {
        if te.th32OwnerProcessID == pid && te.th32ThreadID != tid {
            let h = OpenThread(THREAD_QUERY_INFORMATION, 0, te.th32ThreadID);
            if !h.is_null() {
                let mut tbi: ThreadBasicInformation = std::mem::zeroed();
                if NtQueryInformationThread(h, 0, &mut tbi as *mut _ as *mut _, std::mem::size_of::<ThreadBasicInformation>() as u32, std::ptr::null_mut()) >= 0 && !tbi.teb.is_null() {
                    out.push(tbi.teb);
                }
                CloseHandle(h);
            }
        }
        ok = Thread32Next(snap, &mut te) != 0;
    }
    CloseHandle(snap);
    out
}

/// The title's executable gets TLS slot 0, as on Windows (and the console) the process's
/// executable has it: code built with /GA reads `ThreadLocalStoragePointer[0]` directly, without
/// `_tls_index` (Fortnite: the guard of its thread-safe statics, so its allocator was never
/// made). Slot 0 is xcl.exe's own: its TLS moves to the carrier's slot `k` (all of its code
/// goes through `_tls_index`), the carrier's blocks – the title's – to slot 0. Running threads
/// first get xcl.exe's block in both slots and only then the title's in slot 0, so a TLS
/// access in flight with either index stays right.
pub unsafe fn take_slot_zero(img_name: &str, title_index: *mut u32, k: u32) -> Result<(), String> {
    use std::sync::atomic::fence;
    let own = own_tls_index().ok_or("xcl.exe has no TLS directory")?;
    if *own != 0 {
        return Err(format!("xcl.exe's TLS index is {}, not 0", *own));
    }
    let mut moved = MOVED.lock().map_err(|_| "TLS lock poisoned")?;
    let tebs = thread_tebs();
    // 1. slot k = slot 0 (xcl.exe's block) in every thread; the carrier blocks kept
    let mut carriers: Vec<(*mut *mut u8, *mut u8)> = Vec::new();
    for &teb in &tebs {
        let arr = *(teb.add(0x58) as *const *mut *mut u8);
        if arr.is_null() {
            continue;
        }
        carriers.push((arr, *arr.add(k as usize)));
        *arr.add(k as usize) = *arr;
        moved.push(teb as usize);
    }
    fence(Ordering::SeqCst);
    XCL_SLOT.store(k, Ordering::Release);
    std::ptr::write_volatile(own, k);
    fence(Ordering::SeqCst);
    // 2. accesses that read index 0 before finish, then slot 0 = the title's block
    windows_sys::Win32::System::Threading::Sleep(20);
    for (arr, block) in carriers {
        *arr = block;
    }
    if !title_index.is_null() {
        *title_index = 0;
    }
    if let Ok(images) = IMAGES.read() {
        for img in images.iter().filter(|i| i.name == img_name) {
            img.index.store(0, Ordering::Release);
        }
    }
    drop(moved);
    crate::xlog_info!("TLS: {img_name} has slot 0 (as a process's executable); xcl.exe's own TLS is in slot {k} – {} threads", tebs.len());
    Ok(())
}

/// A new thread: ntdll gave slot 0 xcl.exe's block and slot k the carrier's – exchanged before
/// anything here reads TLS.
unsafe fn exchange_for_new_thread(teb: *mut u8) {
    let k = XCL_SLOT.load(Ordering::Acquire);
    if k == NO_INDEX {
        return;
    }
    let Ok(mut moved) = MOVED.lock() else { return };
    if let Some(i) = moved.iter().position(|&t| t == teb as usize) {
        // exchanged by take_slot_zero before it ran: done (and the TEB may come back later)
        moved.swap_remove(i);
        return;
    }
    let arr = *(teb.add(0x58) as *const *mut *mut u8);
    if arr.is_null() {
        return;
    }
    let x = *arr;
    *arr = *arr.add(k as usize);
    *arr.add(k as usize) = x;
}

/// A thread ends: its TEB may come back for a new thread, which needs the exchange.
pub fn on_thread_detach() {
    if XCL_SLOT.load(Ordering::Acquire) == NO_INDEX {
        return;
    }
    let teb = unsafe { current_teb() } as usize;
    if let Ok(mut moved) = MOVED.lock() {
        moved.retain(|&t| t != teb);
    }
}

/// Called for every new thread from xcl.exe's TLS callback (DLL_THREAD_ATTACH).
pub unsafe fn on_thread_attach() {
    let teb = current_teb();
    exchange_for_new_thread(teb);
    let Ok(images) = IMAGES.try_read() else { return };
    for img in images.iter() {
        let idx = img.index.load(Ordering::Acquire);
        if idx != NO_INDEX {
            fill_block(tls_block_of(teb, idx), img);
        }
    }
}

unsafe fn fill_existing_threads(img: &TlsImage, index: u32) -> usize {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessId, GetCurrentThreadId, OpenThread, THREAD_QUERY_INFORMATION,
    };
    let me_pid = GetCurrentProcessId();
    let me_tid = GetCurrentThreadId();
    fill_block(tls_block_of(current_teb(), index), img);
    let mut filled = 1usize;
    let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if snap == INVALID_HANDLE_VALUE {
        return filled;
    }
    let mut te: THREADENTRY32 = std::mem::zeroed();
    te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
    let mut ok = Thread32First(snap, &mut te) != 0;
    while ok {
        if te.th32OwnerProcessID == me_pid && te.th32ThreadID != me_tid {
            let h = OpenThread(THREAD_QUERY_INFORMATION, 0, te.th32ThreadID);
            if !h.is_null() {
                let mut tbi: ThreadBasicInformation = std::mem::zeroed();
                let st = NtQueryInformationThread(
                    h,
                    0,
                    &mut tbi as *mut _ as *mut _,
                    std::mem::size_of::<ThreadBasicInformation>() as u32,
                    std::ptr::null_mut(),
                );
                if st >= 0 {
                    let b = tls_block_of(tbi.teb, index);
                    if !b.is_null() {
                        fill_block(b, img);
                        filled += 1;
                    }
                }
                CloseHandle(h);
            }
        }
        ok = Thread32Next(snap, &mut te) != 0;
    }
    CloseHandle(snap);
    filled
}

static SEQ: AtomicU32 = AtomicU32::new(0);

/// Create + load a carrier for `img_name` and wire the image's TLS to it.
///
/// * `template`/`template_len` – the image's initialised TLS data (already relocated),
/// * `total_len` – template + zero fill,
/// * `index_ptr` – the image's `_tls_index` variable.
pub unsafe fn install(
    img_name: &str,
    template: usize,
    template_len: usize,
    total_len: u64,
    index_ptr: *mut u32,
    cache_dir: &Path,
) -> Result<u32, String> {
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
    let img = Arc::new(TlsImage {
        name: img_name.to_string(),
        template,
        template_len,
        index: AtomicU32::new(NO_INDEX),
    });
    IMAGES.write().map_err(|_| "TLS lock poisoned")?.push(img.clone());

    let size = carrier_template_size_for(total_len);
    let bytes = build_carrier_dll(size);
    std::fs::create_dir_all(cache_dir).map_err(|e| format!("cache dir: {e}"))?;
    let safe: String = img_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let path: PathBuf = cache_dir.join(format!(
        "xcl_tls_{}_{}_{}.dll",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed),
        safe
    ));
    std::fs::write(&path, &bytes).map_err(|e| format!("writing the TLS carrier {}: {e}", path.display()))?;
    let h = LoadLibraryW(wide(&path.to_string_lossy()).as_ptr());
    if h.is_null() {
        return Err(format!(
            "LoadLibrary of the TLS carrier failed (error {})",
            windows_sys::Win32::Foundation::GetLastError()
        ));
    }
    let index = *((h as usize + (DATA_RVA + OFF_INDEX) as usize) as *const u32);
    if index == NO_INDEX {
        return Err("Windows did not allocate a TLS slot for the carrier (TLS directory ignored)".into());
    }
    if !index_ptr.is_null() {
        *index_ptr = index;
    }
    img.index.store(index, Ordering::Release);
    let n = fill_existing_threads(&img, index);
    crate::xlog_info!(
        "TLS: {img_name}: template {template_len} B, block {size} B, index {index}, threads initialized: {n}"
    );
    Ok(index)
}

/// Best-effort removal of carrier files left behind by previous runs.
pub fn cleanup_cache(cache_dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(cache_dir) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("xcl_tls_") && n.ends_with(".dll") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pe::{Layout, Pe, IMAGE_REL_BASED_DIR64};

    #[test]
    fn carrier_is_a_valid_pe_with_tls_imports_and_relocs() {
        let bytes = build_carrier_dll(0x3000);
        let pe = Pe::parse(&bytes, Layout::File).unwrap();
        assert!(pe.is_dll());
        assert!(pe.has_relocs());
        let tls = pe.tls().unwrap().unwrap();
        assert_eq!(tls.template_len(), 0x3000);
        assert_eq!(tls.size_of_zero_fill, 0);
        assert_eq!(pe.va_to_rva(tls.address_of_index), Some(DATA_RVA + OFF_INDEX));
        let relocs = pe.relocations().unwrap();
        assert_eq!(relocs.len(), 4);
        assert!(relocs.iter().all(|r| r.kind == IMAGE_REL_BASED_DIR64));
        let imps = pe.imports().unwrap();
        assert_eq!(imps.len(), 1);
        assert_eq!(imps[0].dll, "KERNEL32.dll");
        assert!(pe.size_of_image >= DATA_RVA + OFF_TEMPLATE + 0x3000);
    }

    #[test]
    fn template_size_rounding() {
        assert_eq!(carrier_template_size_for(0), 16);
        assert_eq!(carrier_template_size_for(17), 32);
        assert_eq!(carrier_template_size_for(0x1000), 0x1000);
    }

    /// Diagnostic: which carrier variants does this Windows build honour?
    /// `cargo test -p xcl carrier_variants_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn carrier_variants_probe() {
        use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
        let dir = std::env::temp_dir().join(format!("xcl_tls_probe_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (i, (entry, imp)) in [(false, false), (true, false), (false, true), (true, true)].into_iter().enumerate() {
            let bytes = build_carrier_dll_with(0x1000, CarrierOpts { entry_point: entry, with_import: imp });
            let p = dir.join(format!("probe{i}.dll"));
            std::fs::write(&p, bytes).unwrap();
            let h = unsafe { LoadLibraryW(wide(&p.to_string_lossy()).as_ptr()) };
            let idx = (!h.is_null())
                .then(|| unsafe { *((h as usize + (DATA_RVA + OFF_INDEX) as usize) as *const u32) });
            println!("entry={entry} import={imp} -> loaded={} index={idx:?}", !h.is_null());
        }
    }

    #[test]
    fn windows_loads_carrier_and_allocates_full_block_for_all_threads() {
        use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapValidate};
        let dir = std::env::temp_dir().join(format!("xcl_tls_test_{}", std::process::id()));
        let mut template = [0u8; 64];
        for (i, b) in template.iter_mut().enumerate() {
            *b = i as u8 ^ 0x5A;
        }
        let template: &'static [u8; 64] = Box::leak(Box::new(template));
        let mut index_var: u32 = 0xDEAD;
        let total = 0x2000u64;
        let idx = unsafe {
            install("unit", template.as_ptr() as usize, 64, total, &mut index_var, &dir).unwrap()
        };
        assert_eq!(index_var, idx);
        unsafe {
            let b = tls_block_of(current_teb(), idx);
            assert!(!b.is_null());
            assert_eq!(std::slice::from_raw_parts(b, 64), &template[..]);
            // the whole requested size is ours: writing it must not corrupt the heap
            std::ptr::write_bytes(b.add(64), 0xCC, total as usize - 64);
            assert_ne!(HeapValidate(GetProcessHeap(), 0, std::ptr::null()), 0);
            std::ptr::write_bytes(b.add(64), 0, total as usize - 64);
        }
        let ok = std::thread::spawn(move || unsafe {
            on_thread_attach();
            let b = tls_block_of(current_teb(), idx);
            !b.is_null() && std::slice::from_raw_parts(b, 64) == &template[..]
        })
        .join()
        .unwrap();
        assert!(ok);
    }

    #[test]
    fn install_without_an_index_variable() {
        let dir = std::env::temp_dir().join(format!("xcl_tls_test_noidx_{}", std::process::id()));
        let template: &'static [u8; 16] = Box::leak(Box::new([0xA5u8; 16]));
        let idx = unsafe {
            install("unit-noidx", template.as_ptr() as usize, 16, 0x1000, std::ptr::null_mut(), &dir).unwrap()
        };
        assert_ne!(idx, NO_INDEX);
        unsafe {
            let b = tls_block_of(current_teb(), idx);
            assert!(!b.is_null());
            assert_eq!(std::slice::from_raw_parts(b, 16), &template[..]);
        }
    }
}
