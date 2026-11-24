//! The XCL PE loader: maps console executables/DLLs into the current process and wires
//! them to HLE modules, game DLLs and real Windows DLLs (design §5.1).

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE,
    PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
};

use crate::modules::{self, MappedModule};
use crate::pe::{self, Layout, OwnedExports, Pe};
use crate::resolver;

static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();
/// Serialises loads so that two threads calling LoadLibrary on the same game DLL do not
/// map it twice. Re-entrant loads (import resolution mapping dependencies) happen on the
/// same thread, so a re-entrant flag is used instead of holding the mutex across them.
static LOAD_LOCK: Mutex<()> = Mutex::new(());
thread_local! {
    static IN_LOAD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
/// Set once the main image is ready; thread notifications are ignored before that.
static READY: AtomicBool = AtomicBool::new(false);

pub fn set_cache_dir(p: PathBuf) {
    let _ = CACHE_DIR.set(p);
}

fn cache_dir() -> PathBuf {
    CACHE_DIR
        .get()
        .cloned()
        .unwrap_or_else(|| std::env::temp_dir().join("xcl_cache"))
}

fn section_protection(ch: u32) -> u32 {
    let x = ch & pe::SCN_MEM_EXECUTE != 0;
    let r = ch & pe::SCN_MEM_READ != 0;
    let w = ch & pe::SCN_MEM_WRITE != 0;
    match (x, r, w) {
        (true, _, true) => PAGE_EXECUTE_READWRITE,
        (true, true, false) => PAGE_EXECUTE_READ,
        (true, false, false) => PAGE_EXECUTE,
        (false, _, true) => PAGE_READWRITE,
        (false, true, false) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

type TlsCallback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
type ExeEntry = unsafe extern "system" fn(*mut c_void) -> u32;

/// Map an image, apply relocations, resolve imports, set up TLS / exception data and
/// page protections. Does NOT run any code of the image.
pub fn map_image(path: &Path) -> Result<Arc<MappedModule>, String> {
    map_image_ex(path, false)
}

fn map_image_ex(path: &Path, is_main: bool) -> Result<Arc<MappedModule>, String> {
    let file = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let pe = Pe::parse(&file, Layout::File).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = crate::util::dll_key(&path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    let size = pe.size_of_image as usize;

    // 1. reserve + commit
    let mut base = unsafe {
        VirtualAlloc(pe.image_base as *const c_void, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
    } as usize;
    if base == 0 {
        if !pe.has_relocs() {
            return Err(format!(
                "{name}: preferred address {:#x} is taken and the image has no relocations",
                pe.image_base
            ));
        }
        base = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        if base == 0 {
            return Err(format!("{name}: VirtualAlloc({size:#x}) failed"));
        }
    }
    let delta = base as i64 - pe.image_base as i64;
    let fail = |msg: String| {
        unsafe { VirtualFree(base as *mut c_void, 0, MEM_RELEASE) };
        Err(msg)
    };

    // 2. headers + sections
    unsafe {
        let hdr = (pe.size_of_headers as usize).min(file.len()).min(size);
        std::ptr::copy_nonoverlapping(file.as_ptr(), base as *mut u8, hdr);
        for s in &pe.sections {
            let raw = (s.raw_size as usize).min(s.mapped_size() as usize);
            if raw == 0 {
                continue;
            }
            let src = s.raw_ptr as usize;
            if src + raw > file.len() || s.va as usize + raw > size {
                return fail(format!("{name}: section {} outside the file/image", s.name));
            }
            std::ptr::copy_nonoverlapping(file.as_ptr().add(src), (base + s.va as usize) as *mut u8, raw);
        }
    }

    // 3. relocations
    if delta != 0 {
        let relocs = match pe.relocations() {
            Ok(r) => r,
            Err(e) => return fail(format!("{name}: relocations: {e}")),
        };
        for r in relocs {
            let p = base + r.rva as usize;
            unsafe {
                match r.kind {
                    pe::IMAGE_REL_BASED_DIR64 => {
                        let v = (p as *mut u64).read_unaligned();
                        (p as *mut u64).write_unaligned((v as i64 + delta) as u64);
                    }
                    pe::IMAGE_REL_BASED_HIGHLOW => {
                        let v = (p as *mut u32).read_unaligned();
                        (p as *mut u32).write_unaligned((v as i64 + delta) as u32);
                    }
                    k => crate::xlog_warn!("{name}: unsupported relocation type {k} @ {:#x}", r.rva),
                }
            }
        }
    }

    let exports = OwnedExports::from_pe(&pe).unwrap_or_default();
    let module = Arc::new(MappedModule {
        name: name.clone(),
        path: path.to_path_buf(),
        base,
        size,
        entry_rva: pe.entry_rva,
        is_dll: pe.is_dll(),
        exports,
        tls_callbacks: Mutex::new(Vec::new()),
        no_thread_calls: AtomicBool::new(false),
        initialized: AtomicBool::new(false),
        stack_reserve: pe.stack_reserve,
    });
    // registered before import resolution so cyclic game-DLL imports terminate
    modules::register_mapped(module.clone());
    if is_main {
        // game DLLs initialised during import resolution may already ask for
        // GetModuleHandle(NULL) / GetModuleFileName(NULL)
        modules::set_main(module.clone());
    }
    crate::xlog_info!(
        "loaded {name} @ {base:#x} (size {size:#x}, preferred {:#x}{})",
        pe.image_base,
        if delta != 0 { ", relocated" } else { "" }
    );

    // 4. imports (normal + delay-load, resolved eagerly). A failure from here on takes the
    // module out of the registry again: a later load must not find it (its memory freed, or
    // never initialised) and take it for loaded
    let mut all = match pe.imports() {
        Ok(v) => v,
        Err(e) => {
            // nothing was resolved against it yet: its memory goes too
            modules::unregister_mapped(base);
            return fail(format!("{name}: import table: {e}"));
        }
    };
    match pe.delay_imports() {
        Ok(v) => all.extend(v),
        Err(e) => crate::xlog_warn!("{name}: delay-imports: {e}"),
    }
    for dll in &all {
        for e in &dll.entries {
            let addr = resolver::resolve_import(&dll.dll, &e.what, &name, dll.delay);
            unsafe { ((base + e.iat_rva as usize) as *mut u64).write_unaligned(addr as u64) };
        }
        if dll.delay && dll.module_handle_rva != 0 {
            let h = resolver::load_library(&dll.dll).unwrap_or(1);
            unsafe { ((base + dll.module_handle_rva as usize) as *mut u64).write_unaligned(h as u64) };
        }
    }

    // 5. static TLS
    if let Ok(Some(t)) = pe.tls() {
        let adj = |va: u64| (va as i64 + delta) as usize;
        let template = adj(t.start_raw);
        let template_len = t.template_len() as usize;
        let index_ptr = if t.address_of_index == 0 { std::ptr::null_mut() } else { adj(t.address_of_index) as *mut u32 };
        unsafe {
            // game DLLs loaded while its imports were resolved may point into it: its memory stays
            tls_install(&module, template, template_len, t.total_len(), index_ptr).inspect_err(|_| modules::unregister_mapped(base))?;
            if t.address_of_callbacks != 0 {
                let mut cb = adj(t.address_of_callbacks) as *const u64;
                let mut list = module.tls_callbacks.lock().unwrap();
                while *cb != 0 {
                    list.push(*cb as usize);
                    cb = cb.add(1);
                }
            }
        }
    }

    // 6. page protections
    unsafe {
        let mut old = 0u32;
        VirtualProtect(base as *const c_void, pe.size_of_headers as usize, PAGE_READONLY, &mut old);
        for s in &pe.sections {
            let len = ((s.mapped_size() as usize) + 0xFFF) & !0xFFF;
            if len == 0 {
                continue;
            }
            VirtualProtect((base + s.va as usize) as *const c_void, len, section_protection(s.characteristics), &mut old);
        }
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            base as *const c_void,
            size,
        );
    }

    // 7. exception data
    let ex = pe.dir(pe::DIR_EXCEPTION);
    if ex.rva != 0 && ex.size != 0 {
        crate::eh::register_function_table(base, ex.rva, ex.size);
    }
    // 8. stack-cookie failures reported before the process ends
    for s in pe.sections.iter().filter(|s| s.characteristics & 0x2000_0000 != 0) {
        let len = (s.mapped_size() as usize).min(size.saturating_sub(s.va as usize));
        unsafe { crate::eh::hook_gs_failure(base + s.va as usize, len, &name) };
    }
    Ok(module)
}

unsafe fn tls_install(m: &MappedModule, template: usize, len: usize, total: u64, index_ptr: *mut u32) -> Result<(), String> {
    let k = crate::tls::install(&m.name, template, len, total, index_ptr, &cache_dir())?;
    // the title's executable: slot 0, as a process's executable has it
    if !m.is_dll {
        if let Err(e) = crate::tls::take_slot_zero(&m.name, index_ptr, k) {
            crate::xlog_warn!("TLS: {} keeps slot {k} ({e}) – code built with /GA reads slot 0", m.name);
        }
    }
    Ok(())
}

unsafe fn run_tls_callbacks(m: &MappedModule, reason: u32) {
    let cbs = m.tls_callbacks.lock().map(|v| v.clone()).unwrap_or_default();
    for cb in cbs {
        let f: TlsCallback = std::mem::transmute(cb);
        f(m.base as *mut c_void, reason, std::ptr::null_mut());
    }
}

/// Run TLS callbacks (PROCESS_ATTACH) and DllMain of a freshly mapped DLL.
pub fn init_dll(m: &Arc<MappedModule>) -> Result<(), String> {
    unsafe {
        run_tls_callbacks(m, 1);
        if m.entry_rva != 0 {
            let f: DllMain = std::mem::transmute(m.base + m.entry_rva as usize);
            crate::xlog_info!("DllMain({}, PROCESS_ATTACH)", m.name);
            if f(m.base as *mut c_void, 1, std::ptr::null_mut()) == 0 {
                return Err(format!("DllMain({}) returned FALSE", m.name));
            }
        }
    }
    m.initialized.store(true, Ordering::Release);
    Ok(())
}

struct LoadGuard;
impl LoadGuard {
    fn enter() -> (LoadGuard, Option<std::sync::MutexGuard<'static, ()>>) {
        let outer = IN_LOAD.with(|c| {
            let v = c.get();
            c.set(v + 1);
            v == 0
        });
        let g = if outer { Some(LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner())) } else { None };
        (LoadGuard, g)
    }
}
impl Drop for LoadGuard {
    fn drop(&mut self) {
        IN_LOAD.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Map + initialise a DLL that ships with the game (idempotent).
pub fn load_game_dll(path: &Path) -> Result<Arc<MappedModule>, String> {
    let key = crate::util::dll_key(&path.to_string_lossy());
    let (_g, _lock) = LoadGuard::enter();
    if let Some(m) = modules::find_mapped_by_name(&key) {
        return Ok(m);
    }
    let m = map_image(path)?;
    if !m.is_dll {
        crate::xlog_warn!("{key} is not a DLL but is being loaded as a library");
    }
    if let Err(e) = init_dll(&m) {
        // as Windows' loader: a DLL whose DllMain fails is not loaded, a later load maps it and
        // runs DllMain again (here its memory stays: what DllMain ran may have left threads or
        // callbacks in it)
        modules::unregister_mapped(m.base);
        return Err(e);
    }
    Ok(m)
}

/// Map the title's main executable.
pub fn load_main(path: &Path) -> Result<Arc<MappedModule>, String> {
    let (_g, _lock) = LoadGuard::enter();
    map_image_ex(path, true)
}

#[inline(always)]
unsafe fn peb() -> *mut u8 {
    let p: *mut u8;
    core::arch::asm!("mov {}, gs:[0x60]", out(reg) p, options(nostack, readonly, preserves_flags));
    p
}

struct EntryArgs {
    entry: usize,
}

unsafe extern "system" fn main_thread(param: *mut c_void) -> u32 {
    let args = Box::from_raw(param as *mut EntryArgs);
    let f: ExeEntry = std::mem::transmute(args.entry);
    f(peb() as *mut c_void)
}

/// Make the game the "main image" of the process and run its entry point on a thread
/// with the stack size its PE header asks for. Usually never returns (the game calls
/// ExitProcess); if the entry returns, its value is returned.
pub fn run_main(m: &Arc<MappedModule>) -> u32 {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        CreateThread, GetExitCodeThread, WaitForSingleObject, INFINITE, STACK_SIZE_PARAM_IS_A_RESERVATION,
    };
    unsafe {
        // PEB.ImageBaseAddress → GetModuleHandle(NULL), resources with NULL module, …
        if crate::util::under_wine() {
            crate::xlog_info!("Wine: PEB.ImageBaseAddress left alone (its loader needs the real main image there)");
        } else {
            *(peb().add(0x10) as *mut usize) = m.base;
        }
        run_tls_callbacks(m, 1);
    }
    READY.store(true, Ordering::Release);
    let stack = (m.stack_reserve as usize).max(1 << 20);
    let args = Box::into_raw(Box::new(EntryArgs { entry: m.base + m.entry_rva as usize }));
    crate::xlog_info!(
        "start {}: entry point {} , stack {:#x}",
        m.name,
        modules::describe_address(m.base + m.entry_rva as usize),
        stack
    );
    crate::log::flush();
    unsafe {
        let h = CreateThread(
            std::ptr::null(),
            stack,
            Some(main_thread),
            args as *const c_void,
            STACK_SIZE_PARAM_IS_A_RESERVATION,
            std::ptr::null_mut(),
        );
        if h.is_null() {
            crate::xlog_error!("CreateThread for the game thread failed");
            return 0xC000_0001;
        }
        let mut code = 0u32;
        if WaitForSingleObject(h, INFINITE) == WAIT_OBJECT_0 {
            GetExitCodeThread(h, &mut code);
        }
        CloseHandle(h);
        code
    }
}

/// Called from xcl.exe's own TLS callback for every thread event in the process.
pub fn on_tls_event(reason: u32) {
    match reason {
        2 => unsafe {
            // DLL_THREAD_ATTACH: game TLS blocks first, then game callbacks/DllMains.
            crate::tls::on_thread_attach();
            if !READY.load(Ordering::Acquire) {
                return;
            }
            for m in modules::mapped_modules() {
                run_tls_callbacks(&m, 2);
                if m.is_dll && m.initialized.load(Ordering::Acquire) && !m.no_thread_calls.load(Ordering::Relaxed) && m.entry_rva != 0 {
                    let f: DllMain = std::mem::transmute(m.base + m.entry_rva as usize);
                    f(m.base as *mut c_void, 2, std::ptr::null_mut());
                }
            }
        },
        3 => unsafe {
            crate::tls::on_thread_detach();
            if !READY.load(Ordering::Acquire) {
                return;
            }
            for m in modules::mapped_modules().into_iter().rev() {
                if m.is_dll && m.initialized.load(Ordering::Acquire) && !m.no_thread_calls.load(Ordering::Relaxed) && m.entry_rva != 0 {
                    let f: DllMain = std::mem::transmute(m.base + m.entry_rva as usize);
                    f(m.base as *mut c_void, 3, std::ptr::null_mut());
                }
                run_tls_callbacks(&m, 3);
            }
        },
        0 => {
            // DLL_PROCESS_DETACH (ExitProcess): write the report while everything is alive.
            crate::report::write_final(None);
            crate::log::flush();
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PE32+ DLL of one section (.text at RVA 0x1000) holding `code`, its entry point at the
    /// start of it, preferred at `image_base` and relocatable (one empty block at RVA 0x1100);
    /// `import_dir`: (RVA, size) of the import directory, if any.
    fn tiny_dll(image_base: u64, code: &[u8], import_dir: Option<(u32, u32)>) -> Vec<u8> {
        let mut f = vec![0u8; 0x400];
        let put16 = |f: &mut Vec<u8>, at: usize, v: u16| f[at..at + 2].copy_from_slice(&v.to_le_bytes());
        let put32 = |f: &mut Vec<u8>, at: usize, v: u32| f[at..at + 4].copy_from_slice(&v.to_le_bytes());
        f[0..2].copy_from_slice(b"MZ");
        put32(&mut f, 0x3C, 0x40);
        f[0x40..0x44].copy_from_slice(b"PE\0\0");
        // COFF: x64, 1 section, optional header 0xF0, executable | large address aware | DLL
        put16(&mut f, 0x44, 0x8664);
        put16(&mut f, 0x46, 1);
        put16(&mut f, 0x54, 0xF0);
        put16(&mut f, 0x56, 0x2022);
        let o = 0x58;
        put16(&mut f, o, 0x20B);
        put32(&mut f, o + 0x10, 0x1000); // entry point
        put32(&mut f, o + 0x14, 0x1000); // base of code
        f[o + 0x18..o + 0x20].copy_from_slice(&image_base.to_le_bytes());
        put32(&mut f, o + 0x20, 0x1000); // section alignment
        put32(&mut f, o + 0x24, 0x200); // file alignment
        put16(&mut f, o + 0x28, 6);
        put16(&mut f, o + 0x30, 6);
        put32(&mut f, o + 0x38, 0x2000); // size of image
        put32(&mut f, o + 0x3C, 0x200); // size of headers
        put16(&mut f, o + 0x44, 3); // console subsystem
        f[o + 0x48..o + 0x50].copy_from_slice(&0x10_0000u64.to_le_bytes());
        f[o + 0x50..o + 0x58].copy_from_slice(&0x1000u64.to_le_bytes());
        f[o + 0x58..o + 0x60].copy_from_slice(&0x10_0000u64.to_le_bytes());
        f[o + 0x60..o + 0x68].copy_from_slice(&0x1000u64.to_le_bytes());
        put32(&mut f, o + 0x6C, 16); // data directories
        if let Some((rva, size)) = import_dir {
            put32(&mut f, o + 0x70 + 8, rva);
            put32(&mut f, o + 0x70 + 12, size);
        }
        // base relocations: one block (page 0x1000, 8 bytes: no entries) at RVA 0x1100
        put32(&mut f, o + 0x70 + 5 * 8, 0x1100);
        put32(&mut f, o + 0x70 + 5 * 8 + 4, 8);
        put32(&mut f, 0x300, 0x1000);
        put32(&mut f, 0x304, 8);
        // section .text: RVA 0x1000, 0x200 bytes at file offset 0x200, code | execute | read
        let s = o + 0xF0;
        f[s..s + 5].copy_from_slice(b".text");
        put32(&mut f, s + 8, 0x200);
        put32(&mut f, s + 12, 0x1000);
        put32(&mut f, s + 16, 0x200);
        put32(&mut f, s + 20, 0x200);
        put32(&mut f, s + 36, 0x6000_0020);
        f[0x200..0x200 + code.len()].copy_from_slice(code);
        f
    }

    fn write(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xcl-loader-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// DllMain returning FALSE: the load fails, the DLL is not left registered as loaded, and a
    /// second load runs DllMain again (before: the second load returned the uninitialised module).
    #[test]
    fn a_dll_whose_dllmain_fails_is_not_loaded() {
        // xor eax, eax; ret
        let p = write("xcl_dllmain_false.dll", &tiny_dll(0x7E51_0000_0000, &[0x31, 0xC0, 0xC3], None));
        let key = crate::util::dll_key("xcl_dllmain_false.dll");
        for attempt in 0..2 {
            let r = load_game_dll(&p);
            assert!(r.as_ref().is_err_and(|e| e.contains("returned FALSE")), "attempt {attempt}: {:?}", r.map(|m| m.base));
            assert!(modules::find_mapped_by_name(&key).is_none(), "attempt {attempt}: still registered");
        }
        // the same code returning TRUE loads, once
        let p = write("xcl_dllmain_true.dll", &tiny_dll(0x7E52_0000_0000, &[0xB8, 1, 0, 0, 0, 0xC3], None));
        let m = load_game_dll(&p).unwrap();
        assert!(m.initialized.load(Ordering::Acquire));
        assert_eq!(load_game_dll(&p).unwrap().base, m.base);
    }

    /// An import table that does not parse: the load fails and the module – whose memory is
    /// freed – is not found afterwards (before: still registered over freed memory, and a second
    /// load returned it).
    #[test]
    fn a_dll_with_a_broken_import_table_is_not_left_registered() {
        // a descriptor at RVA 0x1010 whose name RVA lies outside the image
        let mut code = vec![0xB8, 1, 0, 0, 0, 0xC3];
        code.resize(0x10, 0xCC);
        for v in [0x1000u32, 0, 0, 0x7FFF_0000, 0x1000] {
            code.extend_from_slice(&v.to_le_bytes());
        }
        let p = write("xcl_bad_imports.dll", &tiny_dll(0x7E53_0000_0000, &code, Some((0x1010, 40))));
        let key = crate::util::dll_key("xcl_bad_imports.dll");
        for attempt in 0..2 {
            let r = load_game_dll(&p);
            assert!(r.as_ref().is_err_and(|e| e.contains("import table")), "attempt {attempt}: {:?}", r.map(|m| m.base));
            assert!(modules::find_mapped_by_name(&key).is_none(), "attempt {attempt}: still registered");
        }
    }
}
