//! Import resolution policy (design §5.2).
//!
//! For `dll!function`:
//! 1. routing/aliases → HLE module (+ cross-HLE lookup, + system fallback for the
//!    kernel family), `@system` or `@stub`;
//! 2. a DLL shipped in the game folder → mapped by our loader;
//! 3. a real Windows DLL → LoadLibrary/GetProcAddress;
//! 4. otherwise → automatic logging stub.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use serde::Serialize;

use crate::hle::{registry, HleModule, Route};
use crate::modules;
use crate::pe::{ExportTarget, ImportName};
use crate::stubs;
use crate::util::{dll_key, wide};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "detail")]
pub enum Source {
    /// Implemented by the HLE module the DLL routes to.
    Hle(String),
    /// Implemented by a different HLE module (cross lookup).
    HleOther(String),
    /// Real Windows DLL.
    System(String),
    /// DLL from the game folder, mapped by XCL.
    GameDll(String),
    Stub,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportRecord {
    pub importer: String,
    pub dll: String,
    pub function: String,
    pub delay: bool,
    pub source: Source,
}

static RECORDS: Mutex<Vec<ImportRecord>> = Mutex::new(Vec::new());
static SEARCH_DIRS: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());
static SYS_CACHE: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

pub fn set_search_dirs(dirs: Vec<PathBuf>) {
    *SEARCH_DIRS.write().unwrap() = dirs;
}

pub fn records() -> Vec<ImportRecord> {
    RECORDS.lock().map(|v| v.clone()).unwrap_or_default()
}

fn fname(what: &ImportName) -> String {
    what.to_string()
}

/// Real Windows module handle (cached). Never maps through our loader.
pub fn system_module(dll: &str) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
    let key = dll_key(dll);
    {
        let g = SYS_CACHE.lock().unwrap();
        if let Some(&h) = g.as_ref().and_then(|m| m.get(&key)) {
            return (h != 0).then_some(h);
        }
    }
    let h = unsafe { LoadLibraryW(wide(dll).as_ptr()) } as usize;
    SYS_CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, h);
    (h != 0).then_some(h)
}

pub fn system_proc_in(h: usize, what: &ImportName) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    let p = unsafe {
        match what {
            ImportName::Name { name, .. } => {
                let c = std::ffi::CString::new(name.as_str()).ok()?;
                GetProcAddress(h as _, c.as_ptr() as *const u8)
            }
            ImportName::Ordinal(o) => GetProcAddress(h as _, *o as usize as *const u8),
        }
    };
    p.map(|f| f as usize)
}

pub fn system_proc(dll: &str, what: &ImportName) -> Option<usize> {
    system_proc_in(system_module(dll)?, what)
}

fn hle_name_for<'a>(m: &'a HleModule, dll: &str, what: &'a ImportName) -> Option<&'a str> {
    match what {
        ImportName::Name { name, .. } => Some(name.as_str()),
        ImportName::Ordinal(o) => m
            .ordinals
            .get(o)
            .copied()
            .or_else(|| stubs::db().ordinal_name(dll, *o)),
    }
}

/// HLE lookup with cross-module search and (kernel family) system fallback.
fn resolve_hle(m: &HleModule, key: &str, what: &ImportName) -> Option<(usize, Source)> {
    let fallback_what = if let Some(name) = hle_name_for(m, key, what) {
        if let Some(&a) = m.exports.get(name) {
            return Some((a, Source::Hle(m.name.to_string())));
        }
        if let Some((other, a)) = registry().find_any(name) {
            return Some((a, Source::HleOther(other.name.to_string())));
        }
        Some(ImportName::Name { hint: 0, name: name.to_string() })
    } else {
        None
    };
    let query = fallback_what.as_ref().unwrap_or(what);
    let ordinal_only = fallback_what.is_none();
    for &c in m.fallback {
        let c = if c == crate::hle::SELF {
            if key == m.name {
                continue;
            }
            key
        } else {
            c
        };
        // Raw ordinals are DLL-specific: never resolve one DLL's ordinal in a different DLL.
        if ordinal_only && c != key {
            continue;
        }
        if let Some(a) = system_proc(c, query) {
            return Some((a, Source::System(c.to_string())));
        }
    }
    None
}

/// Find a DLL shipped with the game (case-insensitive file name match).
pub fn find_game_dll(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    if p.is_absolute() && p.is_file() {
        return Some(p.to_path_buf());
    }
    let key = dll_key(name);
    for dir in SEARCH_DIRS.read().ok()?.iter() {
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().to_ascii_lowercase() == key && e.path().is_file() {
                    return Some(e.path());
                }
            }
        }
    }
    None
}

pub fn resolve_export_of_mapped(m: &modules::MappedModule, what: &ImportName, depth: u32) -> Option<usize> {
    resolve_export_of_mapped_inner(m, what, depth, false)
}

fn resolve_export_of_mapped_inner(
    m: &modules::MappedModule,
    what: &ImportName,
    depth: u32,
    allow_stub: bool,
) -> Option<usize> {
    let t = match what {
        ImportName::Name { name, .. } => m.exports.by_name.get(name)?,
        ImportName::Ordinal(o) => m.exports.by_ordinal.get(&(*o as u32))?,
    };
    match t {
        ExportTarget::Rva(r) => Some(m.base + *r as usize),
        ExportTarget::Forwarder(f) => {
            if depth > 8 {
                return None;
            }
            let (dll, func) = f.rsplit_once('.')?;
            if dll.is_empty() || func.is_empty() {
                return None;
            }
            let what = match func.strip_prefix('#') {
                Some(n) => ImportName::Ordinal(n.parse().ok()?),
                None => ImportName::Name { hint: 0, name: func.to_string() },
            };
            let target_key = if dll.to_ascii_lowercase().ends_with(".dll") {
                dll_key(dll)
            } else {
                dll_key(&format!("{dll}.dll"))
            };
            resolve_inner_opt(&target_key, &what, depth + 1, allow_stub).map(|(a, _)| a)
        }
    }
}

fn stub_for(key: &str, fallback_dll: Option<&str>, what: &ImportName) -> (usize, Source) {
    let name = fname(what);
    let ret = stubs::db().return_for_with_fallback(key, fallback_dll, &name);
    (stubs::make_stub(key, &name, ret), Source::Stub)
}

fn resolve_inner(key: &str, what: &ImportName, depth: u32) -> (usize, Source) {
    resolve_inner_opt(key, what, depth, true).expect("allow_stub=true always resolves")
}

fn resolve_inner_opt(key: &str, what: &ImportName, depth: u32, allow_stub: bool) -> Option<(usize, Source)> {
    let reg = registry();
    match reg.route(key) {
        Some(Route::Stub) => {
            return allow_stub.then(|| {
                modules::hle_handle(key);
                stub_for(key, None, what)
            });
        }
        Some(Route::System) => {
            return match system_proc(key, what) {
                Some(a) => Some((a, Source::System(key.to_string()))),
                None => allow_stub.then(|| stub_for(key, None, what)),
            }
        }
        Some(Route::Hle(i)) => {
            let m = &reg.modules[*i];
            let fallback = (m.name != key).then_some(m.name);
            return resolve_hle(m, key, what).or_else(|| allow_stub.then(|| stub_for(key, fallback, what)));
        }
        None => {}
    }
    // DLL shipped with the game → our loader
    if let Some(m) = modules::find_mapped_by_name(key) {
        if let Some(a) = resolve_export_of_mapped_inner(&m, what, depth, allow_stub) {
            return Some((a, Source::GameDll(key.to_string())));
        }
        return allow_stub.then(|| stub_for(key, None, what));
    }
    if let Some(path) = find_game_dll(key) {
        match crate::loader::load_game_dll(&path) {
            Ok(m) => {
                if let Some(a) = resolve_export_of_mapped_inner(&m, what, depth, allow_stub) {
                    return Some((a, Source::GameDll(key.to_string())));
                }
                return allow_stub.then(|| stub_for(key, None, what));
            }
            Err(e) => crate::xlog_error!("failed to load {}: {e}", path.display()),
        }
    }
    // Real Windows DLL
    if let Some(a) = system_proc(key, what) {
        return Some((a, Source::System(key.to_string())));
    }
    // A known console function imported from an unexpected DLL
    if let ImportName::Name { name, .. } = what {
        if let Some((m, a)) = reg.find_any(name) {
            return Some((a, Source::HleOther(m.name.to_string())));
        }
    }
    allow_stub.then(|| stub_for(key, None, what))
}

/// `XCL_TRACE_IMPORTS`: which imports get a call tracer (arguments and result of every call
/// logged). Comma-separated: `1`/`hle` – every HLE function (the default); `all` – system
/// functions too (a data export imported from a system DLL would be wrapped as well, so this
/// is for short runs); any other word – a prefix of `dll!Function` to include, `-word` one to
/// exclude (e.g. `hle,-kernelx!GetTickCount,gameinput`).
struct TraceFilter {
    hle: bool,
    all: bool,
    include: Vec<String>,
    exclude: Vec<String>,
}

fn trace_filter() -> Option<&'static TraceFilter> {
    static FILTER: std::sync::OnceLock<Option<TraceFilter>> = std::sync::OnceLock::new();
    FILTER
        .get_or_init(|| {
            let v = std::env::var("XCL_TRACE_IMPORTS").ok()?;
            let mut f = TraceFilter { hle: false, all: false, include: Vec::new(), exclude: Vec::new() };
            for w in v.split(',').map(|w| w.trim().to_ascii_lowercase()).filter(|w| !w.is_empty()) {
                match w.as_str() {
                    "1" | "hle" => f.hle = true,
                    "all" => f.all = true,
                    _ => match w.strip_prefix('-') {
                        Some(x) => f.exclude.push(x.to_string()),
                        None => f.include.push(w),
                    },
                }
            }
            if !f.all && f.include.is_empty() {
                f.hle = true;
            }
            Some(f)
        })
        .as_ref()
}

impl TraceFilter {
    fn wants(&self, key: &str, function: &str, source: &Source) -> bool {
        let full = format!("{key}!{function}").to_ascii_lowercase();
        if self.exclude.iter().any(|p| full.starts_with(p.as_str())) {
            return false;
        }
        match source {
            Source::Stub => false, // a stub logs its own calls
            Source::Hle(_) | Source::HleOther(_) if self.hle => true,
            _ => self.all || self.include.iter().any(|p| full.starts_with(p.as_str())),
        }
    }
}

/// Resolve one static/delay import and record the outcome for the report.
pub fn resolve_import(dll: &str, what: &ImportName, importer: &str, delay: bool) -> usize {
    let key = dll_key(dll);
    let (mut addr, source) = resolve_inner(&key, what, 0);
    crate::xlog_debug!("import {importer}: {key}!{what} → {source:?}");
    if let Some(f) = trace_filter() {
        if f.wants(&key, &fname(what), &source) {
            addr = crate::hle::d3d12x::guard::make_call_tracer(addr, format!("{key}!{what}"));
        }
    }
    RECORDS.lock().unwrap().push(ImportRecord {
        importer: importer.to_string(),
        dll: key,
        function: fname(what),
        delay,
        source,
    });
    addr
}

/// Dry-run classification used by `xcl inspect` (does not map game DLLs).
pub fn classify(dll: &str, what: &ImportName) -> Source {
    let key = dll_key(dll);
    let reg = registry();
    match reg.route(&key) {
        Some(Route::Stub) => Source::Stub,
        Some(Route::System) => {
            if system_proc(&key, what).is_some() {
                Source::System(key)
            } else {
                Source::Stub
            }
        }
        Some(Route::Hle(i)) => resolve_hle(&reg.modules[*i], &key, what).map(|x| x.1).unwrap_or(Source::Stub),
        None => {
            if find_game_dll(&key).is_some() {
                return Source::GameDll(key);
            }
            if system_proc(&key, what).is_some() {
                return Source::System(key);
            }
            if let ImportName::Name { name, .. } = what {
                if let Some((m, _)) = reg.find_any(name) {
                    return Source::HleOther(m.name.to_string());
                }
            }
            Source::Stub
        }
    }
}

// ---------------------------------------------------------------------------------------
// Runtime LoadLibrary / GetProcAddress semantics (used by kernelx shims)
// ---------------------------------------------------------------------------------------

/// LoadLibrary as seen by the game. Returns an HMODULE-like value or None.
pub fn load_library(name: &str) -> Option<usize> {
    let key = dll_key(name);
    let reg = registry();
    match reg.route(&key) {
        Some(Route::Hle(i)) => {
            let m = &reg.modules[*i];
            if m.aliases_are_system() && key != m.name {
                // kernel32 / kernelbase / api sets: hand out the real module; GetProcAddress
                // on it still prefers our overrides.
                if let Some(h) = system_module(name) {
                    return Some(h);
                }
            }
            return Some(modules::hle_handle(m.name));
        }
        Some(Route::Stub) => return Some(modules::hle_handle(&key)),
        Some(Route::System) => return system_module(name),
        None => {}
    }
    if let Some(m) = modules::find_mapped_by_name(&key) {
        return Some(m.base);
    }
    let candidate = crate::vfs::get()
        .and_then(|v| v.real_path_for_read(name))
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_game_dll(name));
    if let Some(path) = candidate {
        let under_game = crate::vfs::get().map(|v| v.is_game_path(&path)).unwrap_or(false);
        if under_game || find_game_dll(&key).is_some() {
            match crate::loader::load_game_dll(&path) {
                Ok(m) => return Some(m.base),
                Err(e) => {
                    crate::xlog_error!("LoadLibrary({name}): {e}");
                    return None;
                }
            }
        }
    }
    system_module(name)
}

/// GetProcAddress as seen by the game. Missing functions return None (no stub) so
/// that optional-API probing works.
pub fn get_proc_address(h: usize, what: &ImportName) -> Option<usize> {
    let reg = registry();
    if let Some(name) = modules::hle_name_by_handle(h) {
        let r = match reg.module_by_name(&name).or_else(|| reg.hle_for(&name)) {
            Some(m) => resolve_hle(m, &name, what).map(|x| x.0),
            None => None,
        };
        if r.is_none() {
            crate::xlog_warn!("GetProcAddress({name}, {what}) → NULL (not implemented)");
        }
        return r;
    }
    if let Some(m) = modules::find_mapped_by_base(h) {
        return resolve_export_of_mapped(&m, what, 0);
    }
    // A real module: kernel-family modules prefer our overrides.
    let mut buf = [0u16; 260];
    let n = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW(h as _, buf.as_mut_ptr(), 260) };
    if n > 0 {
        let full = String::from_utf16_lossy(&buf[..n as usize]);
        let key = dll_key(&full);
        if let Some(m) = reg.hle_for(&key) {
            if let ImportName::Name { name, .. } = what {
                if let Some(&a) = m.exports.get(name.as_str()) {
                    return Some(a);
                }
            }
        }
    }
    system_proc_in(h, what)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// advapi32 imports go to kernelx: the registry functions the GDK thunks use for the device
    /// type are XCL's, everything else is the real advapi32.
    #[test]
    fn console_user32_and_legacy_dlls_reach_the_pc_functions() {
        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };
        let (a, s) = resolve_inner("minuserx.dll", &name("PeekMessageW"), 0);
        assert!(a != 0 && matches!(s, Source::System(ref d) if d == "user32.dll"), "{s:?}");
        // the title's window is the console's screen whichever way it is made (`hle::user`) …
        let (a, s) = resolve_inner("minuserx.dll", &name("CreateWindowExW"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::MINUSERX), "{s:?}");
        let (a, s) = resolve_inner("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", &name("CreateWindowExW"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::MINUSERX), "{s:?}");
        // … and the rest of that API set is user32
        let (a, s) = resolve_inner("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", &name("ShowWindow"), 0);
        assert!(a != 0 && matches!(s, Source::System(ref d) if d == "user32.dll"), "{s:?}");
        let (a, s) = resolve_inner("kernel32legacy.dll", &name("GetProcessAffinityMask"), 0);
        assert!(a != 0 && matches!(s, Source::System(_)), "{s:?}");
        let (a, s) = resolve_inner("kernel32legacy.dll", &name("SetThreadAffinityMask"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        let (a, s) = resolve_inner("minappmodelx.dll", &name("RegisterAppStateChangeNotification"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        let (a, s) = resolve_inner("advapi32legacy.dll", &name("RegOpenKeyW"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        // Non-hooked advapi32legacy imports (including LSA policy & privilege APIs) fall through
        // to Windows' advapi32.dll / sechost.dll rather than becoming stubs.
        for sym in ["LsaOpenPolicy", "LsaQueryInformationPolicy", "LsaClose", "LookupPrivilegeValueW"] {
            let (a, s) = resolve_inner("advapi32legacy.dll", &name(sym), 0);
            assert!(a != 0 && matches!(s, Source::System(ref d) if d == "advapi32.dll" || d == "sechost.dll"), "{sym}: {s:?}");
        }
        let h = load_library("advapi32legacy.dll").expect("LoadLibrary(advapi32legacy.dll)");
        assert!(get_proc_address(h, &name("LsaOpenPolicy")).is_some(), "GetProcAddress(advapi32legacy, LsaOpenPolicy)");
    }

    #[test]
    fn advapi32_imports_route_through_kernelx() {
        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };
        let (a, s) = resolve_inner("advapi32.dll", &name("RegOpenKeyW"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        let (a, s) = resolve_inner("advapi32.dll", &name("RegSetValueExW"), 0);
        assert!(a != 0 && matches!(s, Source::System(ref d) if d == "advapi32.dll"), "{s:?}");
    }

    #[test]
    fn hle_alias_stubs_inherit_canonical_module_default() {
        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };
        let e_notimpl = crate::util::E_NOTIMPL as u32 as u64;

        for (dll, sym) in [
            ("xgameplatform.dll", "XUnimplementedAliasProbe"),
            ("xgameruntime.thunks.dll", "XUnimplementedThunkProbe"),
            ("d3d12x.dll", "D3D12UnimplementedAliasProbe"),
        ] {
            let (addr, src) = resolve_inner(dll, &name(sym), 0);
            assert_eq!(src, Source::Stub, "{dll}!{sym}");
            let info = stubs::info_of_stub(addr).expect("stub info registered");
            assert_eq!(info.ret, e_notimpl, "{dll}!{sym} should inherit E_NOTIMPL");
            assert!(info.dll.eq_ignore_ascii_case(dll));
            let f: extern "system" fn() -> u64 = unsafe { std::mem::transmute(addr) };
            assert_eq!(f(), e_notimpl, "{dll}!{sym} thunk return");
        }

        // Kernel-family aliases keep default 0 (BOOL/handle APIs).
        let (addr, src) = resolve_inner("advapi32legacy.dll", &name("UnimplementedKernelAliasProbe"), 0);
        assert_eq!(src, Source::Stub);
        assert_eq!(stubs::info_of_stub(addr).unwrap().ret, 0);
    }

    #[test]
    fn ordinal_imports_fall_through_only_to_matching_system_dll() {
        let ord1 = ImportName::Ordinal(1);
        let expected = system_proc("kernel32.dll", &ord1).expect("kernel32 ordinal #1 exists");
        let (addr, src) = resolve_inner("kernel32.dll", &ord1, 0);
        assert_eq!(src, Source::System("kernel32.dll".into()));
        assert_eq!(addr, expected);

        // Console-only kernelx aliases must NOT borrow unrelated ordinals from kernelbase/kernel32.
        let reg = registry();
        let kx = reg.module_by_name(crate::hle::KERNELX).unwrap();
        assert!(resolve_hle(kx, "kernelx.dll", &ord1).is_none());
        assert!(resolve_hle(kx, "advapi32legacy.dll", &ord1).is_none());
    }

    #[test]
    fn mapped_forwarders_preserve_get_proc_address_contract_and_dotted_dll_names() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_NOACCESS};

        // Reserve two disjoint 64 KiB ranges (0 committed RAM) so test bases never collide with
        // real Windows DLLs in `modules::image_base_for_addr`.
        let target_base = unsafe { VirtualAlloc(std::ptr::null(), 0x10000, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        let proxy_base = unsafe { VirtualAlloc(std::ptr::null(), 0x10000, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(target_base, 0);
        assert_ne!(proxy_base, 0);

        let mut target_exports = crate::pe::OwnedExports::default();
        target_exports.by_name.insert("RealExport".into(), ExportTarget::Rva(0x1234));
        let target_mod = Arc::new(modules::MappedModule {
            name: "dotted.fwdtarget.dll".into(),
            path: PathBuf::from("dotted.fwdtarget.dll"),
            base: target_base,
            size: 0x10000,
            entry_rva: 0,
            is_dll: true,
            exports: target_exports,
            tls_callbacks: Mutex::new(Vec::new()),
            no_thread_calls: AtomicBool::new(false),
            initialized: AtomicBool::new(true),
            stack_reserve: 0,
        });
        modules::register_mapped(target_mod);

        let mut proxy_exports = crate::pe::OwnedExports::default();
        proxy_exports
            .by_name
            .insert("MissingFwd".into(), ExportTarget::Forwarder("nonexistent_fwd_target.MissingApi".into()));
        proxy_exports.by_name.insert(
            "MissingHleFwd".into(),
            ExportTarget::Forwarder("xgameruntime.thunks.XMissingForwardedApi".into()),
        );
        proxy_exports
            .by_name
            .insert("DottedFwd".into(), ExportTarget::Forwarder("dotted.fwdtarget.RealExport".into()));
        proxy_exports
            .by_name
            .insert("KernelFwd".into(), ExportTarget::Forwarder("KERNEL32.GetCurrentProcessId".into()));
        let proxy_mod = Arc::new(modules::MappedModule {
            name: "fwd_proxy_unit.dll".into(),
            path: PathBuf::from("fwd_proxy_unit.dll"),
            base: proxy_base,
            size: 0x10000,
            entry_rva: 0,
            is_dll: true,
            exports: proxy_exports,
            tls_callbacks: Mutex::new(Vec::new()),
            no_thread_calls: AtomicBool::new(false),
            initialized: AtomicBool::new(true),
            stack_reserve: 0,
        });
        modules::register_mapped(Arc::clone(&proxy_mod));

        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };

        // GetProcAddress must return None (NOT synthesize a stub) on unresolvable forwarders.
        assert_eq!(get_proc_address(proxy_base, &name("MissingFwd")), None);
        assert_eq!(get_proc_address(proxy_base, &name("MissingHleFwd")), None);

        // Dotted DLL forwarders split on the LAST dot and resolve cleanly.
        assert_eq!(get_proc_address(proxy_base, &name("DottedFwd")), Some(target_base + 0x1234));
        assert_eq!(
            get_proc_address(proxy_base, &name("KernelFwd")),
            system_proc("kernel32.dll", &name("GetCurrentProcessId"))
        );

        // Static import resolution through a mapped forwarder still synthesizes a stub when missing,
        // including inheriting canonical E_NOTIMPL for dotted HLE alias forwarders.
        let (stub_addr, _) = resolve_inner("fwd_proxy_unit.dll", &name("MissingHleFwd"), 0);
        assert_eq!(
            stubs::info_of_stub(stub_addr).map(|i| i.ret),
            Some(crate::util::E_NOTIMPL as u32 as u64)
        );
    }
}
