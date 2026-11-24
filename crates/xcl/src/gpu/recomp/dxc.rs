//! In-process HLSL → DXIL compilation with DXC.
//!
//! `dxcompiler.dll` is not part of Windows, so it is loaded at run time from
//! `%XCL_DXC_DIR%`, the folder of xcl.exe, or the Windows SDK with the newest DXC
//! (`Windows Kits\10\bin\<ver>\x64`). `dxil.dll` from the same folder is loaded first so
//! the compiler can sign the DXIL (unsigned DXIL is rejected by D3D12).
//!
//! Which DXC it is matters ([`Toolchain::outdated`]), so the one that was loaded is logged and
//! kept for `report.json` ([`toolchain`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::core::{Interface, GUID, HRESULT, PCWSTR};
use windows::Win32::Graphics::Direct3D::Dxc::{DxcBuffer, IDxcBlob, IDxcCompiler3, IDxcResult, CLSID_DxcCompiler, DXC_CP_UTF8};

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

/// A dotted version, `10.0.26100.0` or `1.9.2602.17`, as four numbers (missing ones are 0).
fn parse_version(s: &str) -> Option<[u32; 4]> {
    let mut v = [0u32; 4];
    let mut parts = s.split('.');
    for x in v.iter_mut() {
        match parts.next() {
            Some(p) => *x = p.parse().ok()?,
            None => break,
        }
    }
    parts.next().is_none().then_some(v)
}

/// The version resource of a file.
fn file_version(path: &Path) -> Option<[u32; 4]> {
    use windows_sys::Win32::Storage::FileSystem::{GetFileVersionInfoSizeW, GetFileVersionInfoW, VerQueryValueW, VS_FIXEDFILEINFO};
    let name = crate::util::wide(&path.to_string_lossy());
    unsafe {
        let n = GetFileVersionInfoSizeW(name.as_ptr(), std::ptr::null_mut());
        if n == 0 {
            return None;
        }
        let mut data = vec![0u8; n as usize];
        if GetFileVersionInfoW(name.as_ptr(), 0, n, data.as_mut_ptr() as *mut c_void) == 0 {
            return None;
        }
        let (mut p, mut len) = (std::ptr::null_mut::<c_void>(), 0u32);
        if VerQueryValueW(data.as_ptr() as *const c_void, crate::util::wide("\\").as_ptr(), &mut p, &mut len) == 0 || (len as usize) < std::mem::size_of::<VS_FIXEDFILEINFO>() {
            return None;
        }
        let f = std::ptr::read_unaligned(p as *const VS_FIXEDFILEINFO);
        Some([f.dwFileVersionMS >> 16, f.dwFileVersionMS & 0xFFFF, f.dwFileVersionLS >> 16, f.dwFileVersionLS & 0xFFFF])
    }
}

/// The DXC release a `dxcompiler.dll` with this version resource is, for comparing folders:
/// `1.7.2308.16` is itself; the old SDKs' compilers carry the number of their Windows build
/// (`10.0.19041.x`, DXC 1.5 and before), so those, and files that say nothing, come last.
fn dxc_release(file: Option<[u32; 4]>) -> [u32; 4] {
    match file {
        Some(v) if v[0] < 10 => v,
        _ => [0; 4],
    }
}

/// Windows SDK folders (name, version resource of the `dxcompiler.dll` in its `x64`), the one
/// to try first in front: the newest DXC, and between equal ones the newest SDK. The names are
/// compared as versions (`10.0.100000.0` is newer than `10.0.26100.0`), other names last.
fn order_sdk_folders(mut v: Vec<(String, Option<[u32; 4]>)>) -> Vec<String> {
    v.sort_by(|a, b| (dxc_release(b.1), parse_version(&b.0), &b.0).cmp(&(dxc_release(a.1), parse_version(&a.0), &a.0)));
    v.into_iter().map(|(name, _)| name).collect()
}

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("XCL_DXC_DIR") {
        v.push(PathBuf::from(d));
    }
    if let Some(d) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        v.push(d);
    }
    for root in [r"C:\Program Files (x86)\Windows Kits\10\bin", r"C:\Program Files\Windows Kits\10\bin"] {
        if let Ok(rd) = std::fs::read_dir(root) {
            let vers = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir());
            let vers = vers.map(|p| (p.file_name().unwrap_or_default().to_string_lossy().into_owned(), file_version(&p.join("x64").join("dxcompiler.dll"))));
            for ver in order_sdk_folders(vers.collect()) {
                v.push(Path::new(root).join(ver).join("x64"));
            }
        }
    }
    v
}

/// One of DXC's two libraries, as it identifies itself.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Library {
    /// the file's version resource: `1.9.2602.17` is the GitHub release 1.9.2602. The Windows
    /// SDK's `dxil.dll` says `101.7.2308.24`, and SDKs before 10.0.22000 the number of their
    /// Windows build
    pub file_version: String,
    /// `IDxcVersionInfo::GetVersion`: the newest DXIL the library knows, which is not always
    /// its release (1.8.2505 says 1.9). 0 and 0: the library did not say
    pub major: u32,
    pub minor: u32,
    /// `IDxcVersionInfo2::GetCommitInfo`: the commit DXC was built from
    pub commit: String,
}

impl Library {
    /// The file's version when it is a DXC release number.
    fn release(&self) -> Option<[u32; 4]> {
        parse_version(&self.file_version).filter(|v| (1..10).contains(&v[0]))
    }

    /// The release, (1, 9) for 1.9.2602: from the file, else what the library said.
    pub fn version(&self) -> Option<(u32, u32)> {
        match self.release() {
            Some(v) => Some((v[0], v[1])),
            None => ((self.major, self.minor) != (0, 0)).then_some((self.major, self.minor)),
        }
    }
}

impl std::fmt::Display for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.release(), self.version(), self.file_version.is_empty()) {
            (Some(_), ..) => write!(f, "{}", self.file_version)?,
            (None, Some((major, minor)), true) => write!(f, "{major}.{minor}")?,
            (None, Some((major, minor)), false) => write!(f, "{major}.{minor} (file {})", self.file_version)?,
            (None, None, false) => write!(f, "(file {})", self.file_version)?,
            (None, None, true) => write!(f, "of unknown version")?,
        }
        if !self.commit.is_empty() {
            write!(f, " ({})", self.commit)?;
        }
        Ok(())
    }
}

/// The DXC this process compiles with (`dxc` in `report.json`).
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Toolchain {
    /// the folder the libraries came from; empty while DXC is not loaded
    pub dir: String,
    pub dxcompiler: Library,
    /// the signer from the same folder; empty when the folder has none
    pub dxil: Library,
}

/// The oldest DXC release XCL recommends (see [`Toolchain::outdated`]).
pub const RECOMMENDED: (u32, u32) = (1, 8);

impl Toolchain {
    /// Whether the compiler is older than [`RECOMMENDED`]. Measured with 1.7.2308 (what Windows
    /// SDK 10.0.26100.4654 installs; other revisions of that SDK have 1.8.2502) against
    /// 1.9.2602 on an RTX 5070 Ti, synthetic translated compute programs (issue #156): DXIL
    /// 2.1–2.7 times the size, pipeline creation in the driver 1.4–2.2 times as long. 1.7 also
    /// rejects `[WaveSize]` on the entries of a library, which 1.8 accepts; the sizes of 1.8's
    /// DXIL were not measured.
    pub fn outdated(&self) -> bool {
        self.dxcompiler.version().is_some_and(|v| v < RECOMMENDED)
    }
}

/// Why an outdated DXC is worth replacing, and how.
pub const OUTDATED_COST: &str = "DXC 1.7 made DXIL 2–2.7 times the size of DXC 1.9's from translated compute programs, and an NVIDIA driver took 1.4–2.2 times as long to build pipelines from it";
pub const OUTDATED_FIX: &str = r"download DXC 1.8 or newer from github.com/microsoft/DirectXShaderCompiler/releases and set XCL_DXC_DIR to its bin\x64 folder";

/// Ask a DXC library for its version. Nothing here fails the load: what the library does not
/// answer stays empty.
unsafe fn library(create: CreateInstance, class: &GUID, file: &Path) -> Library {
    use windows::Win32::Graphics::Direct3D::Dxc::{IDxcVersionInfo, IDxcVersionInfo2};
    let mut lib = Library { file_version: file_version(file).map(|v| format!("{}.{}.{}.{}", v[0], v[1], v[2], v[3])).unwrap_or_default(), ..Default::default() };
    let mut p = std::ptr::null_mut();
    if create(class, &IDxcVersionInfo::IID, &mut p).is_err() || p.is_null() {
        return lib;
    }
    let info = IDxcVersionInfo::from_raw(p);
    let _ = info.GetVersion(&mut lib.major, &mut lib.minor);
    if let Ok(info2) = info.cast::<IDxcVersionInfo2>() {
        let (mut count, mut hash) = (0u32, std::ptr::null_mut::<i8>());
        if info2.GetCommitInfo(&mut count, &mut hash).is_ok() && !hash.is_null() {
            lib.commit = std::ffi::CStr::from_ptr(hash).to_string_lossy().into_owned();
            windows::Win32::System::Com::CoTaskMemFree(Some(hash as *const c_void));
        }
    }
    lib
}

static LIB: OnceLock<Result<(PathBuf, usize), String>> = OnceLock::new();
static TOOLCHAIN: OnceLock<Toolchain> = OnceLock::new();

/// The DXC in use; empty while none is loaded (asking does not load it).
pub fn toolchain() -> Toolchain {
    TOOLCHAIN.get().cloned().unwrap_or_default()
}

#[cfg(test)]
mod version_tests {
    use super::*;

    #[test]
    fn versions_parse_and_compare_as_numbers() {
        assert_eq!(parse_version("10.0.26100.0"), Some([10, 0, 26100, 0]));
        assert_eq!(parse_version("1.9.2602.17"), Some([1, 9, 2602, 17]));
        assert_eq!(parse_version("1.8"), Some([1, 8, 0, 0]));
        for bad in ["", "x64", "1.8.", "1.x", "1.2.3.4.5", "-1.0"] {
            assert_eq!(parse_version(bad), None, "{bad:?}");
        }
        assert!(parse_version("10.0.100000.0") > parse_version("10.0.26100.0"));
        assert!(parse_version("1.10") > parse_version("1.9.2602.17"));
    }

    #[test]
    fn the_sdk_with_the_newest_dxc_is_tried_first() {
        let sdk = |name: &str, dxc: Option<[u32; 4]>| (name.to_string(), dxc);
        let order = order_sdk_folders(vec![
            sdk("arm64", None),
            sdk("10.0.19041.0", Some([10, 0, 19041, 5609])), // DXC 1.5 under Windows' own number
            sdk("10.0.22621.0", Some([1, 6, 2112, 16])),
            sdk("10.0.26100.0", Some([1, 7, 2308, 16])),
            sdk("10.0.100000.0", Some([1, 7, 2308, 16])), // sorts below 10.0.2… as a name
            sdk("10.0.22000.0", Some([1, 8, 2505, 32])), // a newer DXC copied into an older SDK
            sdk("10.0.17134.0", None),
            sdk("x64", None),
        ]);
        assert_eq!(order, ["10.0.22000.0", "10.0.100000.0", "10.0.26100.0", "10.0.22621.0", "10.0.19041.0", "10.0.17134.0", "x64", "arm64"]);
    }

    #[test]
    fn dxc_below_1_8_is_outdated() {
        let lib = |major, minor, file: &str| Library { major, minor, file_version: file.into(), commit: String::new() };
        let outdated = |l: Library| Toolchain { dir: r"C:\dxc".into(), dxcompiler: l, dxil: Library::default() }.outdated();
        // the four compilers of the note in the pull request, as they answer
        assert!(outdated(lib(1, 7, "1.7.2308.16")));
        assert!(outdated(lib(1, 5, "10.0.19041.5609")));
        assert!(!outdated(lib(1, 9, "1.8.2505.32")), "the release counts, not the DXIL version it knows");
        assert!(!outdated(lib(1, 9, "1.9.2602.17")));
        assert!(outdated(lib(1, 9, "1.7.2308.16")));
        // a file that says nothing of the release: what the library says; neither: no verdict
        assert!(outdated(lib(1, 6, "")));
        assert!(!outdated(lib(1, 10, "")));
        assert!(!outdated(lib(2, 0, "101.0.1.2")));
        assert!(outdated(lib(0, 0, "1.6.2112.16")));
        assert!(!outdated(lib(0, 0, "10.0.19041.5609")));
        assert!(!outdated(lib(0, 0, "")));
        assert!(!Toolchain::default().outdated());

        assert_eq!(lib(1, 9, "1.8.2505.32").to_string(), "1.8.2505.32");
        assert_eq!(lib(1, 5, "10.0.19041.5609").to_string(), "1.5 (file 10.0.19041.5609)");
        assert_eq!(lib(1, 7, "101.7.2308.24").to_string(), "1.7 (file 101.7.2308.24)");
        assert_eq!(Library { commit: "21d28f72".into(), ..lib(1, 9, "1.9.2602.17") }.to_string(), "1.9.2602.17 (21d28f72)");
        assert_eq!(lib(1, 9, "").to_string(), "1.9");
        assert_eq!(lib(0, 0, "10.0.1.2").to_string(), "(file 10.0.1.2)");
    }

    /// The DXC of this PC, when it has one: what `report.json` says of it.
    #[test]
    fn the_loaded_dxc_is_identified() {
        let Ok(dir) = location() else {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        };
        let t = toolchain();
        assert_eq!(Path::new(&t.dir), dir);
        let major = t.dxcompiler.major;
        assert!(major >= 1, "IDxcVersionInfo of dxcompiler.dll: {t:?}");
        assert!(!t.dxcompiler.file_version.is_empty(), "{t:?}");
        assert_eq!(Some(t.outdated()), t.dxcompiler.version().map(|v| v < RECOMMENDED));
        if dir.join("dxil.dll").is_file() {
            assert!(t.dxil.version().is_some(), "{t:?}");
        }
        // as the `dxc` of a report, and a report from before the field reads as "not loaded"
        let report = crate::report::Report { dxc: t.clone(), ..Default::default() };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["dxc"]["dir"], t.dir);
        assert_eq!(json["dxc"]["dxcompiler"]["major"], major);
        let mut old = json.clone();
        old.as_object_mut().unwrap().remove("dxc");
        assert_eq!(serde_json::from_value::<crate::report::Report>(old).unwrap().dxc, Toolchain::default());
        println!("DXC: dxcompiler {}, dxil {}, from {}", t.dxcompiler, t.dxil, t.dir);
    }
}

/// Folder with dxcompiler.dll (+ dxil.dll) that will be used.
pub fn location() -> Result<&'static Path, String> {
    load().map(|(p, _)| p.as_path())
}

fn load() -> Result<&'static (PathBuf, usize), String> {
    LIB.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        for dir in candidates() {
            let dxc = dir.join("dxcompiler.dll");
            if !dxc.is_file() {
                continue;
            }
            unsafe {
                let dxil = dir.join("dxil.dll");
                let signer = if dxil.is_file() { LoadLibraryW(crate::util::wide(&dxil.to_string_lossy()).as_ptr()) } else { std::ptr::null_mut() };
                let h = LoadLibraryW(crate::util::wide(&dxc.to_string_lossy()).as_ptr());
                if h.is_null() {
                    continue;
                }
                if let Some(f) = GetProcAddress(h, b"DxcCreateInstance\0".as_ptr()) {
                    let mut t = Toolchain { dir: dir.display().to_string(), ..Default::default() };
                    t.dxcompiler = library(std::mem::transmute::<usize, CreateInstance>(f as usize), &CLSID_DxcCompiler, &dxc);
                    if let Some(g) = if signer.is_null() { None } else { GetProcAddress(signer, b"DxcCreateInstance\0".as_ptr()) } {
                        t.dxil = library(std::mem::transmute::<usize, CreateInstance>(g as usize), &windows::Win32::Graphics::Direct3D::Dxc::CLSID_DxcValidator, &dxil);
                    }
                    let signer = if signer.is_null() { "no dxil.dll".to_string() } else { format!("dxil {}", t.dxil) };
                    crate::xlog_info!("DXC: dxcompiler {}, {signer}, from {}", t.dxcompiler, t.dir);
                    if t.outdated() {
                        crate::xlog_warn!("DXC {} is older than {}.{}: {OUTDATED_COST}; {OUTDATED_FIX}", t.dxcompiler, RECOMMENDED.0, RECOMMENDED.1);
                    }
                    let _ = TOOLCHAIN.set(t);
                    return Ok((dir, f as usize));
                }
            }
        }
        Err("dxcompiler.dll not found (set XCL_DXC_DIR or install the Windows SDK)".into())
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// The PC compiler's DxcCreateInstance (for titles that use the console's `dxcompiler_xs.dll`).
pub fn create_instance() -> Option<usize> {
    load().ok().map(|&(_, f)| f)
}

fn create_fn() -> Result<CreateInstance, String> {
    let &(_, f) = load()?;
    Ok(unsafe { std::mem::transmute::<usize, CreateInstance>(f) })
}

thread_local! {
    static COMPILER: RefCell<Option<IDxcCompiler3>> = const { RefCell::new(None) };
}

fn compiler() -> Result<IDxcCompiler3, String> {
    COMPILER.with(|c| {
        if let Some(x) = c.borrow().as_ref() {
            return Ok(x.clone());
        }
        let create = create_fn()?;
        let mut p = std::ptr::null_mut();
        let hr = unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut p) };
        if hr.is_err() || p.is_null() {
            return Err(format!("DxcCreateInstance: {hr:?}"));
        }
        let x = unsafe { IDxcCompiler3::from_raw(p) };
        *c.borrow_mut() = Some(x.clone());
        Ok(x)
    })
}

/// Stack of the thread DXC runs on: its LLVM passes recurse along long dependency chains of
/// big translated programs (2 MB overflowed on a 7 000-instruction retail PS); the caller may
/// be a game thread with a small stack. Reserved, committed only as used.
const DXC_STACK: usize = 64 << 20;

struct Job {
    src: String,
    entry: String,
    profile: String,
    extra: Vec<String>,
    reply: std::sync::mpsc::Sender<Result<Vec<u8>, String>>,
    /// one of the title's threads waits for it
    wanted: Wanted,
}

/// Whether one of the title's threads waits for a compile (asked again while it is held back).
pub type Wanted = Box<dyn Fn() -> bool + Send>;

/// Compiles the title waits for that go on at once however little commit is left (each takes
/// 0.1 – 0.7 GB).
const WANTED_AT_ONCE: usize = 6;

/// Compile threads (DXC workers, and the PSO workers of `pso_async` that feed them): every core
/// but one, `XCL_DXC_THREADS` to choose. Memory is what `wait_for_memory` watches: a retail
/// pack's biggest translations take ~0.7 GB in DXC (a typical one 0.1–0.3 GB). The cap used
/// to be 6, which left most of a 16-thread CPU idle while a title with ~24 000 shaders
/// compiled for half an hour. (Normal priority: a title thread can wait for a pipeline –
/// `resolve_wait`, a compute pipeline's first bind – and must not wait behind its own threads.)
pub fn compile_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::env::var("XCL_DXC_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(cores.saturating_sub(1)).clamp(1, 64)
    })
}

/// GB of commit `wait_for_memory` keeps free besides the compiles: 6 for a running title (its
/// own allocations), less without one (`set_memory_reserve`: `xcl shaders --precompile`).
static RESERVE_GB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(6);

/// The commit `wait_for_memory` keeps free besides the compiles (no title running: less).
pub fn set_memory_reserve(gb: u64) {
    RESERVE_GB.store(gb, std::sync::atomic::Ordering::Relaxed);
}

/// Compiles running on the DXC workers (`wait_for_memory` lets them in).
static WORKING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A compile let in by `wait_for_memory`, until it is dropped.
pub struct MemorySlot;

impl Drop for MemorySlot {
    fn drop(&mut self) {
        WORKING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Before a compile: with 6 GB of commit left or more, plus 1 GB for every compile already
/// running (each can still grow to ~0.7 GB), at once; with less, one compile at a
/// time – it goes when no other is running (and after a minute in any case). Each compile used
/// to wait the minute while commit was short: with XDefiant's own 10 GB committed, 4 GB were
/// left, and its last cold-cache shaders came a few a minute. One that the title waits for
/// (`wanted`) goes on at once too, up to `WANTED_AT_ONCE`: at a retail title's world loading five
/// of its threads waited for cold compiles, four of which stood in this queue, until its watchdog
/// ended it.
pub fn wait_for_memory(wanted: &Wanted) -> MemorySlot {
    use std::sync::atomic::Ordering;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let start = std::time::Instant::now();
    loop {
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let need = (RESERVE_GB.load(Ordering::Relaxed) << 30) + ((WORKING.load(Ordering::Acquire) as u64) << 30);
        let plenty = unsafe { GlobalMemoryStatusEx(&mut m) } == 0 || m.ullAvailPageFile >= need;
        if plenty || start.elapsed().as_secs() >= 60 || (WORKING.load(Ordering::Acquire) < WANTED_AT_ONCE && wanted()) {
            WORKING.fetch_add(1, Ordering::AcqRel);
            return MemorySlot;
        }
        if WORKING.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            return MemorySlot;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// DXC workers (`compile_threads`, each with its own compiler and a `DXC_STACK` stack), started
/// on first use; callers wait for their job.
fn pool() -> &'static std::sync::Mutex<std::sync::mpsc::Sender<Job>> {
    static POOL: OnceLock<std::sync::Mutex<std::sync::mpsc::Sender<Job>>> = OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        for k in 0..compile_threads() {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-dxc-{k}")).stack_size(DXC_STACK).spawn(move || loop {
                let job = match rx.lock().unwrap().recv() {
                    Ok(j) => j,
                    Err(_) => return,
                };
                let _slot = wait_for_memory(&job.wanted);
                let extra: Vec<&str> = job.extra.iter().map(|s| s.as_str()).collect();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_here(&job.src, &job.entry, &job.profile, &extra)))
                    .unwrap_or_else(|_| Err("DXC: panic in the compile thread".into()));
                let _ = job.reply.send(r);
            });
        }
        std::sync::Mutex::new(tx)
    })
}

/// Compile HLSL to signed DXIL. `profile` e.g. "cs_6_6"; `extra` are dxc arguments.
pub fn compile(src: &str, entry: &str, profile: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
    let (reply, rx) = std::sync::mpsc::channel();
    let job = Job { src: src.into(), entry: entry.into(), profile: profile.into(), extra: extra.iter().map(|s| s.to_string()).collect(), reply, wanted: crate::hle::d3d12x::pso_async::wanted_here() };
    pool().lock().unwrap().send(job).map_err(|_| "DXC: no compile threads".to_string())?;
    rx.recv().map_err(|_| "DXC: the compile thread exited".to_string())?
}

fn compile_here(src: &str, entry: &str, profile: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
    let c = compiler()?;
    let mut args: Vec<Vec<u16>> = ["-E", entry, "-T", profile, "-HV", "2021"].iter().map(|a| crate::util::wide(a)).collect();
    // DXC's optimiser was superlinear on translated programs full of short-circuit `?:` /
    // `&&` / `if` per instruction (a 1589-instruction console CS: ~6 min and > 30 GB at -O3).
    // With branch-free emission (select / and / or) it is fast: that shader 0.3 s, a
    // 15 720-instruction one 6 s (585 KB DXIL vs 1.6 MB at -O0, which the driver could not
    // compile). -O0 only as a safety net for enormous sources.
    // -O1 from 1 MB on: the same DXIL size at a quarter less time (a retail game's 4.8 MB HLSL of
    // a 19 600-instruction CS: 13.5 s instead of 17.7 at -O3; -O0 10.6 s, but twice the DXIL)
    if !extra.iter().any(|a| a.starts_with("-O")) {
        if src.len() > 8 * 1024 * 1024 {
            args.push(crate::util::wide("-O0"));
        } else if src.len() > 1024 * 1024 {
            args.push(crate::util::wide("-O1"));
        }
    }
    args.extend(extra.iter().map(|a| crate::util::wide(a)));
    let ptrs: Vec<PCWSTR> = args.iter().map(|a| PCWSTR(a.as_ptr())).collect();
    let buf = DxcBuffer { Ptr: src.as_ptr() as *const c_void, Size: src.len(), Encoding: DXC_CP_UTF8.0 };
    unsafe {
        let r: IDxcResult = c.Compile(&buf, Some(&ptrs), None).map_err(|e| format!("Compile: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_err() {
            let msg = r
                .GetErrorBuffer()
                .ok()
                .map(|b| {
                    let s = std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize());
                    String::from_utf8_lossy(s).trim_end_matches('\0').to_string()
                })
                .unwrap_or_default();
            return Err(format!("dxc: {msg}"));
        }
        let blob: IDxcBlob = r.GetResult().map_err(|e| format!("{e}"))?;
        Ok(std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()).to_vec())
    }
}

pub(super) fn toolchain_identity() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let Ok(dir) = location() else {
            return "unavailable".into();
        };
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for name in ["dxcompiler.dll", "dxil.dll"] {
            for byte in name.bytes().chain([0]) {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            let Ok(metadata) = std::fs::metadata(dir.join(name)) else {
                h ^= 0xff;
                h = h.wrapping_mul(0x100_0000_01b3);
                continue;
            };
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |duration| duration.as_nanos());
            for byte in metadata.len().to_le_bytes().into_iter().chain(modified.to_le_bytes()) {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        format!("{h:016x}")
    })
}

/// FNV-1a 64 of the inputs that determine the DXIL.
fn cache_key_for_toolchain(src: &str, entry: &str, profile: &str, toolchain: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in [env!("CARGO_PKG_VERSION"), toolchain, profile, entry, src] {
        for b in part.bytes().chain([0u8]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn cache_key(src: &str, entry: &str, profile: &str) -> u64 {
    cache_key_for_toolchain(src, entry, profile, toolchain_identity())
}

/// Folder of the DXIL cache: `XCL_DXIL_CACHE`, else `%LOCALAPPDATA%\xcl\dxil`.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("XCL_DXIL_CACHE") {
        return PathBuf::from(d);
    }
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    base.join("xcl").join("dxil")
}

/// DXIL cache statistics for this process.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CacheStats {
    /// Immediate cache hits in memory or on disk, including cached-only requests.
    pub hits: u64,
    /// Initial misses, excluding cached-only requests.
    pub misses: u64,
    /// Misses filled by another caller; included in `misses`.
    pub coalesced_hits: u64,
    /// Cached-only misses; no compilation attempted.
    pub deferred_misses: u64,
    /// Compilation attempts, including failures.
    pub compilations: u64,
    pub compilation_failures: u64,
    /// First 20 distinct compiler errors.
    pub compilation_errors: Vec<String>,
}

#[derive(Default)]
struct CacheCounters {
    hits: AtomicU64,
    misses: AtomicU64,
    coalesced_hits: AtomicU64,
    deferred_misses: AtomicU64,
    compilations: AtomicU64,
    compilation_failures: AtomicU64,
    compilation_errors: Mutex<Vec<String>>,
}

impl CacheCounters {
    fn snapshot(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            coalesced_hits: self.coalesced_hits.load(Ordering::Relaxed),
            deferred_misses: self.deferred_misses.load(Ordering::Relaxed),
            compilations: self.compilations.load(Ordering::Relaxed),
            compilation_failures: self.compilation_failures.load(Ordering::Relaxed),
            compilation_errors: self.compilation_errors.lock().unwrap().clone(),
        }
    }

    fn failure(&self, error: String) -> String {
        self.compilation_failures.fetch_add(1, Ordering::Relaxed);
        let mut errors = self.compilation_errors.lock().unwrap();
        if errors.len() < 20 && !errors.contains(&error) {
            errors.push(error.clone());
        }
        error
    }
}

fn cache_counters() -> &'static CacheCounters {
    static COUNTERS: OnceLock<CacheCounters> = OnceLock::new();
    COUNTERS.get_or_init(CacheCounters::default)
}

pub fn cache_stats() -> CacheStats {
    cache_counters().snapshot()
}

pub(super) fn note_memory_hit() {
    cache_counters().hits.fetch_add(1, Ordering::Relaxed);
}

/// `compile` through a disk cache in `dir` (big translated programs take seconds in DXC;
/// the same console shader comes back every run). Returns (DXIL, from cache).
pub fn compile_cached_in(dir: &Path, src: &str, entry: &str, profile: &str) -> Result<(Vec<u8>, bool), String> {
    compile_cached_in_with(dir, src, entry, profile, || compile(src, entry, profile, &[]))
}

type CacheFlight = (PathBuf, u64);

fn cache_flight(key: CacheFlight) -> Arc<Mutex<()>> {
    static FLIGHTS: OnceLock<Mutex<HashMap<CacheFlight, Weak<Mutex<()>>>>> = OnceLock::new();
    let mut flights = FLIGHTS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    flights.retain(|_, v| v.strong_count() != 0);
    if let Some(flight) = flights.get(&key).and_then(Weak::upgrade) {
        return flight;
    }
    let flight = Arc::new(Mutex::new(()));
    flights.insert(key, Arc::downgrade(&flight));
    flight
}

struct ProcessCacheLock {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

impl Drop for ProcessCacheLock {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::System::Threading::ReleaseMutex(self.handle);
            windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

fn cache_mutex_name(file: &Path) -> String {
    let absolute = file
        .parent()
        .and_then(|parent| file.file_name().map(|name| normalized_cache_dir(parent).join(name)))
        .unwrap_or_else(|| std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf()));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in absolute.to_string_lossy().to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!(r"Local\XclDxilCache-{h:016x}")
}

fn normalized_cache_dir(dir: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(dir) {
        return canonical;
    }
    let absolute = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    for ancestor in absolute.ancestors().skip(1) {
        if let Ok(canonical) = std::fs::canonicalize(ancestor) {
            if let Ok(remainder) = absolute.strip_prefix(ancestor) {
                return canonical.join(remainder);
            }
        }
    }
    absolute
}

fn process_cache_lock(file: &Path) -> Result<ProcessCacheLock, String> {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_ABANDONED, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{CreateMutexW, WaitForSingleObject, INFINITE};
    let name = crate::util::wide(&cache_mutex_name(file));
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if handle.is_null() {
        return Err("DXIL cache: cannot create the cross-process mutex".into());
    }
    let wait = unsafe { WaitForSingleObject(handle, INFINITE) };
    if wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED {
        unsafe { CloseHandle(handle) };
        return Err(format!("DXIL cache: waiting for the cross-process mutex failed ({wait:#x})"));
    }
    Ok(ProcessCacheLock { handle })
}

fn cached_file(file: &Path) -> Option<Vec<u8>> {
    if let Ok(b) = std::fs::read(&file) {
        if valid_dxil_container(&b) {
            return Some(b);
        }
    }
    None
}

/// Cheap validation before handing a cached blob to D3D12. DXIL uses the DXBC
/// container header: magic, hash, version, total size, chunk count and offsets.
fn valid_dxil_container(blob: &[u8]) -> bool {
    let rd = |at: usize| {
        blob.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    if blob.len() < 36 || blob.get(..4) != Some(b"DXBC") || rd(20) != Some(1) || rd(24) != Some(blob.len()) {
        return false;
    }
    let Some(count) = rd(28) else { return false };
    if count == 0 || count > 64 {
        return false;
    }
    let Some(table_end) = 32usize.checked_add(count * 4) else { return false };
    if table_end > blob.len() {
        return false;
    }
    let mut ranges = Vec::with_capacity(count);
    let mut has_dxil = false;
    for i in 0..count {
        let Some(off) = rd(32 + i * 4) else { return false };
        if off < table_end || off & 3 != 0 {
            return false;
        }
        let Some(size) = rd(off + 4) else { return false };
        let Some(end) = off.checked_add(8).and_then(|v| v.checked_add(size)) else { return false };
        if end > blob.len() || ranges.iter().any(|&(a, b)| off < b && a < end) {
            return false;
        }
        if blob.get(off..off + 4) == Some(b"DXIL") {
            // DxilProgramHeader is six dwords before the bitcode payload.
            if size < 24 || blob.get(off + 16..off + 20) != Some(b"DXIL") {
                return false;
            }
            has_dxil = true;
        }
        ranges.push((off, end));
    }
    has_dxil
}

fn compile_cached_in_with<F>(dir: &Path, src: &str, entry: &str, profile: &str, compile_one: F) -> Result<(Vec<u8>, bool), String>
where
    F: FnOnce() -> Result<Vec<u8>, String>,
{
    compile_cached_in_observed(dir, src, entry, profile, cache_counters(), compile_one)
}

fn compile_cached_in_observed<F>(
    dir: &Path,
    src: &str,
    entry: &str,
    profile: &str,
    counters: &CacheCounters,
    compile_one: F,
) -> Result<(Vec<u8>, bool), String>
where
    F: FnOnce() -> Result<Vec<u8>, String>,
{
    let key = cache_key(src, entry, profile);
    let dir = normalized_cache_dir(dir);
    let file = dir.join(format!("{key:016x}.dxil"));
    if let Some(b) = cached_file(&file) {
        counters.hits.fetch_add(1, Ordering::Relaxed);
        return Ok((b, true));
    }
    // before the locks: a cached-only attempt (non-blocking pipeline creation) must not wait
    // for a worker compiling the same shader
    if CACHED_ONLY.with(|c| c.get()) {
        counters.deferred_misses.fetch_add(1, Ordering::Relaxed);
        MISSED.with(|m| m.set(true));
        return Err("DXIL not compiled yet".into());
    }
    counters.misses.fetch_add(1, Ordering::Relaxed);
    let flight = cache_flight((dir.clone(), key));
    let _guard = flight.lock().unwrap_or_else(|e| e.into_inner());
    let _process_guard = process_cache_lock(&file)?;
    // Another caller may have populated the cache while this one waited.
    if let Some(b) = cached_file(&file) {
        counters.coalesced_hits.fetch_add(1, Ordering::Relaxed);
        return Ok((b, true));
    }
    // A corrupt target must be removed before the atomic rename on Windows;
    // rename does not replace an existing file there.
    if file.is_file() {
        let _ = std::fs::remove_file(&file);
    }
    counters.compilations.fetch_add(1, Ordering::Relaxed);
    let b = {
        let _active = Compiling::start();
        compile_one()
    }
    .and_then(|b| {
        if valid_dxil_container(&b) {
            Ok(b)
        } else {
            Err("DXIL cache: compiler returned an invalid DXIL container".into())
        }
    })
    .map_err(|e| {
        counters.failure(format!(
            "DXIL cache {} ({profile}, entry {entry:?}, {} B HLSL): {e}",
            file.display(),
            src.len()
        ))
    })?;
    if std::fs::create_dir_all(&dir).is_ok() {
        // write to a temporary name first: another thread / process may be reading
        let tmp = file.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, &b).is_ok() && std::fs::rename(&tmp, &file).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok((b, false))
}

/// DXC compiles of programs the disk cache did not have: (running now, finished) – the XCL
/// window shows them (`present::show_compiling`), whatever path the pipeline took (built in
/// the background, or in the title's call when it could not get a stand-in).
static COMPILES: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];

struct Compiling;

impl Compiling {
    fn start() -> Compiling {
        COMPILES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Compiling
    }
}

impl Drop for Compiling {
    fn drop(&mut self) {
        COMPILES[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        COMPILES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// (DXC compiles running now, finished so far) – cache misses only.
pub fn compile_counts() -> (usize, usize) {
    (COMPILES[0].load(std::sync::atomic::Ordering::Relaxed), COMPILES[1].load(std::sync::atomic::Ordering::Relaxed))
}

thread_local! {
    static CACHED_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static MISSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` without compiling: a program missing from the disk cache fails at once instead
/// (non-blocking pipeline creation). Returns `f`'s result and whether something was missing.
pub fn cached_only<R>(f: impl FnOnce() -> R) -> (R, bool) {
    CACHED_ONLY.with(|c| c.set(true));
    MISSED.with(|m| m.set(false));
    let r = f();
    CACHED_ONLY.with(|c| c.set(false));
    (r, MISSED.with(|m| m.replace(false)))
}

/// Inside `cached_only`: what is being made is deferred as if its DXIL were missing (the
/// driver's compile of a big program is the slow part then). False outside `cached_only`.
pub fn defer() -> bool {
    let inside = CACHED_ONLY.with(|c| c.get());
    if inside {
        MISSED.with(|m| m.set(true));
    }
    inside
}

/// Inside `cached_only`.
pub fn in_cached_only() -> bool {
    CACHED_ONLY.with(|c| c.get())
}

pub fn compile_cached(src: &str, entry: &str, profile: &str) -> Result<(Vec<u8>, bool), String> {
    compile_cached_in(&cache_dir(), src, entry, profile)
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn synthetic_dxil(marker: u8) -> Vec<u8> {
        let mut b = vec![0u8; 68];
        b[..4].copy_from_slice(b"DXBC");
        b[20..24].copy_from_slice(&1u32.to_le_bytes());
        b[24..28].copy_from_slice(&68u32.to_le_bytes());
        b[28..32].copy_from_slice(&1u32.to_le_bytes());
        b[32..36].copy_from_slice(&36u32.to_le_bytes());
        b[36..40].copy_from_slice(b"DXIL");
        b[40..44].copy_from_slice(&24u32.to_le_bytes());
        b[52..56].copy_from_slice(b"DXIL"); // inner DxilBitcodeHeader magic
        b[67] = marker;
        b
    }

    #[test]
    fn cache_statistics_separate_deferred_misses_failures_and_hits() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-stats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let counters = CacheCounters::default();
        let src = "cache diagnostics";
        let request = |compile_one: fn() -> Result<Vec<u8>, String>| {
            compile_cached_in_observed(&dir, src, "shade", "ps_6_6", &counters, compile_one)
        };

        let (r, missed) = cached_only(|| request(|| panic!("a cached-only miss must not compile")));
        assert!(r.is_err() && missed);
        assert_eq!(
            counters.snapshot(),
            CacheStats {
                deferred_misses: 1,
                ..Default::default()
            }
        );

        let error = request(|| Err("synthetic DXC diagnostic\nline 2".into())).unwrap_err();
        assert!(
            error.contains(&format!("{:016x}.dxil", cache_key(src, "shade", "ps_6_6"))),
            "{error}"
        );
        assert!(
            error.contains("ps_6_6, entry \"shade\", 17 B HLSL"),
            "{error}"
        );
        assert!(
            error.ends_with("synthetic DXC diagnostic\nline 2"),
            "{error}"
        );
        let invalid = request(|| Ok(b"invalid DXIL".to_vec())).unwrap_err();
        assert!(invalid.contains("invalid DXIL container"), "{invalid}");

        let (dxil, hit) = request(|| Ok(synthetic_dxil(6))).unwrap();
        assert!(!hit);
        assert_eq!(dxil, synthetic_dxil(6));
        let (_, hit) = request(|| panic!("a cache hit must not compile")).unwrap();
        assert!(hit);
        let (r, missed) = cached_only(|| request(|| panic!("a cached-only hit must not compile")));
        assert!(r.unwrap().1 && !missed);

        assert_eq!(
            counters.snapshot(),
            CacheStats {
                hits: 2,
                misses: 3,
                deferred_misses: 1,
                compilations: 3,
                compilation_failures: 2,
                compilation_errors: vec![error, invalid],
                ..Default::default()
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compilation_error_samples_are_bounded_and_distinct() {
        let counters = CacheCounters::default();
        for _ in 0..25 {
            counters.failure("same shader failed".into());
        }
        assert_eq!(
            counters.snapshot().compilation_errors,
            ["same shader failed"]
        );
        for k in 0..25 {
            counters.failure(format!("another shader {k}"));
        }
        let stats = counters.snapshot();
        assert_eq!(stats.compilation_failures, 50);
        assert_eq!(stats.compilation_errors.len(), 20);
        assert_eq!(stats.compilation_errors[0], "same shader failed");
        assert_eq!(stats.compilation_errors[19], "another shader 18");
    }

    #[test]
    fn dxil_cache_round_trip() {
        if location().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("xcl-dxil-test-{}", std::process::id()));
        let src = "[numthreads(1, 1, 1)] void main() {}";
        let (a, hit) = compile_cached_in(&dir, src, "main", "cs_6_6").unwrap();
        assert!(!hit);
        let (b, hit) = compile_cached_in(&dir, src, "main", "cs_6_6").unwrap();
        assert!(hit && a == b);
        assert_ne!(cache_key(src, "main", "cs_6_6"), cache_key(src, "main", "cs_6_7"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_key_tracks_the_dxc_toolchain() {
        let a = cache_key_for_toolchain("source", "main", "cs_6_6", "compiler-a/signer-a");
        assert_eq!(a, cache_key_for_toolchain("source", "main", "cs_6_6", "compiler-a/signer-a"));
        assert_ne!(a, cache_key_for_toolchain("source", "main", "cs_6_6", "compiler-b/signer-a"));
        assert_ne!(a, cache_key_for_toolchain("source", "main", "cs_6_6", "compiler-a/signer-b"));
    }

    #[test]
    fn validates_dxil_cache_structure_and_rebuilds_corruption() {
        let valid = synthetic_dxil(7);
        assert!(valid_dxil_container(&valid));
        assert!(!valid_dxil_container(&valid[..valid.len() - 1]));
        let mut bad = valid.clone();
        bad[0] = b'X';
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[24..28].copy_from_slice(&999u32.to_le_bytes());
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[32..36].copy_from_slice(&35u32.to_le_bytes());
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[36..40].copy_from_slice(b"STAT");
        assert!(!valid_dxil_container(&bad));
        let mut bad = valid.clone();
        bad[52..56].fill(0);
        assert!(!valid_dxil_container(&bad));

        let mut overlap = vec![0u8; 80];
        overlap[..4].copy_from_slice(b"DXBC");
        overlap[20..24].copy_from_slice(&1u32.to_le_bytes());
        overlap[24..28].copy_from_slice(&80u32.to_le_bytes());
        overlap[28..32].copy_from_slice(&2u32.to_le_bytes());
        overlap[32..36].copy_from_slice(&40u32.to_le_bytes());
        overlap[36..40].copy_from_slice(&40u32.to_le_bytes());
        overlap[40..44].copy_from_slice(b"DXIL");
        overlap[44..48].copy_from_slice(&24u32.to_le_bytes());
        overlap[56..60].copy_from_slice(b"DXIL");
        assert!(!valid_dxil_container(&overlap));

        let dir = std::env::temp_dir().join(format!("xcl-dxil-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = cache_key("corrupt", "main", "cs_6_6");
        let file = dir.join(format!("{key:016x}.dxil"));
        std::fs::write(&file, b"truncated cache entry").unwrap();
        let calls = AtomicUsize::new(0);
        let (rebuilt, hit) = compile_cached_in_with(&dir, "corrupt", "main", "cs_6_6", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(synthetic_dxil(8))
        })
        .unwrap();
        assert!(!hit);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(rebuilt, synthetic_dxil(8));
        let (_, hit) = compile_cached_in_with(&dir, "corrupt", "main", "cs_6_6", || panic!("valid rebuilt entry was not reused")).unwrap();
        assert!(hit);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_invalid_fresh_dxil_without_poisoning_the_cache() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-fresh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let key = cache_key("fresh validation", "main", "cs_6_6");
        let file = dir.join(format!("{key:016x}.dxil"));

        let error = compile_cached_in_with(&dir, "fresh validation", "main", "cs_6_6", || Ok(b"not DXIL".to_vec())).unwrap_err();
        assert!(error.contains("invalid DXIL container"));
        assert!(!file.exists());

        let valid = synthetic_dxil(11);
        let (rebuilt, hit) = compile_cached_in_with(&dir, "fresh validation", "main", "cs_6_6", || Ok(valid.clone())).unwrap();
        assert!(!hit);
        assert_eq!(rebuilt, valid);
        let (_, hit) = compile_cached_in_with(&dir, "fresh validation", "main", "cs_6_6", || panic!("valid retry was not cached")).unwrap();
        assert!(hit);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_cache_misses_compile_once() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-flight-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let calls = AtomicUsize::new(0);
        let hits = AtomicUsize::new(0);
        let counters = CacheCounters::default();
        std::thread::scope(|s| {
            let mut threads = Vec::new();
            for _ in 0..8 {
                threads.push(s.spawn(|| {
                    let (dxil, hit) = compile_cached_in_observed(&dir, "same source", "main", "cs_6_6", &counters, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(25));
                        Ok(synthetic_dxil(4))
                    })
                    .unwrap();
                    assert_eq!(dxil, synthetic_dxil(4));
                    hits.fetch_add(hit as usize, Ordering::SeqCst);
                }));
            }
            for thread in threads {
                thread.join().unwrap();
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(hits.load(Ordering::SeqCst), 7);
        let stats = counters.snapshot();
        assert_eq!(stats.compilations, 1);
        assert_eq!(stats.compilation_failures, 0);
        assert_eq!(stats.hits + stats.coalesced_hits, 7);
        assert_eq!(stats.misses, 1 + stats.coalesced_hits);
        assert_eq!(stats.hits + stats.misses, 8);
        assert_eq!(stats.deferred_misses, 0);

        // A cached-only attempt (non-blocking pipeline creation on the title's thread) does not
        // wait for a worker compiling the same shader: it misses at once.
        std::thread::scope(|s| {
            let worker = s.spawn(|| {
                compile_cached_in_with(&dir, "slow", "main", "cs_6_6", || {
                    std::thread::sleep(Duration::from_millis(400));
                    Ok(synthetic_dxil(5))
                })
            });
            std::thread::sleep(Duration::from_millis(50));
            let t0 = std::time::Instant::now();
            let (r, missed) = cached_only(|| compile_cached_in_with(&dir, "slow", "main", "cs_6_6", || Ok(synthetic_dxil(5))));
            assert!(r.is_err() && missed, "cached-only: a miss");
            assert!(t0.elapsed() < Duration::from_millis(200), "cached-only waited {:?} for the compile", t0.elapsed());
            worker.join().unwrap().unwrap();
        });

        // A failed leader releases the flight; the next caller can retry.
        let failed = compile_cached_in_with(&dir, "retry", "main", "cs_6_6", || Err("synthetic failure".into()));
        assert!(failed.is_err());
        let (dxil, hit) = compile_cached_in_with(&dir, "retry", "main", "cs_6_6", || Ok(synthetic_dxil(9))).unwrap();
        assert!(!hit);
        assert_eq!(dxil, synthetic_dxil(9));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn different_cache_keys_do_not_block_each_other() {
        let dir = std::env::temp_dir().join(format!("xcl-dxil-parallel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|s| {
            let a = s.spawn(|| compile_cached_in_with(&dir, "source a", "main", "cs_6_6", || {
                barrier.wait();
                Ok(synthetic_dxil(1))
            }));
            let b = s.spawn(|| compile_cached_in_with(&dir, "source b", "main", "cs_6_6", || {
                barrier.wait();
                Ok(synthetic_dxil(2))
            }));
            assert_eq!(a.join().unwrap().unwrap().0, synthetic_dxil(1));
            assert_eq!(b.join().unwrap().unwrap().0, synthetic_dxil(2));
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_aliases_share_the_same_cache_flight() {
        let root = std::env::temp_dir().join(format!("xcl-dxil-alias-{}", std::process::id()));
        let dir = root.join("cache");
        let child = dir.join("child");
        let alias = child.join("..");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&child).unwrap();
        assert_eq!(normalized_cache_dir(&dir), normalized_cache_dir(&alias));
        assert_eq!(normalized_cache_dir(&root.join("future")), normalized_cache_dir(&child.join("..").join("..").join("future")));
        let key = cache_key("aliased source", "main", "cs_6_6");
        assert_eq!(
            cache_mutex_name(&dir.join(format!("{key:016x}.dxil"))),
            cache_mutex_name(&alias.join(format!("{key:016x}.dxil")))
        );

        let start = std::sync::Barrier::new(2);
        let calls = AtomicUsize::new(0);
        let hits = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                start.wait();
                compile_cached_in_with(&dir, "aliased source", "main", "cs_6_6", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    Ok(synthetic_dxil(12))
                })
                .unwrap()
            });
            let second = scope.spawn(|| {
                start.wait();
                compile_cached_in_with(&alias, "aliased source", "main", "cs_6_6", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    Ok(synthetic_dxil(12))
                })
                .unwrap()
            });
            for result in [first.join().unwrap(), second.join().unwrap()] {
                assert_eq!(result.0, synthetic_dxil(12));
                hits.fetch_add(result.1 as usize, Ordering::SeqCst);
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn process_cache_mutex_serializes_the_same_file() {
        let file = std::env::temp_dir().join(format!("xcl-dxil-mutex-{}-{}.dxil", std::process::id(), cache_key("mutex", "main", "cs_6_6")));
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (second_tx, second_rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let first_file = file.clone();
            s.spawn(move || {
                let _guard = process_cache_lock(&first_file).unwrap();
                held_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            held_rx.recv().unwrap();
            s.spawn(move || {
                let _guard = process_cache_lock(&file).unwrap();
                second_tx.send(()).unwrap();
            });
            assert!(second_rx.recv_timeout(Duration::from_millis(30)).is_err());
            release_tx.send(()).unwrap();
            second_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        });
    }
}
