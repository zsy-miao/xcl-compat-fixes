//! Optional DirectX 12 Agility SDK loading via `ID3D12SDKConfiguration1`.
//!
//! Windows 10's inbox `System32\D3D12Core.dll` stops at Shader Model 6.5, even when
//! the GPU and driver support SM 6.6+. Windows 10 22H2 (build 19045.3393+, KB5029331)
//! and Windows 11 expose `ID3D12SDKConfiguration1::CreateDeviceFactory`, which loads an
//! app-supplied Agility SDK `D3D12Core.dll` at run time without Developer Mode and
//! without static `D3D12SDKVersion` / `D3D12SDKPath` PE exports.
//!
//! Search order (first folder containing a valid x64 `D3D12Core.dll` exporting
//! `D3D12SDKVersion`):
//! 1. `%XCL_D3D12_DIR%` (when set and non-empty),
//! 2. `<exe_dir>\D3D12`,
//! 3. `<exe_dir>`,
//! 4. `%LOCALAPPDATA%\XCL\D3D12`.
//!
//! If no Agility SDK is found or the OS lacks `ID3D12SDKConfiguration1`, device creation
//! falls back transparently to inbox `D3D12CreateDevice`.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows::core::{Interface, IUnknown, Param, GUID, PCSTR};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL;
use windows::Win32::Graphics::Direct3D12::*;

use crate::pe::{Layout, Pe};

/// Environment variable overriding the Agility SDK folder.
pub const ENV: &str = "XCL_D3D12_DIR";

/// Result of attempting to load the Agility SDK once for this process.
#[derive(Debug)]
pub enum State {
    /// Loaded through `ID3D12SDKConfiguration1::CreateDeviceFactory`.
    Loaded {
        factory: ID3D12DeviceFactory,
        version: u32,
        dir: PathBuf,
    },
    /// No Agility SDK DLL was found in any search path; inbox D3D12 will be used.
    NotFound,
    /// An Agility SDK was requested or found, but could not be activated; inbox D3D12 will be used.
    Unavailable(String),
}

static STATE: OnceLock<State> = OnceLock::new();
static DEBUG_LAYER: OnceLock<Result<(), String>> = OnceLock::new();

/// Candidate directories in search order.
pub(crate) fn candidates(env_override: Option<PathBuf>, exe_dir: Option<PathBuf>, local_app_data: Option<PathBuf>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = env_override {
        if !d.as_os_str().is_empty() {
            dirs.push(d);
            // When the user explicitly sets XCL_D3D12_DIR, do not silently fall through
            // to unrelated folders if that path is wrong.
            return dirs;
        }
    }
    if let Some(d) = exe_dir {
        dirs.push(d.join("D3D12"));
        dirs.push(d);
    }
    if let Some(d) = local_app_data {
        dirs.push(d.join("XCL").join("D3D12"));
    }
    dirs
}

/// Read the exported `u32` `D3D12SDKVersion` directly from a `D3D12Core.dll` image on disk
/// without loading or executing the DLL.
pub(crate) fn read_sdk_version_from_bytes(bytes: &[u8]) -> Result<u32, String> {
    let pe = Pe::parse(bytes, Layout::File).map_err(|e| format!("invalid PE: {e}"))?;
    if !pe.is_dll() {
        return Err("not a DLL".into());
    }
    let exports = pe.exports().map_err(|e| format!("export table: {e}"))?.ok_or("no export table")?;
    let rva = exports
        .names
        .iter()
        .find_map(|(name, idx)| {
            if name == "D3D12SDKVersion" {
                exports.functions.get(*idx as usize).copied()
            } else {
                None
            }
        })
        .ok_or("missing D3D12SDKVersion export")?;
    if rva == 0 || exports.is_forwarder_rva(rva) {
        return Err("invalid D3D12SDKVersion export RVA".into());
    }
    let version = pe.u32_at(rva).ok_or("D3D12SDKVersion RVA outside image")?;
    if version == 0 {
        return Err("D3D12SDKVersion is 0".into());
    }
    Ok(version)
}

/// Read `D3D12SDKVersion` from `<dir>\D3D12Core.dll`.
pub(crate) fn probe_dir(dir: &Path) -> Result<u32, String> {
    let dll = dir.join("D3D12Core.dll");
    let bytes = std::fs::read(&dll).map_err(|e| format!("{}: {e}", dll.display()))?;
    read_sdk_version_from_bytes(&bytes).map_err(|e| format!("{}: {e}", dll.display()))
}

/// Encode a UTF-16 code-unit slice for `cp` as a NUL-terminated multi-byte string,
/// rejecting unrepresentable characters and unpaired surrogates.
pub(crate) fn encode_wide_for_code_page(wide: &[u16], cp: u32) -> Result<CString, String> {
    use windows_sys::Win32::Globalization::{WideCharToMultiByte, CP_UTF8, WC_ERR_INVALID_CHARS, WC_NO_BEST_FIT_CHARS};

    if wide.is_empty() {
        return Err("empty Agility SDK path".into());
    }
    if wide.contains(&0) {
        return Err("Agility SDK path contains NUL".into());
    }
    let wide_len = i32::try_from(wide.len()).map_err(|_| "Agility SDK path too long")?;
    // CP_UTF8 (65001) and GB18030 (54936) forbid WC_NO_BEST_FIT_CHARS and non-null lpUsedDefaultChar.
    let utf8_like = cp == CP_UTF8 || cp == 54936;
    let flags = if utf8_like { WC_ERR_INVALID_CHARS } else { WC_NO_BEST_FIT_CHARS };
    let mut used_default = 0i32;
    let used_ptr = if utf8_like { std::ptr::null_mut() } else { &mut used_default };
    // SAFETY: `wide` is a valid non-empty slice of UTF-16 code units of length `wide_len`;
    // querying the required byte count with a null output buffer and 0 length is documented
    // by Win32, and `flags`/`used_ptr` satisfy the code page's Win32 contract.
    let needed = unsafe {
        WideCharToMultiByte(cp, flags, wide.as_ptr(), wide_len, std::ptr::null_mut(), 0, std::ptr::null(), used_ptr)
    };
    if needed <= 0 || used_default != 0 {
        return Err("could not encode Agility SDK path in active code page".into());
    }
    let mut buf = vec![0u8; needed as usize];
    used_default = 0;
    let used_ptr = if utf8_like { std::ptr::null_mut() } else { &mut used_default };
    // SAFETY: `buf` has exact length `needed` returned by the sizing call above.
    let written = unsafe {
        WideCharToMultiByte(cp, flags, wide.as_ptr(), wide_len, buf.as_mut_ptr(), needed, std::ptr::null(), used_ptr)
    };
    if written <= 0 || used_default != 0 {
        return Err("Agility SDK path contains characters not representable in the system ANSI code page".into());
    }
    buf.truncate(written as usize);
    CString::new(buf).map_err(|_| "Agility SDK path contains interior NUL".into())
}

/// Trim whitespace-only paths (matching `XCL_ADAPTER` handling) while preserving non-UTF-8 paths.
pub(crate) fn clean_env_path(val: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let v = val?;
    if v.is_empty() {
        return None;
    }
    if let Some(s) = v.to_str() {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return None;
        }
        return Some(PathBuf::from(trimmed));
    }
    Some(PathBuf::from(v))
}

/// Format a directory path as a null-terminated ANSI string with a trailing backslash,
/// as required by `ID3D12SDKConfiguration1::CreateDeviceFactory`.
pub(crate) fn format_sdk_path(dir: &Path) -> Result<CString, String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Globalization::GetACP;

    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    if wide.is_empty() {
        return Err("empty Agility SDK path".into());
    }
    if *wide.last().unwrap() == b'/' as u16 {
        *wide.last_mut().unwrap() = b'\\' as u16;
    } else if *wide.last().unwrap() != b'\\' as u16 {
        wide.push(b'\\' as u16);
    }
    // SAFETY: `GetACP` takes no arguments and returns the process's active ANSI code page.
    let acp = unsafe { GetACP() };
    encode_wide_for_code_page(&wide, acp)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DirResolution {
    Found { dir: PathBuf, version: u32 },
    NotFound,
    Unavailable(String),
}

pub(crate) fn resolve_dir(
    env_override: Option<PathBuf>,
    exe_dir: Option<PathBuf>,
    local_app_data: Option<PathBuf>,
) -> DirResolution {
    let explicit = env_override.as_ref().is_some_and(|d| !d.as_os_str().is_empty());
    let dirs = candidates(env_override, exe_dir, local_app_data);

    let mut last_err: Option<String> = None;
    for dir in dirs {
        let dll = dir.join("D3D12Core.dll");
        if !dll.is_file() {
            if explicit {
                return DirResolution::Unavailable(format!("{ENV}={} has no D3D12Core.dll", dir.display()));
            }
            continue;
        }
        match probe_dir(&dir) {
            Ok(version) => return DirResolution::Found { dir, version },
            Err(err) => {
                last_err = Some(err);
                if explicit {
                    break;
                }
            }
        }
    }

    match last_err {
        Some(err) => DirResolution::Unavailable(err),
        None => DirResolution::NotFound,
    }
}

fn init() -> State {
    let env_dir = clean_env_path(std::env::var_os(ENV));
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let local_app_data = clean_env_path(std::env::var_os("LOCALAPPDATA"));

    let (dir, version) = match resolve_dir(env_dir, exe_dir, local_app_data) {
        DirResolution::Found { dir, version } => (dir, version),
        DirResolution::NotFound => return State::NotFound,
        DirResolution::Unavailable(reason) => return State::Unavailable(reason),
    };

    // CreateDeviceFactory resolves relative SDK paths relative to the executable directory,
    // whereas probe_dir checks relative to CWD. Absolutize before passing to Win32.
    let abs_dir = std::path::absolute(&dir).unwrap_or(dir);
    let path_cstr = match format_sdk_path(&abs_dir) {
        Ok(p) => p,
        Err(err) => return State::Unavailable(err),
    };

    match create_factory(version, &abs_dir, &path_cstr) {
        Ok(factory) => State::Loaded { factory, version, dir: abs_dir },
        Err(err) => State::Unavailable(err),
    }
}

type D3D12GetInterfaceFn = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut std::ffi::c_void) -> windows::core::HRESULT;

unsafe fn resolve_d3d12_get_interface() -> Result<D3D12GetInterfaceFn, String> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};

    let dll = crate::util::wide("d3d12.dll");
    let h = LoadLibraryExW(dll.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
    if h.is_null() {
        return Err("failed to load d3d12.dll".into());
    }
    let sym = GetProcAddress(h, b"D3D12GetInterface\0".as_ptr())
        .ok_or_else(|| "D3D12GetInterface unavailable (update Windows 10 to 22H2 KB5029331+)".to_string())?;
    Ok(std::mem::transmute::<_, D3D12GetInterfaceFn>(sym))
}

fn create_factory(version: u32, dir: &Path, path_cstr: &CString) -> Result<ID3D12DeviceFactory, String> {
    // SAFETY: `D3D12GetInterface` (resolved dynamically so pre-2004 Windows 10 builds do not
    // fail at IAT load time) writes a COM interface pointer on success into `cfg_raw`.
    // `ID3D12SDKConfiguration1::CreateDeviceFactory` takes a valid NUL-terminated ANSI path
    // with trailing backslash and returns an owned `ID3D12DeviceFactory`.
    unsafe {
        let get_iface = resolve_d3d12_get_interface()?;
        let mut cfg_raw: *mut std::ffi::c_void = std::ptr::null_mut();
        get_iface(&CLSID_D3D12SDKConfiguration, &ID3D12SDKConfiguration1::IID, &mut cfg_raw)
            .ok()
            .map_err(|e| format!("ID3D12SDKConfiguration1 unavailable (update Windows 10 to 22H2 KB5029331+): {e}"))?;
        if cfg_raw.is_null() {
            return Err("D3D12GetInterface returned null ID3D12SDKConfiguration1".into());
        }
        let cfg = ID3D12SDKConfiguration1::from_raw(cfg_raw);
        let factory: ID3D12DeviceFactory = cfg
            .CreateDeviceFactory(version, PCSTR(path_cstr.as_ptr().cast()))
            .map_err(|e| format!("CreateDeviceFactory(v{version}, {}): {e}", dir.display()))?;
        factory
            .SetFlags(D3D12_DEVICE_FACTORY_FLAG_ALLOW_RETURNING_EXISTING_DEVICE)
            .map_err(|e| format!("ID3D12DeviceFactory::SetFlags: {e}"))?;
        // Upgrade process-wide D3D12 global state before XCL creates its first device so
        // free functions not exposed on ID3D12DeviceFactory (notably
        // D3D12SerializeVersionedRootSignature for SM 6.6 directly-indexed heap flags)
        // dispatch to the Agility SDK's D3D12Core.dll rather than Windows 10's inbox DLL.
        factory
            .ApplyToGlobalState()
            .map_err(|e| format!("ID3D12DeviceFactory::ApplyToGlobalState: {e}"))?;
        Ok(factory)
    }
}

/// Process-wide Agility SDK state (initialized on first D3D12 use).
pub fn state() -> &'static State {
    STATE.get_or_init(init)
}

/// Diagnostic message when a device reports Shader Model < 6.6.
pub(crate) fn format_no_sm66_message(adapter: &str, state: &State) -> String {
    match state {
        State::Loaded { version, .. } => format!("{adapter}: no SM 6.6 (Agility SDK v{version} loaded)"),
        State::Unavailable(reason) => format!("{adapter}: no SM 6.6 (Agility SDK unavailable: {reason})"),
        State::NotFound => format!("{adapter}: no SM 6.6 (on Windows 10, set {ENV} to a D3D12 Agility SDK folder)"),
    }
}

/// Active Agility SDK `(version, dir)` if loaded, otherwise `None`.
pub fn loaded() -> Option<(u32, &'static Path)> {
    match state() {
        State::Loaded { version, dir, .. } => Some((*version, dir.as_path())),
        _ => None,
    }
}

/// Create a D3D12 device through the Agility SDK factory when loaded, otherwise via
/// inbox `D3D12CreateDevice`.
pub fn create_device<P>(adapter: P, feature_level: D3D_FEATURE_LEVEL) -> windows::core::Result<ID3D12Device>
where
    P: Param<IUnknown>,
{
    let mut device: Option<ID3D12Device> = None;
    match state() {
        State::Loaded { factory, .. } => {
            // SAFETY: `factory` is a valid `ID3D12DeviceFactory` kept alive for `'static`,
            // and `device` is a valid out-pointer for `ID3D12Device`.
            unsafe { factory.CreateDevice(adapter, feature_level, &mut device)? };
        }
        _ => {
            // SAFETY: standard `D3D12CreateDevice` call with valid out-pointer.
            unsafe { D3D12CreateDevice(adapter, feature_level, &mut device)? };
        }
    }
    device.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))
}

/// Retrieve a D3D12 configuration/debug interface (`ID3D12Debug`,
/// `ID3D12DeviceRemovedExtendedDataSettings`, etc.) from the Agility SDK factory when
/// loaded, otherwise from inbox `D3D12GetDebugInterface`.
pub fn configuration_interface<T: Interface>(clsid: &GUID) -> windows::core::Result<T> {
    match state() {
        State::Loaded { factory, .. } => {
            // SAFETY: `factory` is a valid `ID3D12DeviceFactory` kept alive for `'static`
            // and `clsid` is a valid GUID ref. Do not fall back to global
            // `D3D12GetDebugInterface`, which configures inbox singleton state rather than
            // this factory and can remove existing singleton devices.
            unsafe { factory.GetConfigurationInterface::<T>(clsid) }
        }
        _ => {
            let mut out: Option<T> = None;
            // SAFETY: `D3D12GetDebugInterface` writes a COM interface pointer into `out` on success.
            unsafe { D3D12GetDebugInterface(&mut out)? };
            out.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_NOINTERFACE))
        }
    }
}

/// Enable the debug layer at most once per process. The GPU test runner may create a
/// device before D3D12.X starts; enabling it again after that removes the device.
pub fn enable_debug_layer() -> &'static Result<(), String> {
    DEBUG_LAYER.get_or_init(|| {
        let debug = configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug).map_err(|e| e.to_string())?;
        unsafe { debug.EnableDebugLayer() };
        Ok(())
    })
}

#[cfg(test)]
#[path = "agility_tests.rs"]
mod tests;
