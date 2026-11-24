//! Test and diagnostic oracles from the Windows SDK: DXC's validator
//! (`IDxcValidator` from `dxcompiler.dll`, which carries the internal
//! validator since DXC 1.8 — the SDK's `dxil.dll` copies may be older) and
//! its disassembler. Emitted shaders do
//! not need either: their container digest is computed by
//! [`super::container::hash`].

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::Dxc::*;

type CreateInstance = unsafe extern "system" fn(*const windows::core::GUID, *const windows::core::GUID, *mut *mut c_void) -> windows::core::HRESULT;

fn compiler_create() -> Result<CreateInstance, String> {
    let f = crate::gpu::recomp::dxc::create_instance().ok_or("dxcompiler.dll not found")?;
    Ok(unsafe { std::mem::transmute::<usize, CreateInstance>(f) })
}

fn instance<T: Interface>(create: CreateInstance, clsid: &windows::core::GUID) -> Result<T, String> {
    let mut p = std::ptr::null_mut();
    let hr = unsafe { create(clsid, &T::IID, &mut p) };
    if hr.is_err() || p.is_null() {
        return Err(format!("DxcCreateInstance: {hr:?}"));
    }
    Ok(unsafe { T::from_raw(p) })
}

fn blob_text(b: &IDxcBlob) -> String {
    unsafe {
        let (p, n) = (b.GetBufferPointer() as *const u8, b.GetBufferSize());
        // An empty blob may have a null buffer, which `from_raw_parts` forbids.
        if p.is_null() || n == 0 {
            return String::new();
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).trim_end_matches('\0').to_string()
    }
}

/// The text of an operation's error buffer (empty if there is none).
fn error_text(e: windows::core::Result<IDxcBlobEncoding>) -> String {
    e.ok().and_then(|e| e.cast::<IDxcBlob>().ok()).map(|b| blob_text(&b)).unwrap_or_default()
}

fn blob(bytes: &[u8]) -> Result<IDxcBlob, String> {
    let utils: IDxcUtils = instance(compiler_create()?, &CLSID_DxcUtils)?;
    unsafe {
        let len = u32::try_from(bytes.len()).map_err(|_| "blob larger than 4 GiB".to_string())?;
        let b = utils.CreateBlob(bytes.as_ptr() as *const c_void, len, DXC_CP_ACP).map_err(|e| format!("CreateBlob: {e}"))?;
        b.cast::<IDxcBlob>().map_err(|e| format!("{e}"))
    }
}

/// The validator version our metadata declares (`dx.valver`). It stays 1.8 rather than
/// 1.7: the PSV0 part carries `PSVRuntimeInfo3` (the entry name), which validator 1.8
/// introduced, so a 1.7 validator would reject the container anyway.
pub const VALVER: (u32, u32) = (1, 8);

/// The loaded validator's version (`IDxcVersionInfo`), queried once.
pub fn validator_version() -> Result<(u32, u32), String> {
    static V: std::sync::OnceLock<Result<(u32, u32), String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let v: IDxcValidator = instance(compiler_create()?, &CLSID_DxcValidator)?;
        let info: IDxcVersionInfo = v.cast().map_err(|e| format!("IDxcVersionInfo: {e}"))?;
        let (mut major, mut minor) = (0, 0);
        unsafe { info.GetVersion(&mut major, &mut minor) }.map_err(|e| format!("GetVersion: {e}"))?;
        Ok((major, minor))
    })
    .clone()
}

/// Whether the loaded validator can check our containers (version >= [`VALVER`]). The
/// Windows SDK's DXC 1.7, for one, cannot: it rejects `dx.valver` 1.8 before anything else.
pub fn validator_supported() -> bool {
    validator_version().is_ok_and(|v| v >= VALVER)
}

/// Validates a container with DXC's validator; `Err` carries the validator's messages.
///
/// With a validator older than [`VALVER`] the check is skipped (`Ok`, and under test a
/// `SKIPPED <test>: validator <ver> < 1.8` line once per test): emitted shaders do not need
/// the validator at run time, and the GPU tests still run. Malformed containers still fail.
pub fn validate(container: &[u8]) -> Result<(), String> {
    header_fits(container)?;
    let v: IDxcValidator = instance(compiler_create()?, &CLSID_DxcValidator)?;
    let b = blob(container)?;
    unsafe {
        let r = v.Validate(&b, DxcValidatorFlags_Default).map_err(|e| format!("Validate: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_ok() {
            return Ok(());
        }
        let msg = error_text(r.GetErrorBuffer());
        if !validator_supported() && msg.contains("Validator version in metadata") {
            skip_validation();
            return Ok(());
        }
        Err(if msg.is_empty() { format!("validation failed: {status:?}") } else { msg })
    }
}

/// Reports (once per test thread, under test) that validation was skipped for an old validator.
pub fn skip_validation() {
    if !cfg!(test) {
        return;
    }
    use std::io::Write;
    let thread = std::thread::current();
    // Only test threads (`module::test`), not e.g. the translator's worker threads.
    if !thread.name().is_some_and(|n| n.contains("::")) {
        return;
    }
    thread_local!(static REPORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
    if REPORTED.with(|r| r.replace(true)) {
        return;
    }
    let (major, minor) = validator_version().unwrap_or_default();
    let _ = writeln!(std::io::stderr(), "SKIPPED {}: validator {major}.{minor} < {}.{} (validation step only)", thread.name().unwrap_or("?"), VALVER.0, VALVER.1);
}

/// Refuses input shorter than a container header (32 bytes) before DXC sees it: DXC 1.8's
/// validator (dxcompiler 1.8.2502.11, the Windows SDK's) reads the whole header of a blob
/// that starts with `DXBC`, past the end of a shorter one (`b"DXBC"` faults in it).
fn header_fits(container: &[u8]) -> Result<(), String> {
    if container.len() < 32 {
        return Err(format!("not a DXIL container: {} bytes, shorter than its 32-byte header", container.len()));
    }
    Ok(())
}

/// DXC's textual disassembly of a container (for diagnostics and tests).
pub fn disassemble(container: &[u8]) -> Result<String, String> {
    header_fits(container)?;
    let c: IDxcCompiler3 = instance(compiler_create()?, &CLSID_DxcCompiler)?;
    let buf = DxcBuffer { Ptr: container.as_ptr() as *const c_void, Size: container.len(), Encoding: 0 };
    unsafe {
        let r: IDxcResult = c.Disassemble(&buf).map_err(|e| format!("Disassemble: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("{e}"))?;
        if status.is_err() {
            let msg = error_text(r.GetErrorBuffer());
            return Err(format!("disassembly failed: {msg}"));
        }
        let out: IDxcBlob = r.GetResult().map_err(|e| format!("{e}"))?;
        Ok(blob_text(&out))
    }
}

#[cfg(test)]
mod tests {
    /// Malformed input must come back as `Err`, never as a panic or UB in the
    /// FFI glue (empty error blobs may carry a null buffer pointer).
    #[test]
    fn malformed_containers_are_errors() {
        let s = super::super::shader::Shader::new(super::super::container::Stage::Compute, (6, 6));
        let good = s.finish([1, 1, 1], None);
        assert_eq!(super::validate(&good), Ok(()));
        // Garbage and truncations. (A wrong digest is *not* an error here:
        // the validator re-signs rather than checks it; D3D12 does check it,
        // which the GPU tests cover.)
        let inputs: [&[u8]; 5] = [&[], b"DXBC", &[0u8; 64], &good[..good.len() / 2], &good[..good.len() - 4]];
        for c in inputs {
            if crate::gpu::recomp::dxc::create_instance().is_none() {
                return crate::gpu::recomp::runner::skip_test("no dxcompiler.dll");
            }
            assert!(super::validate(c).is_err(), "accepted {} bytes", c.len());
            let _ = super::disassemble(c);
        }
    }
}
