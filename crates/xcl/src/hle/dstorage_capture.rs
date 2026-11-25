//! Opt-in local fixtures for BCPack modes 5 and 7. No decoding or destination writes.

use super::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAX_SAMPLES: usize = 32;
const MAX_BYTES: u32 = 16 << 20;

#[derive(Default)]
struct Captures {
    seen: HashSet<u64>,
    initialized: bool,
}

/// Called while the title's source is still valid at EnqueueRequest, before queueing.
pub(super) unsafe fn capture_refused(request: &Request) {
    let Some(directory) = std::env::var_os("XCL_BCPACK_CAPTURE_DIR").filter(|v| !v.is_empty()) else { return };
    static STATE: OnceLock<Mutex<HashMap<PathBuf, Captures>>> = OnceLock::new();
    let directory = PathBuf::from(directory);
    let mut state = STATE.get_or_init(Default::default).lock().unwrap();
    let captures = state.entry(directory.clone()).or_default();
    match capture_to(&directory, captures, request) {
        Ok(Some(path)) => crate::xlog_info!("DirectStorage: captured BCPack mode {} source locally: {}", request.bcpack(), path.display()),
        Ok(None) => {}
        Err(error) => warn_once("BCPack capture", format_args!("local fixture capture failed: {error}")),
    }
}

fn initialize(directory: &Path, captures: &mut Captures) -> Result<(), String> {
    if captures.initialized {
        return Ok(());
    }
    match std::fs::read_dir(directory) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(hex) = name.strip_prefix("bcpack7-").or_else(|| name.strip_prefix("bcpack5-"))
                    .and_then(|s| s.strip_suffix(".bin")) {
                    if let Ok(hash) = u64::from_str_radix(hex, 16) {
                        captures.seen.insert(hash);
                    }
                }
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    captures.initialized = true;
    Ok(())
}

unsafe fn source_bytes(request: &Request) -> Result<(Vec<u8>, Option<String>), String> {
    let mut bytes = vec![0u8; request.source_size as usize];
    if request.memory_source() {
        if request.source_physical_pages() {
            let pages = snapshot_page_array(request.source, request.source_page_offset, request.source_size)
                .map_err(|hr| format!("invalid source page array: {hr:#x}"))?;
            if !crate::hle::kernelx::read_physical_pages(&pages, request.source_page_offset as usize, &mut bytes) {
                return Err("physical source pages are unavailable".into());
            }
        } else {
            if !crate::gpu::shader::readable(request.source as usize, bytes.len()) {
                return Err("source memory is unreadable".into());
            }
            std::ptr::copy_nonoverlapping(request.source as *const u8, bytes.as_mut_ptr(), bytes.len());
        }
        Ok((bytes, None))
    } else {
        let object = request.source as *mut c_void;
        if !crate::gpu::shader::readable(object as usize, std::mem::size_of::<Object<FileInner>>())
            || *(object as *const *const usize) != file_vtbl().as_ptr()
        {
            return Err("source is not an open DirectStorage file".into());
        }
        let file = Arc::clone(inner::<FileInner>(object));
        let handle = file.handle().ok_or_else(|| "source DirectStorage file is closed".to_string())?;
        request.offset.checked_add(request.source_size as u64).ok_or_else(|| "source file range overflows".to_string())?;
        read_exact_at(&handle, request.offset, &mut bytes).map_err(|hr| format!("source file read failed: {hr:#x}"))?;
        let end = file.path.iter().position(|&c| c == 0).unwrap_or(file.path.len());
        Ok((bytes, Some(String::from_utf16_lossy(&file.path[..end]))))
    }
}

fn fixture_hash(request: &Request, bytes: &[u8]) -> u64 {
    // Diagnostic deduplication, not a security checksum. Pointer values are deliberately omitted.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in request.options.to_le_bytes().iter()
        .chain(request.destination_size.to_le_bytes().iter())
        .chain(request.intermediate_size.to_le_bytes().iter())
        .chain(bytes)
    {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
    }
    hash
}

unsafe fn capture_to(directory: &Path, captures: &mut Captures, request: &Request) -> Result<Option<PathBuf>, String> {
    if !matches!(request.bcpack(), 5 | 7) || request.source_size == 0 || request.source_size > MAX_BYTES
        || request.destination_size == 0 || request.destination_size > MAX_BYTES || request.intermediate_size > MAX_BYTES
    {
        return Ok(None);
    }
    initialize(directory, captures)?;
    if captures.seen.len() >= MAX_SAMPLES {
        return Ok(None);
    }
    let (bytes, source_file) = source_bytes(request)?;
    let hash = fixture_hash(request, &bytes);
    if captures.seen.contains(&hash) {
        return Ok(None);
    }
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let stem = format!("bcpack{}-{hash:016x}", request.bcpack());
    let binary = directory.join(format!("{stem}.bin"));
    let mut output = match std::fs::OpenOptions::new().write(true).create_new(true).open(&binary) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            captures.seen.insert(hash);
            return Ok(None);
        }
        Err(error) => return Err(error.to_string()),
    };
    output.write_all(&bytes).map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    captures.seen.insert(hash);
    let metadata = serde_json::json!({
        "capture_version": 1,
        "compression": "bcpack",
        "bcpack_mode": request.bcpack(),
        "zlib_before_bcpack": request.zlib(),
        "swizzle_mode": request.swizzle(),
        "options_hex": format!("{:#x}", request.options),
        "source_kind": if request.memory_source() { if request.source_physical_pages() { "physical_pages" } else { "memory" } } else { "file" },
        "source_file": source_file,
        "source_offset": request.offset,
        "source_page_offset": request.source_page_offset,
        "source_size": request.source_size,
        "intermediate_size": request.intermediate_size,
        "destination_size": request.destination_size,
        "destination_is_physical_pages": request.dest_physical_pages(),
        "destination_page_offset": request.destination_page_offset,
        "request_tag_hex": format!("{:#x}", request.tag),
        "payload_file": binary.file_name().unwrap().to_string_lossy(),
        "dedup_fnv1a64": format!("{hash:016x}"),
        "decoder_output_available": false
    });
    let json = serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?;
    std::fs::write(directory.join(format!("{stem}.json")), json).map_err(|e| e.to_string())?;
    Ok(Some(binary))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            Self(std::env::temp_dir().join(format!("xcl-bcpack-capture-{}-{nonce}", std::process::id())))
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            assert!(self.0.is_absolute() && self.0.starts_with(std::env::temp_dir()));
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn bcpack_failure_capture_preserves_memory_payload_and_refusal() {
        let directory = TempDir::new();
        let source = [1u8, 3, 5, 7, 9];
        let destination = [0xA5u8; 64];
        let request = Request { options: SOURCE_MEMORY_BIT | (7 << BCPACK_SHIFT), source: source.as_ptr() as u64,
            source_size: source.len() as u32, destination: destination.as_ptr() as u64, destination_size: 64, ..Default::default() };
        assert_eq!(request.check(true), Ok(()));
        assert_eq!(unsafe { execute(&request, None, None, false, None) }, Err(E_DECOMPRESSION));
        let mut captures = Captures::default();
        let binary = unsafe { capture_to(&directory.0, &mut captures, &request) }.unwrap().unwrap();
        assert_eq!(std::fs::read(&binary).unwrap(), source);
        assert!(unsafe { capture_to(&directory.0, &mut captures, &request) }.unwrap().is_none());
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(binary.with_extension("json")).unwrap()).unwrap();
        assert_eq!(metadata["bcpack_mode"], 7);
        assert_eq!(metadata["source_size"], 5);
        assert_eq!(metadata["destination_size"], 64);
        assert_eq!(metadata["source_kind"], "memory");
        assert_eq!(destination, [0xA5; 64]);
        assert_eq!(request.check(true), Ok(()));
        assert_eq!(unsafe { execute(&request, None, None, false, None) }, Err(E_DECOMPRESSION));
    }

    #[test]
    fn bcpack_failure_capture_reads_the_validated_file_offset() {
        let directory = TempDir::new();
        std::fs::create_dir_all(&directory.0).unwrap();
        let source_path = directory.0.join("synthetic-source.bin");
        std::fs::write(&source_path, b"prefixPAYLOADsuffix").unwrap();
        let file = Arc::new(FileInner { id: 1, path: crate::util::wide(&source_path.to_string_lossy()),
            file: Mutex::new(Some(Arc::new(std::fs::File::open(&source_path).unwrap()))) });
        unsafe {
            let object = new_object(file_vtbl(), file);
            let request = Request { options: DEST_PHYS_PAGES_BIT | (7 << BCPACK_SHIFT), source: object as u64,
                offset: 6, source_size: 7, destination: 0x1000, destination_size: 64, ..Default::default() };
            let binary = capture_to(&directory.0, &mut Captures::default(), &request).unwrap().unwrap();
            assert_eq!(std::fs::read(&binary).unwrap(), b"PAYLOAD");
            let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(binary.with_extension("json")).unwrap()).unwrap();
            assert_eq!(metadata["source_kind"], "file");
            assert_eq!(metadata["source_offset"], 6);
            assert_eq!(metadata["destination_is_physical_pages"], true);
            assert_eq!(request.check(false), Ok(()));
            release::<FileInner>(object);
        }
    }

    #[test]
    fn bcpack_mode5_failure_capture_reads_the_validated_file_offset() {
        let directory = TempDir::new();
        std::fs::create_dir_all(&directory.0).unwrap();
        let source_path = directory.0.join("synthetic-mode5-source.bin");
        std::fs::write(&source_path, b"prefixPAYLOADsuffix").unwrap();
        let file = Arc::new(FileInner { id: 1, path: crate::util::wide(&source_path.to_string_lossy()),
            file: Mutex::new(Some(Arc::new(std::fs::File::open(&source_path).unwrap()))) });
        let mut destination = [0xA5u8; 64];
        unsafe {
            let object = new_object(file_vtbl(), Arc::clone(&file));
            let request = Request { options: 5 << BCPACK_SHIFT, source: object as u64,
                offset: 6, source_size: 7, destination: destination.as_mut_ptr() as u64,
                destination_size: destination.len() as u32, ..Default::default() };
            // Native BC5 support admits the request; this deliberately truncated
            // seven-byte stream must still fail during decoding, not validation.
            assert_eq!(request.check(false), Ok(()));
            assert!(crate::hle::bcpack::decode(5, b"PAYLOAD", destination.len()).is_err());
            let binary = capture_to(&directory.0, &mut Captures::default(), &request).unwrap().unwrap();
            assert!(binary.file_name().unwrap().to_string_lossy().starts_with("bcpack5-"));
            assert_eq!(std::fs::read(&binary).unwrap(), b"PAYLOAD");
            let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(binary.with_extension("json")).unwrap()).unwrap();
            assert_eq!(metadata["bcpack_mode"], 5);
            assert_eq!(metadata["source_kind"], "file");
            assert_eq!(metadata["source_offset"], 6);
            assert_eq!(metadata["source_size"], 7);
            assert_eq!(metadata["source_file"], source_path.to_string_lossy().as_ref());
            assert_eq!(execute(&request, Some(&file), None, false, None), Err(E_DECOMPRESSION));
            assert_eq!(destination, [0xA5; 64], "capture must not change the destination or refusal");
            release::<FileInner>(object);
        }
    }

    #[test]
    fn bcpack_failure_capture_caps_samples_and_rejects_oversized_requests_before_reading() {
        let directory = TempDir::new();
        let mut captures = Captures::default();
        for byte in 0..(MAX_SAMPLES + 2) as u8 {
            let source = [byte; 4];
            let mode = if byte % 2 == 0 { 5 } else { 7 };
            let request = Request { options: SOURCE_MEMORY_BIT | (mode << BCPACK_SHIFT), source: source.as_ptr() as u64,
                source_size: 4, destination: 0x1000, destination_size: 64, ..Default::default() };
            let captured = unsafe { capture_to(&directory.0, &mut captures, &request) }.unwrap();
            assert_eq!(captured.is_some(), (byte as usize) < MAX_SAMPLES);
        }
        assert_eq!(captures.seen.len(), MAX_SAMPLES);
        let request = Request { options: SOURCE_MEMORY_BIT | (7 << BCPACK_SHIFT), source: 1,
            source_size: MAX_BYTES + 1, destination: 0x1000, destination_size: 64, ..Default::default() };
        assert!(unsafe { capture_to(&directory.0, &mut Captures::default(), &request) }.unwrap().is_none());
        let mut fresh = Captures::default();
        initialize(&directory.0, &mut fresh).unwrap();
        assert_eq!(fresh.seen.len(), MAX_SAMPLES, "the cap must persist across process runs");
    }
}
