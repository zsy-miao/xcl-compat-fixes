//! Bounded BC7 BCPack adapter. Exact native subsets run first; an explicitly
//! configured separate research executable may supply fitted-history decoding.

#[path = "bcpack.rs"]
mod entropy;
#[path = "bcpack_bc7.rs"]
mod bc7;

use std::io::{Read, Write};
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub(super) const MAX_DESTINATION: usize = entropy::MAX_OUTPUT_SYMBOLS / 2;

pub(super) fn supported_sizes(source: usize, destination: usize) -> bool {
    source > 0 && source <= entropy::MAX_INPUT_BYTES
        && destination > 0 && destination <= MAX_DESTINATION && destination % 16 == 0
}

pub(super) fn decode(source: &[u8], destination: usize) -> Result<Vec<u8>, String> {
    if !supported_sizes(source.len(), destination) {
        return Err("BC7 BCPack sizes exceed the bounded decoder or are not whole blocks".into());
    }
    let lanes = entropy::decode_bc7_entropy(source, destination / 16).map_err(|e| e.to_string())?;
    match bc7::reconstruct_mode6(&lanes) {
        Ok(bytes) => Ok(bytes),
        Err(native_error) => {
            let Some(executable) = std::env::var_os("XCL_BCPACK_EXTERNAL_DECODER").filter(|s| !s.is_empty()) else {
                return Err(native_error);
            };
            external(Path::new(&executable), &lanes, destination)
        }
    }
}

fn validate_response(bytes: Vec<u8>, expected: usize, diagnostics: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len() != expected || expected == 0 || expected % 16 != 0
        || bytes.chunks_exact(16).any(|b| b[0] == 0)
    {
        return Err("external BC7 decoder returned incomplete output or an invalid block prefix".into());
    }
    let report: serde_json::Value = serde_json::from_slice(diagnostics)
        .map_err(|_| "external BC7 decoder diagnostics are malformed".to_string())?;
    if report["accepted"] != true || report["decoded_output_bytes"].as_u64() != Some(expected as u64)
        || ["unfitted_blocks", "invalid_blocks", "normalized_blocks"].iter().any(|key| report[*key].as_u64() != Some(0))
    {
        return Err("external BC7 decoder reported unavailable history, invalid blocks, or normalization".into());
    }
    if report["provisional"] == true {
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| crate::xlog_warn!("DirectStorage: external BCPack BC7 reconstruction uses provisional fitted history; hardware-exact output is not established"));
    }
    Ok(bytes)
}

fn external(executable: &Path, lanes: &[Vec<u8>; 16], expected: usize) -> Result<Vec<u8>, String> {
    // The separate research tool has a smaller per-request limit. Never split a
    // packet: the predictor history and entropy clocks belong to the whole tile.
    if !executable.is_absolute() || !executable.is_file() || expected > 65536 {
        return Err("external BC7 decoder requires an absolute executable path and at most 64 KiB output".into());
    }
    let request = serde_json::to_vec(&serde_json::json!({ "lanes": lanes, "expected_bytes": expected }))
        .map_err(|e| e.to_string())?;
    if request.len() > 4 << 20 { return Err("external BC7 decoder request exceeds 4 MiB".into()); }
    let mut child = Command::new(executable).arg("--stdio").creation_flags(0x0800_0000)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
        .map_err(|e| format!("external BC7 decoder could not start: {e}"))?;
    let mut input = child.stdin.take().unwrap();
    let output = child.stdout.take().unwrap();
    let errors = child.stderr.take().unwrap();
    let writer = std::thread::spawn(move || input.write_all(&request));
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        output.take(expected as u64 + 1).read_to_end(&mut bytes).map(|_| bytes)
    });
    let diagnostics = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        errors.take(8193).read_to_end(&mut bytes).map(|_| bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Ok(None) => break Err("external BC7 decoder timed out".to_string()),
            Err(error) => break Err(format!("external BC7 decoder wait failed: {error}")),
        }
    };
    if status.is_err() { let _ = child.kill(); let _ = child.wait(); }
    let write_result = writer.join().map_err(|_| "external BC7 input writer failed".to_string())?;
    let output = reader.join().map_err(|_| "external BC7 output reader failed".to_string())?
        .map_err(|e| e.to_string())?;
    let diagnostics = diagnostics.join().map_err(|_| "external BC7 diagnostics reader failed".to_string())?
        .map_err(|e| e.to_string())?;
    let status = status?;
    if !status.success() || write_result.is_err() || diagnostics.len() > 8192 {
        return Err(format!("external BC7 decoder refused the request ({status})"));
    }
    validate_response(output, expected, &diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control() -> Vec<u8> {
        // Fixed MIT Xbox2Windows issue-442 packet, without game data/history.
        let text = "101010101010101004040303030304040101010101010140000000000000024002027546281800000101df5f4b7c03410202024110101010040404031010101040041001030303044001080001010104000000080000000041020994e4db27a4410511fe6b020005";
        text.as_bytes().chunks_exact(2).map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(),16).unwrap()).collect()
    }

    #[test]
    fn native_packet_to_virtual_destination_and_failure_leave_complete_results() {
        use super::super::{Request, execute, SOURCE_MEMORY_BIT, E_DECOMPRESSION};
        let source = control();
        let expected = [0x40,0x31,0x72,0x04,0xa5,0x68,0x19,0xbe,0x02,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0];
        let mut destination = [0xA5u8;16];
        let mut request = Request { options: SOURCE_MEMORY_BIT | (7 << 1), source: source.as_ptr() as u64,
            source_size: source.len() as u32, destination: destination.as_mut_ptr() as u64, destination_size: 16, ..Default::default() };
        assert_eq!(request.check(true), Ok(()));
        assert_eq!(unsafe { execute(&request,None,None,None) }, Ok(()));
        assert_eq!(destination, expected);
        destination.fill(0xA5);
        request.source_size -= 1;
        assert_eq!(unsafe { execute(&request,None,None,None) }, Err(E_DECOMPRESSION));
        assert_eq!(destination, [0xA5;16]);
    }

    #[test]
    fn native_file_packet_reaches_scattered_physical_pages() {
        use super::super::{Request, FileInner, execute, DEST_PHYS_PAGES_BIT};
        use crate::hle::kernelx::{AllocateTitlePhysicalPages, FreeTitlePhysicalPages, read_physical_pages, PHYS_PAGE_SIZE};
        use std::sync::{Arc,Mutex};
        let source = control();
        let expected = [0x40,0x31,0x72,0x04,0xa5,0x68,0x19,0xbe,0x02,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0];
        assert!(crate::gpu::uma::get().is_some(), "physical-page BCPack regression requires the local UMA reservation");
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("xcl-bcpack-file-{}-{nonce}.bin", std::process::id()));
        let mut file_bytes = b"prefix".to_vec(); file_bytes.extend(source.iter()); file_bytes.extend(b"suffix");
        std::fs::write(&path,&file_bytes).unwrap();
        let file = FileInner { id: 1, path: crate::util::wide(&path.to_string_lossy()),
            file: Mutex::new(Some(Arc::new(std::fs::File::open(&path).unwrap()))) };
        unsafe {
            let mut count = 2usize;
            let mut pages = [0usize;2];
            assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(),0,&mut count,pages.as_mut_ptr()),1);
            let scattered = [pages[1],pages[0]];
            let offset = (PHYS_PAGE_SIZE - 8) as u16;
            let request = Request { options: DEST_PHYS_PAGES_BIT | (7 << 1), source: 0, offset: 6,
                source_size: source.len() as u32, destination: scattered.as_ptr() as u64, destination_page_offset: offset,
                destination_size: 16, ..Default::default() };
            assert_eq!(request.check(false),Ok(()));
            let result = execute(&request,Some(&file),Some(&scattered),None);
            let mut readback = [0u8;16];
            let read = read_physical_pages(&scattered,offset as usize,&mut readback);
            let freed = FreeTitlePhysicalPages(std::ptr::null_mut(),count,pages.as_mut_ptr());
            assert_eq!(result,Ok(())); assert!(read); assert_eq!(freed,1); assert_eq!(readback,expected);
        }
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn external_output_requires_exact_blocks_and_clean_diagnostics() {
        let mut block = vec![0;16]; block[0] = 0x40;
        let good = br#"{"accepted":true,"decoded_output_bytes":16,"unfitted_blocks":0,"invalid_blocks":0,"normalized_blocks":0}"#;
        assert_eq!(validate_response(block.clone(),16,good).unwrap(), block);
        assert!(validate_response(block.clone(),32,good).is_err());
        assert!(validate_response(vec![0;16],16,good).is_err());
        for key in ["unfitted_blocks", "invalid_blocks", "normalized_blocks"] {
            let mut report: serde_json::Value = serde_json::from_slice(good).unwrap(); report[key] = 1.into();
            assert!(validate_response(block.clone(),16,&serde_json::to_vec(&report).unwrap()).is_err());
        }
        assert!(validate_response(block.clone(),16,b"{}").is_err());
        assert!(validate_response(block,16,b"not json").is_err());
        assert!(!supported_sizes(0,16));
        assert!(!supported_sizes(16,17));
        assert!(!supported_sizes(16,MAX_DESTINATION+16));
    }
}
