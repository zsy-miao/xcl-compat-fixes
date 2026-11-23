//! Bounded BC7 BCPack adapter. Exact native subsets run first; an explicitly
//! configured separate research executable may supply fitted-history decoding.

#[path = "bcpack.rs"]
mod entropy;
#[path = "bcpack_bc7_general.rs"]
mod bc7;

use std::io::{Read, Write};
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows_sys::Win32::System::JobObjects::{CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject};
use windows_sys::Win32::System::Threading::{CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess,
    InitializeProcThreadAttributeList, PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROC_THREAD_ATTRIBUTE_JOB_LIST,
    STARTF_USESHOWWINDOW, STARTF_USESTDHANDLES, STARTUPINFOEXW, UpdateProcThreadAttribute, WaitForSingleObject,
    CREATE_NO_WINDOW, EXTENDED_STARTUPINFO_PRESENT};

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
    match bc7::reconstruct(&lanes) {
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
    let deadline = Instant::now() + Duration::from_secs(10);
    // The separate research tool has a smaller per-request limit. Never split a
    // packet: the predictor history and entropy clocks belong to the whole tile.
    if !executable.is_absolute() || !executable.is_file() || expected > MAX_DESTINATION {
        return Err("external BC7 decoder requires an absolute executable path and bounded output".into());
    }
    let capacity = lanes.iter().try_fold(0usize, |sum, lane| sum.checked_add(lane.len()))
        .and_then(|symbols| symbols.checked_mul(4)).and_then(|bytes| bytes.checked_add(128))
        .filter(|&bytes| bytes <= 80 << 20).ok_or_else(|| "external BC7 decoder request is too large".to_string())?;
    let mut request = Vec::new();
    request.try_reserve_exact(capacity).map_err(|_| "external BC7 decoder request allocation failed".to_string())?;
    write!(&mut request, "{{\"expected_bytes\":{expected},\"lanes\":").map_err(|e| e.to_string())?;
    // Serialize the borrowed lanes directly. A json! Value would clone every
    // symbol into a much larger tree while the title is already under memory load.
    serde_json::to_writer(&mut request, lanes).map_err(|e| e.to_string())?;
    request.push(b'}');
    if request.len() > 80 << 20 { return Err("external BC7 decoder request exceeds 80 MiB".into()); }
    let (output, diagnostics) = run_external(executable, &[OsStr::new("--stdio")], request, expected, deadline)?;
    validate_response(output, expected, &diagnostics)
}

/// The job is assigned by CreateProcessW itself, before the helper can execute
/// or spawn descendants. AssignProcessToJobObject after spawn has a race here.
struct HelperJob(OwnedHandle);

impl HelperJob {
    fn new() -> std::io::Result<Self> {
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() { return Err(std::io::Error::last_os_error()); }
            let job = Self(OwnedHandle::from_raw_handle(handle));
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(handle, JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _, std::mem::size_of_val(&limits) as u32) == 0
            { return Err(std::io::Error::last_os_error()); }
            Ok(job)
        }
    }
}

impl Drop for HelperJob {
    fn drop(&mut self) {
        // No breakaway flags are enabled, so this includes inherited-pipe descendants.
        unsafe { TerminateJobObject(self.0.as_raw_handle(), 1); }
        // OwnedHandle closes the final, non-inheritable job handle as a second guard.
    }
}

struct AttributeList(Vec<usize>);

impl AttributeList {
    fn new() -> std::io::Result<Self> {
        unsafe {
            let mut bytes = 0;
            InitializeProcThreadAttributeList(std::ptr::null_mut(), 2, 0, &mut bytes);
            if bytes == 0 { return Err(std::io::Error::last_os_error()); }
            let mut storage = vec![0; bytes.div_ceil(std::mem::size_of::<usize>())];
            if InitializeProcThreadAttributeList(storage.as_mut_ptr().cast(), 2, 0, &mut bytes) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self(storage))
        }
    }
}

impl Drop for AttributeList {
    fn drop(&mut self) { unsafe { DeleteProcThreadAttributeList(self.0.as_mut_ptr().cast()); } }
}

fn command_line(executable: &Path, arguments: &[&OsStr]) -> Vec<u16> {
    let mut line = Vec::new();
    for argument in std::iter::once(executable.as_os_str()).chain(arguments.iter().copied()) {
        if !line.is_empty() { line.push(b' ' as u16); }
        line.push(b'"' as u16);
        let mut slashes = 0;
        for unit in argument.encode_wide() {
            if unit == b'\\' as u16 { slashes += 1; continue; }
            line.extend(std::iter::repeat_n(b'\\' as u16, if unit == b'"' as u16 { slashes * 2 + 1 } else { slashes }));
            line.push(unit);
            slashes = 0;
        }
        line.extend(std::iter::repeat_n(b'\\' as u16, slashes * 2));
        line.push(b'"' as u16);
    }
    line.push(0);
    line
}

fn spawn_helper(executable: &Path, arguments: &[&OsStr], job: &HelperJob)
    -> std::io::Result<(OwnedHandle, std::io::PipeWriter, std::io::PipeReader, std::io::PipeReader)>
{
    let (input, writer) = std::io::pipe()?;
    let (reader, output) = std::io::pipe()?;
    let (errors, diagnostics) = std::io::pipe()?;
    let inherited = [input.as_raw_handle(), output.as_raw_handle(), diagnostics.as_raw_handle()];
    let jobs = [job.0.as_raw_handle()];
    let mut attributes = AttributeList::new()?;
    unsafe {
        for &handle in &inherited {
            if SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) == 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        for (key, ptr, size) in [
            (PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.as_ptr().cast(), std::mem::size_of_val(&inherited)),
            (PROC_THREAD_ATTRIBUTE_JOB_LIST, jobs.as_ptr().cast(), std::mem::size_of_val(&jobs)),
        ] {
            if UpdateProcThreadAttribute(attributes.0.as_mut_ptr().cast(), 0, key as usize,
                ptr, size, std::ptr::null_mut(), std::ptr::null()) == 0
            { return Err(std::io::Error::last_os_error()); }
        }
        let mut startup = STARTUPINFOEXW::default();
        startup.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        startup.StartupInfo.hStdInput = inherited[0];
        startup.StartupInfo.hStdOutput = inherited[1];
        startup.StartupInfo.hStdError = inherited[2];
        startup.lpAttributeList = attributes.0.as_mut_ptr().cast();
        let application: Vec<u16> = executable.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut command = command_line(executable, arguments);
        let mut process = PROCESS_INFORMATION::default();
        if CreateProcessW(application.as_ptr(), command.as_mut_ptr(), std::ptr::null(), std::ptr::null(), 1,
            CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, std::ptr::null(), std::ptr::null(),
            &startup.StartupInfo, &mut process) == 0
        { return Err(std::io::Error::last_os_error()); }
        let process_handle = OwnedHandle::from_raw_handle(process.hProcess);
        drop(OwnedHandle::from_raw_handle(process.hThread));
        Ok((process_handle, writer, reader, errors))
    }
}

enum PipeResult {
    Input(std::io::Result<()>),
    Output(std::io::Result<Vec<u8>>),
    Diagnostics(std::io::Result<Vec<u8>>),
}

fn run_external(executable: &Path, arguments: &[&OsStr], request: Vec<u8>, expected: usize, deadline: Instant)
    -> Result<(Vec<u8>, Vec<u8>), String>
{
    let io_error = |e: std::io::Error| format!("external BC7 decoder I/O failed: {e}");
    let job = HelperJob::new().map_err(io_error)?;
    let (process, mut input, output, diagnostics) = spawn_helper(executable, arguments, &job).map_err(io_error)?;
    let (send, receive) = mpsc::channel();
    let input_send = send.clone();
    std::thread::Builder::new().name("bc7-input".into()).spawn(move || {
        let result = input.write_all(&request);
        drop(input);
        let _ = input_send.send(PipeResult::Input(result));
    }).map_err(io_error)?;
    let output_send = send.clone();
    std::thread::Builder::new().name("bc7-output".into()).spawn(move || {
        let mut bytes = Vec::new();
        let result = output.take(expected as u64 + 1).read_to_end(&mut bytes).map(|_| bytes);
        let _ = output_send.send(PipeResult::Output(result));
    }).map_err(io_error)?;
    std::thread::Builder::new().name("bc7-errors".into()).spawn(move || {
        let mut bytes = Vec::new();
        let result = diagnostics.take(8193).read_to_end(&mut bytes).map(|_| bytes);
        let _ = send.send(PipeResult::Diagnostics(result));
    }).map_err(io_error)?;
    let (mut wrote, mut bytes, mut report, mut exited) = (false, None, None, false);
    loop {
        let remaining = deadline.checked_duration_since(Instant::now())
            .ok_or_else(|| "external BC7 decoder timed out (including pipe I/O)".to_string())?;
        if !exited {
            match unsafe { WaitForSingleObject(process.as_raw_handle(), 0) } {
                WAIT_OBJECT_0 => {
                    let mut code = 0;
                    if unsafe { GetExitCodeProcess(process.as_raw_handle(), &mut code) } == 0 {
                        return Err(io_error(std::io::Error::last_os_error()));
                    }
                    if code != 0 { return Err(format!("external BC7 decoder refused the request (exit {code})")); }
                    exited = true;
                }
                WAIT_TIMEOUT => {}
                _ => return Err(io_error(std::io::Error::last_os_error())),
            }
        }
        if exited && wrote && bytes.is_some() && report.is_some() {
            return Ok((bytes.unwrap(), report.unwrap()));
        }
        match receive.recv_timeout(remaining.min(Duration::from_millis(10))) {
            Ok(PipeResult::Input(result)) => { result.map_err(io_error)?; wrote = true; }
            Ok(PipeResult::Output(result)) => {
                let value = result.map_err(io_error)?;
                if value.len() != expected { return Err("external BC7 decoder returned the wrong output length".into()); }
                bytes = Some(value);
            }
            Ok(PipeResult::Diagnostics(result)) => {
                let value = result.map_err(io_error)?;
                if value.len() > 8192 { return Err("external BC7 decoder exceeded the diagnostics limit".into()); }
                report = Some(value);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) if wrote && bytes.is_some() && report.is_some() => {
                std::thread::sleep(remaining.min(Duration::from_millis(10)));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err("external BC7 decoder pipe worker stopped".into()),
        }
        // No thread joins: every return kills the job, closing descendant pipe ends.
        // The shared deadline also bounds callers if an I/O worker cannot finish.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subprocess_test_name(name: &str) -> String {
        let module = module_path!().split_once("::").unwrap().1;
        format!("{module}::{name}")
    }

    fn pipe_helper_marker() -> std::path::PathBuf {
        let args: Vec<_> = std::env::args().collect();
        let index = args.iter().position(|arg| arg == "--skip").expect("pipe helper marker argument");
        std::env::temp_dir().join(&args[index + 1])
    }

    #[test]
    #[ignore = "subprocess fixture, launched only by external_timeout_includes_inherited_pipes"]
    fn external_pipe_descendant() {
        std::fs::write(pipe_helper_marker(), std::process::id().to_string()).unwrap();
        std::thread::sleep(Duration::from_secs(30));
    }

    #[test]
    #[ignore = "subprocess fixture, launched only by external_timeout_includes_inherited_pipes"]
    fn external_pipe_parent() {
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};
        let marker = pipe_helper_marker();
        let name = marker.file_name().unwrap();
        let descendant = Command::new(std::env::current_exe().unwrap())
            .args([OsStr::new("--ignored"), OsStr::new("--exact"),
                OsStr::new(&subprocess_test_name("external_pipe_descendant")), OsStr::new("--nocapture"),
                OsStr::new("--skip"), name])
            .stdin(Stdio::inherit()).stdout(Stdio::inherit()).stderr(Stdio::inherit())
            .creation_flags(CREATE_NO_WINDOW).spawn().unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        while !marker.exists() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
        assert!(marker.exists(), "descendant must hold the inherited pipes before the parent exits");
        drop(descendant); // Child::drop does not terminate the descendant.
        std::process::exit(0);
    }

    #[test]
    fn external_timeout_includes_inherited_pipes() {
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE};
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let marker = format!("xcl-bc7-pipes-{}-{nonce}", std::process::id());
        let executable = std::env::current_exe().unwrap();
        let helper = subprocess_test_name("external_pipe_parent");
        let start = Instant::now();
        // The parent exits 0 after starting a sleeping descendant. No process
        // reads the input, so a large write stalls too; the descendant also keeps
        // stdout/stderr open. Waiting only for the parent misses all three pipes.
        let result = run_external(&executable, &[OsStr::new("--ignored"), OsStr::new("--exact"),
            OsStr::new(&helper), OsStr::new("--nocapture"), OsStr::new("--skip"), OsStr::new(&marker)],
            vec![b'x'; 1 << 20], 65536, start + Duration::from_secs(2));
        assert!(result.unwrap_err().contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(4), "pipe I/O must not add an unbounded wait");
        let marker_path = std::env::temp_dir().join(marker);
        let pid: u32 = std::fs::read_to_string(&marker_path).expect("descendant started and inherited the pipes").parse().unwrap();
        std::fs::remove_file(marker_path).unwrap();
        unsafe {
            let raw = OpenProcess(PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if !raw.is_null() {
                let descendant = OwnedHandle::from_raw_handle(raw);
                assert_eq!(WaitForSingleObject(descendant.as_raw_handle(), 1000), WAIT_OBJECT_0,
                    "timed-out helper descendants must terminate with their job");
            }
        }
    }

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
