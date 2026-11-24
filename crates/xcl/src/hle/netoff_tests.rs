//! These tests use child processes because installed detours live for the process. Network
//! operations use loopback only. Refusal checks use invalid sockets, numeric addresses and
//! invalid HTTP handles, so even a missing hook cannot contact an external service.

use super::*;
use std::sync::atomic::AtomicI32;
use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

#[link(name = "ws2_32", kind = "raw-dylib")]
extern "system" {
    fn WSAStartup(version: u16, data: *mut u8) -> i32;
    fn WSACleanup() -> i32;
    fn WSAGetLastError() -> i32;
    fn socket(af: i32, ty: i32, protocol: i32) -> usize;
    fn closesocket(s: usize) -> i32;
    fn bind(s: usize, name: *const u8, len: i32) -> i32;
    fn freeaddrinfo(p: *mut c_void);
    fn FreeAddrInfoW(p: *mut c_void);
    fn WSAGetOverlappedResult(s: usize, ov: *mut c_void, transferred: *mut u32, wait: i32, flags: *mut u32) -> i32;
}

fn isolated(name: &str, network: &str, body: impl FnOnce()) {
    if std::env::var("XCL_TEST_NETOFF_CASE").as_deref() == Ok(name) {
        body();
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &format!("hle::netoff::tests::{name}"), "--nocapture"])
        .env("XCL_TEST_NETOFF_CASE", name)
        .env("XCL_NETWORK", network)
        .output().expect("start isolated networking test");
    assert!(output.status.success(), "{name}: {}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
}

unsafe fn system(dll: &str, name: &[u8]) -> usize {
    GetProcAddress(LoadLibraryW(crate::util::wide(dll).as_ptr()), name.as_ptr()).expect("system export") as usize
}

fn v4(a: [u8; 4], port: u16) -> [u8; 16] {
    let p = port.to_be_bytes();
    [2, 0, p[0], p[1], a[0], a[1], a[2], a[3], 0, 0, 0, 0, 0, 0, 0, 0]
}

struct Winsock;
impl Winsock {
    unsafe fn start() -> Self {
        let mut data = [0u64; 64];
        assert_eq!(WSAStartup(0x0202, data.as_mut_ptr() as *mut u8), 0);
        Self
    }
}
impl Drop for Winsock {
    fn drop(&mut self) { unsafe { WSACleanup(); } }
}

#[test]
fn only_loopback_goes_through() {
    let mut v6 = [0u8; 28];
    v6[0] = 23;
    v6[23] = 1;
    unsafe {
        for a in [[127, 0, 0, 1], [127, 255, 12, 3]] { assert!(loopback(v4(a, 80).as_ptr(), 16)); }
        for a in [[192, 0, 2, 1], [10, 0, 0, 1], [0, 0, 0, 0]] { assert!(!loopback(v4(a, 80).as_ptr(), 16)); }
        assert!(loopback(v6.as_ptr(), 28));
        v6[8] = 0x20;
        assert!(!loopback(v6.as_ptr(), 28));
        assert!(!loopback(std::ptr::null(), 28));
        assert!(!loopback(v4([127, 0, 0, 1], 80).as_ptr(), 7));
        assert!(!loopback(v6.as_ptr(), 23));
    }
}

#[test]
fn local_names_are_numeric_or_local() {
    for name in ["", "LOCALHOST", "127.0.0.1", "127.9.8.7", "::1", "0:0:0:0:0:0:0:1"] { assert!(local_name(name), "{name}"); }
    for name in ["192.0.2.1", "2001:db8::1", "127.remote.invalid", "127.0.0.1.invalid", "localhost.invalid"] { assert!(!local_name(name), "{name}"); }
}

fn loopback_tcp() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let mut client = std::net::TcpStream::connect_timeout(&listener.local_addr().unwrap(), std::time::Duration::from_secs(3)).unwrap();
    let (mut server, peer) = listener.accept().unwrap();
    assert!(peer.ip().is_loopback());
    server.set_read_timeout(Some(std::time::Duration::from_secs(3))).unwrap();
    client.write_all(b"offline loopback").unwrap();
    let mut data = [0u8; 16];
    server.read_exact(&mut data).unwrap();
    assert_eq!(&data, b"offline loopback");
}

#[test]
fn the_systems_own_network_functions_refuse() {
    isolated("the_systems_own_network_functions_refuse", "0", || unsafe {
        assert!(enabled());
        let r = super::super::build(&crate::config::Config::default());
        assert!(r.module_by_name(NETOFF).is_some());
        assert!(r.hle_for("ws2_32.dll").is_some());
        for original in [&O_CONNECT, &O_SENDTO, &O_WSAIOCTL, &O_GETADDRINFO, &O_GETADDRINFO_W] { assert_ne!(original.load(Ordering::Acquire), 0, "hook installed"); }
        assert!(O_UNUSED.iter().filter(|o| o.load(Ordering::Acquire) != 0).count() >= 14);
        let _winsock = Winsock::start();
        let send: unsafe extern "system" fn(*mut c_void, *const u16, u32, *const c_void, u32, u32, usize) -> i32 = std::mem::transmute(system("winhttp.dll", b"WinHttpSendRequest\0"));
        SetLastError(0);
        assert_eq!(send(1 as *mut c_void, std::ptr::null(), 0, std::ptr::null(), 0, 0, 0), 0);
        assert_eq!(GetLastError(), ERROR_WINHTTP_NAME_NOT_RESOLVED);
        let open_url: unsafe extern "system" fn(*mut c_void, *const u16, *const u16, u32, u32, usize) -> *mut c_void = std::mem::transmute(system("wininet.dll", b"InternetOpenUrlW\0"));
        assert!(open_url(1 as *mut c_void, crate::util::wide("http://127.0.0.1/").as_ptr(), std::ptr::null(), 0, 0, 0).is_null());
        assert_eq!(GetLastError(), ERROR_WINHTTP_NAME_NOT_RESOLVED);
        let gai: unsafe extern "system" fn(*const u8, *const u8, *const c_void, *mut *mut c_void) -> i32 = std::mem::transmute(system("ws2_32.dll", b"getaddrinfo\0"));
        let gai_w: unsafe extern "system" fn(*const u16, *const u16, *const c_void, *mut *mut c_void) -> i32 = std::mem::transmute(system("ws2_32.dll", b"GetAddrInfoW\0"));
        let mut out = 1 as *mut c_void;
        assert_eq!(gai(b"192.0.2.1\0".as_ptr(), std::ptr::null(), std::ptr::null(), &mut out), WSAHOST_NOT_FOUND);
        assert!(out.is_null());
        assert_eq!(gai(b"127.0.0.1\0".as_ptr(), std::ptr::null(), std::ptr::null(), &mut out), 0);
        assert!(!out.is_null());
        freeaddrinfo(out);
        assert_eq!(gai_w(crate::util::wide("::1").as_ptr(), std::ptr::null(), std::ptr::null(), &mut out), 0);
        assert!(!out.is_null());
        FreeAddrInfoW(out);
        let invalid_address = [0u8; 16];
        assert_eq!(real_connect(usize::MAX, invalid_address.as_ptr(), 16), -1);
        assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        assert_eq!(real_sendto(usize::MAX, b"x".as_ptr(), 1, 0, invalid_address.as_ptr(), 16), -1);
        assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        let s = socket(2, 1, 6);
        assert_ne!(s, usize::MAX);
        let mut f = 0usize;
        let mut returned = 0u32;
        assert_eq!(real_wsai_ioctl(s, SIO_GET_EXTENSION_FUNCTION_POINTER, WSAID_CONNECTEX.as_ptr() as *const c_void, 16, &mut f as *mut usize as *mut c_void, std::mem::size_of::<usize>() as u32, &mut returned, std::ptr::null_mut(), std::ptr::null()), 0);
        assert_eq!(returned, std::mem::size_of::<usize>() as u32);
        assert_eq!(f, connect_ex_detour as usize);
        assert_eq!(closesocket(s), 0);
        let f: unsafe extern "system" fn(usize, *const u8, i32, *const c_void, u32, *mut u32, *mut c_void) -> i32 = std::mem::transmute(f);
        let mut sent = 0x1234;
        assert_eq!(f(usize::MAX, invalid_address.as_ptr(), 16, std::ptr::null(), 0, &mut sent, std::ptr::null_mut()), 0);
        assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        assert_eq!(sent, 0x1234);
        loopback_tcp();
    });
}

#[test]
fn online_process_does_not_install_hooks() {
    isolated("online_process_does_not_install_hooks", "1", || unsafe {
        assert!(!enabled());
        let r = super::super::build(&crate::config::Config::default());
        assert!(r.module_by_name(NETOFF).is_none());
        assert!(r.hle_for("ws2_32.dll").is_none());
        assert!(r.hle_for("winhttp.dll").is_none());
        for original in [&O_CONNECT, &O_SENDTO, &O_WSAIOCTL, &O_GETADDRINFO, &O_GETADDRINFO_W] { assert_eq!(original.load(Ordering::Acquire), 0); }
        assert!(O_UNUSED.iter().all(|o| o.load(Ordering::Acquire) == 0));
        let _winsock = Winsock::start();
        let s = socket(2, 1, 6);
        assert_ne!(s, usize::MAX);
        let mut f = 0usize;
        let mut returned = 0u32;
        assert_eq!(real_wsai_ioctl(s, SIO_GET_EXTENSION_FUNCTION_POINTER, WSAID_CONNECTEX.as_ptr() as *const c_void, 16, &mut f as *mut usize as *mut c_void, std::mem::size_of::<usize>() as u32, &mut returned, std::ptr::null_mut(), std::ptr::null()), 0);
        assert_ne!(f, 0);
        assert_ne!(f, connect_ex_detour as usize);
        closesocket(s);
        loopback_tcp();
    });
}

#[test]
fn connectex_loopback_reaches_the_socket_provider() {
    isolated("connectex_loopback_reaches_the_socket_provider", "0", || unsafe {
        use windows_sys::Win32::System::IO::OVERLAPPED;
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
        use windows_sys::Win32::Foundation::CloseHandle;
        install_system_hooks();
        let _winsock = Winsock::start();
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let target = v4([127, 0, 0, 1], listener.local_addr().unwrap().port());
        let s = socket(2, 1, 6);
        assert_ne!(s, usize::MAX);
        assert_eq!(bind(s, v4([127, 0, 0, 1], 0).as_ptr(), 16), 0);
        let mut f = 0usize;
        let mut returned = 0u32;
        assert_eq!(real_wsai_ioctl(s, SIO_GET_EXTENSION_FUNCTION_POINTER, WSAID_CONNECTEX.as_ptr() as *const c_void, 16, &mut f as *mut usize as *mut c_void, std::mem::size_of::<usize>() as u32, &mut returned, std::ptr::null_mut(), std::ptr::null()), 0);
        assert_eq!(f, connect_ex_detour as usize);
        let event = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
        assert!(!event.is_null());
        let mut ov: OVERLAPPED = std::mem::zeroed();
        ov.hEvent = event;
        let f: unsafe extern "system" fn(usize, *const u8, i32, *const c_void, u32, *mut u32, *mut c_void) -> i32 = std::mem::transmute(f);
        let mut sent = 0;
        let ov_ptr = &mut ov as *mut OVERLAPPED as *mut c_void;
        let result = f(s, target.as_ptr(), 16, std::ptr::null(), 0, &mut sent, ov_ptr);
        if result == 0 { assert_eq!(WSAGetLastError(), 997, "loopback connect pending"); }
        assert_eq!(WaitForSingleObject(event, 5000), 0, "loopback connect completed");
        let mut flags = 0;
        assert_ne!(WSAGetOverlappedResult(s, ov_ptr, &mut sent, 0, &mut flags), 0);
        listener.set_nonblocking(true).unwrap();
        let (_, peer) = listener.accept().unwrap();
        assert!(peer.ip().is_loopback());
        closesocket(s);
        CloseHandle(event);
    });
}

static IOCTL_CALLS: AtomicUsize = AtomicUsize::new(0);
static IOCTL_RESULT: AtomicI32 = AtomicI32::new(0);
static IOCTL_POINTER: AtomicUsize = AtomicUsize::new(0);
static IOCTL_RETURNED: AtomicUsize = AtomicUsize::new(std::mem::size_of::<usize>());
static LAST_PROVIDER: AtomicUsize = AtomicUsize::new(0);
static LAST_ARGS: std::sync::Mutex<[usize; 9]> = std::sync::Mutex::new([0; 9]);

unsafe extern "system" fn fake_ioctl(s: usize, code: u32, input: *const c_void, input_len: u32, output: *mut c_void, output_len: u32, returned: *mut u32, ov: *mut c_void, cb: *const c_void) -> i32 {
    IOCTL_CALLS.fetch_add(1, Ordering::Relaxed);
    *LAST_ARGS.lock().unwrap() = [s, code as usize, input as usize, input_len as usize, output as usize, output_len as usize, returned as usize, ov as usize, cb as usize];
    let result = IOCTL_RESULT.load(Ordering::Relaxed);
    if result != 0 { WSASetLastError(10022); return result; }
    if !returned.is_null() { *returned = IOCTL_RETURNED.load(Ordering::Relaxed) as u32; }
    if !output.is_null() && output_len as usize >= std::mem::size_of::<usize>() {
        let pointer = match IOCTL_POINTER.load(Ordering::Relaxed) {
            0 => if s == 0xa { provider_a as usize } else { provider_b as usize },
            p => p,
        };
        (output as *mut usize).write_unaligned(pointer);
    }
    0
}

unsafe fn provider(id: usize, s: usize, name: *const u8, len: i32, buf: *const c_void, size: u32, sent: *mut u32, ov: *mut c_void) -> i32 {
    LAST_PROVIDER.store(id, Ordering::Relaxed);
    *LAST_ARGS.lock().unwrap() = [s, name as usize, len as usize, buf as usize, size as usize, sent as usize, ov as usize, 0, 0];
    if !sent.is_null() { *sent = size; }
    1
}
unsafe extern "system" fn provider_a(s: usize, n: *const u8, l: i32, b: *const c_void, z: u32, sent: *mut u32, o: *mut c_void) -> i32 { provider(1, s, n, l, b, z, sent, o) }
unsafe extern "system" fn provider_b(s: usize, n: *const u8, l: i32, b: *const c_void, z: u32, sent: *mut u32, o: *mut c_void) -> i32 { provider(2, s, n, l, b, z, sent, o) }

#[test]
fn connectex_queries_and_providers_are_isolated() {
    isolated("connectex_queries_and_providers_are_isolated", "0", || unsafe {
        O_WSAIOCTL.store(fake_ioctl as usize, Ordering::Release);
        let mut output = [0u8; 16];
        let out = output.as_mut_ptr().add(1) as *mut c_void; // unaligned caller buffer
        let mut returned = 0;
        let guid = WSAID_CONNECTEX.as_ptr() as *const c_void;
        let call = |s, code, guid, in_len, out_len, ov, cb| wsai_ioctl(s, code, guid, in_len, out, out_len, &mut returned, ov, cb);
        let mut call = call;
        for s in [0xa, 0xb] {
            assert_eq!(call(s, SIO_GET_EXTENSION_FUNCTION_POINTER, guid, 16, 8, std::ptr::null_mut(), std::ptr::null()), 0);
            assert_eq!((out as *const usize).read_unaligned(), connect_ex_detour as usize);
        }
        let other_guid = [0u8; 16];
        for (code, guid, len) in [(0x1234, guid, 16), (SIO_GET_EXTENSION_FUNCTION_POINTER, other_guid.as_ptr() as *const c_void, 16), (SIO_GET_EXTENSION_FUNCTION_POINTER, guid, 15)] {
            assert_eq!(call(0xa, code, guid, len, 8, std::ptr::null_mut(), std::ptr::null()), 0);
            assert_eq!((out as *const usize).read_unaligned(), provider_a as usize);
        }
        let before = IOCTL_CALLS.load(Ordering::Relaxed);
        for (ov, cb) in [(1 as *mut c_void, std::ptr::null()), (std::ptr::null_mut(), 1 as *const c_void)] {
            (out as *mut usize).write_unaligned(0x1234);
            assert_eq!(call(0xa, SIO_GET_EXTENSION_FUNCTION_POINTER, guid, 16, 8, ov, cb), -1);
            assert_eq!(WSAGetLastError(), WSAEOPNOTSUPP);
            assert_eq!((out as *const usize).read_unaligned(), 0x1234);
        }
        assert_eq!(IOCTL_CALLS.load(Ordering::Relaxed), before);
        // Unrelated asynchronous controls still retain every argument and result.
        assert_eq!(call(0xa, 0x1234, guid, 16, 8, 2 as *mut c_void, 3 as *const c_void), 0);
        let args = *LAST_ARGS.lock().unwrap();
        assert_eq!(&args[..6], &[0xa, 0x1234, guid as usize, 16, out as usize, 8]);
        assert_ne!(args[6], 0);
        assert_eq!(&args[7..], &[2, 3]);
        IOCTL_RESULT.store(-1, Ordering::Relaxed);
        (out as *mut usize).write_unaligned(0x4567);
        assert_eq!(call(0xa, SIO_GET_EXTENSION_FUNCTION_POINTER, guid, 16, 8, std::ptr::null_mut(), std::ptr::null()), -1);
        assert_eq!((out as *const usize).read_unaligned(), 0x4567);
        assert_eq!(WSAGetLastError(), 10022);
        IOCTL_RESULT.store(0, Ordering::Relaxed);
        IOCTL_RETURNED.store(0, Ordering::Relaxed);
        assert_eq!(call(0xa, SIO_GET_EXTENSION_FUNCTION_POINTER, guid, 16, 8, std::ptr::null_mut(), std::ptr::null()), -1);
        assert_eq!(WSAGetLastError(), WSAEFAULT);
        IOCTL_RETURNED.store(8, Ordering::Relaxed);
        assert_eq!(call(0xa, SIO_GET_EXTENSION_FUNCTION_POINTER, guid, 16, 7, std::ptr::null_mut(), std::ptr::null()), -1);
        assert_eq!(WSAGetLastError(), WSAEFAULT);
        let local = v4([127, 0, 0, 1], 80);
        for (s, expected) in [(0xa, 1), (0xb, 2), (0xa, 1)] {
            let mut sent = 99;
            assert_eq!(connect_ex_detour(s, local.as_ptr(), 16, b"abc".as_ptr() as *const c_void, 3, &mut sent, 0x1234 as *mut c_void), 1);
            assert_eq!(LAST_PROVIDER.load(Ordering::Relaxed), expected);
            assert_eq!(sent, 3);
            let args = *LAST_ARGS.lock().unwrap();
            assert_eq!(&args[..5], &[s, local.as_ptr() as usize, 16, b"abc".as_ptr() as usize, 3]);
            assert_ne!(args[5], 0);
            assert_eq!(args[6], 0x1234);
        }
        let calls = IOCTL_CALLS.load(Ordering::Relaxed);
        let mut sent = 99;
        assert_eq!(connect_ex_detour(0xa, v4([192, 0, 2, 1], 443).as_ptr(), 16, std::ptr::null(), 0, &mut sent, std::ptr::null_mut()), 0);
        assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        assert_eq!(IOCTL_CALLS.load(Ordering::Relaxed), calls);
        assert_eq!(sent, 99);
        IOCTL_RESULT.store(-1, Ordering::Relaxed);
        assert_eq!(connect_ex_detour(0xa, local.as_ptr(), 16, std::ptr::null(), 0, &mut sent, std::ptr::null_mut()), 0);
        assert_eq!(WSAGetLastError(), 10022);
        IOCTL_RESULT.store(0, Ordering::Relaxed);
        for pointer in [connect_ex_detour as usize, 1] {
            IOCTL_POINTER.store(pointer, Ordering::Relaxed);
            IOCTL_RETURNED.store(if pointer == 1 { 0 } else { 8 }, Ordering::Relaxed);
            assert_eq!(connect_ex_detour(0xa, local.as_ptr(), 16, std::ptr::null(), 0, &mut sent, std::ptr::null_mut()), 0);
            assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        }
    });
}

static RESOLVER_CALLS: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn fake_resolver(_n: *const u8, _s: *const u8, _h: *const c_void, out: *mut *mut c_void) -> i32 {
    RESOLVER_CALLS.fetch_add(1, Ordering::Relaxed);
    if !out.is_null() { *out = 0x1234 as *mut c_void; }
    1234
}
unsafe extern "system" fn fake_resolver_w(_n: *const u16, _s: *const u16, _h: *const c_void, out: *mut *mut c_void) -> i32 { fake_resolver(std::ptr::null(), std::ptr::null(), std::ptr::null(), out) }

#[test]
fn external_name_refusal_never_calls_the_resolver() {
    isolated("external_name_refusal_never_calls_the_resolver", "0", || unsafe {
        O_GETADDRINFO.store(fake_resolver as usize, Ordering::Release);
        O_GETADDRINFO_W.store(fake_resolver_w as usize, Ordering::Release);
        let mut out = std::ptr::null_mut();
        assert_eq!(getaddrinfo(b"localhost\0".as_ptr() as *const c_void, std::ptr::null(), std::ptr::null(), &mut out), 1234);
        assert_eq!(getaddrinfo_w(crate::util::wide("::1").as_ptr(), std::ptr::null(), std::ptr::null(), &mut out), 1234);
        let calls = RESOLVER_CALLS.load(Ordering::Relaxed);
        for name in ["remote.invalid", "127.remote.invalid", "127.0.0.1.invalid"] {
            let narrow = std::ffi::CString::new(name).unwrap();
            out = 1 as *mut c_void;
            assert_eq!(getaddrinfo(narrow.as_ptr() as *const c_void, std::ptr::null(), std::ptr::null(), &mut out), WSAHOST_NOT_FOUND);
            assert!(out.is_null());
            out = 1 as *mut c_void;
            assert_eq!(getaddrinfo_w(crate::util::wide(name).as_ptr(), std::ptr::null(), std::ptr::null(), &mut out), WSAHOST_NOT_FOUND);
            assert!(out.is_null());
        }
        assert_eq!(RESOLVER_CALLS.load(Ordering::Relaxed), calls);
    });
}
