//! Small helpers shared by all modules: strings, HRESULTs, GDK-style string outputs.

use std::ffi::c_void;

pub type HRESULT = i32;

pub const S_OK: HRESULT = 0;
pub const S_FALSE: HRESULT = 1;
pub const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
pub const E_NOINTERFACE: HRESULT = 0x8000_4002u32 as i32;
pub const E_POINTER: HRESULT = 0x8000_4003u32 as i32;
pub const E_ABORT: HRESULT = 0x8000_4004u32 as i32;
pub const E_FAIL: HRESULT = 0x8000_4005u32 as i32;
pub const E_PENDING: HRESULT = 0x8000_000Au32 as i32;
pub const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
pub const E_OUTOFMEMORY: HRESULT = 0x8007_000Eu32 as i32;
pub const E_NOT_SUFFICIENT_BUFFER: HRESULT = 0x8007_007Au32 as i32;
pub const E_NOT_FOUND: HRESULT = 0x8007_0490u32 as i32;
pub const E_NOT_SUPPORTED: HRESULT = 0x8007_0032u32 as i32;

pub fn hresult_from_win32(code: u32) -> HRESULT {
    if code == 0 {
        0
    } else {
        ((code & 0xFFFF) | 0x8007_0000) as i32
    }
}

/// Parse a return value spec used by the stub DB and config: "E_NOTIMPL", "S_OK",
/// "0x80004001", "-1", "42".
pub fn parse_ret(s: &str) -> Option<u64> {
    let t = s.trim();
    let named = match t.to_ascii_uppercase().as_str() {
        "S_OK" | "0" | "FALSE" | "NULL" => Some(0u64),
        "S_FALSE" | "TRUE" => Some(1),
        "E_NOTIMPL" => Some(E_NOTIMPL as u32 as u64),
        "E_NOINTERFACE" => Some(E_NOINTERFACE as u32 as u64),
        "E_FAIL" => Some(E_FAIL as u32 as u64),
        "E_ABORT" => Some(E_ABORT as u32 as u64),
        "E_INVALIDARG" => Some(E_INVALIDARG as u32 as u64),
        "E_NOT_SUPPORTED" => Some(E_NOT_SUPPORTED as u32 as u64),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u64::from_str_radix(h, 16).ok();
    }
    if let Some(n) = t.strip_prefix('-') {
        return n.parse::<u64>().ok().map(|v| (v as i64).wrapping_neg() as u64);
    }
    t.parse::<u64>().ok()
}

/// UTF-16, NUL-terminated.
pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub unsafe fn wcslen(p: *const u16) -> usize {
    let mut n = 0;
    while *p.add(n) != 0 {
        n += 1;
    }
    n
}

pub unsafe fn from_wide_ptr(p: *const u16) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let n = wcslen(p);
    Some(String::from_utf16_lossy(std::slice::from_raw_parts(p, n)))
}

pub unsafe fn from_cstr_ptr(p: *const u8) -> Option<String> {
    if p.is_null() {
        return None;
    }
    Some(
        std::ffi::CStr::from_ptr(p as *const std::ffi::c_char)
            .to_string_lossy()
            .into_owned(),
    )
}

/// ANSI (current code page) → String. Falls back to lossy UTF-8.
pub unsafe fn from_ansi_ptr(p: *const u8) -> Option<String> {
    from_code_page_ptr(p, windows_sys::Win32::Globalization::CP_ACP)
}

pub(crate) unsafe fn from_file_ansi_ptr(p: *const u8) -> Option<String> {
    use windows_sys::Win32::Globalization::{CP_ACP, CP_OEMCP};
    use windows_sys::Win32::Storage::FileSystem::AreFileApisANSI;
    if p.is_null() {
        return None;
    }
    let code_page = if AreFileApisANSI() != 0 {
        CP_ACP
    } else {
        CP_OEMCP
    };
    from_code_page_ptr(p, code_page)
}

unsafe fn from_code_page_ptr(p: *const u8, code_page: u32) -> Option<String> {
    use windows_sys::Win32::Globalization::MultiByteToWideChar;
    if p.is_null() {
        return None;
    }
    let len = std::ffi::CStr::from_ptr(p as *const std::ffi::c_char).to_bytes().len();
    if len == 0 {
        return Some(String::new());
    }
    let n = MultiByteToWideChar(code_page, 0, p, len as i32, std::ptr::null_mut(), 0);
    if n <= 0 {
        return from_cstr_ptr(p);
    }
    let mut buf = vec![0u16; n as usize];
    MultiByteToWideChar(code_page, 0, p, len as i32, buf.as_mut_ptr(), n);
    Some(String::from_utf16_lossy(&buf))
}

/// The process runs under Wine (its ntdll exports `wine_get_version`).
pub fn under_wine() -> bool {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    static WINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // SAFETY: ntdll is always loaded; both names are NUL-terminated.
    *WINE.get_or_init(|| unsafe {
        GetProcAddress(GetModuleHandleA(c"ntdll.dll".as_ptr() as *const u8), c"wine_get_version".as_ptr() as *const u8).is_some()
    })
}

/// Normalised DLL key: lower-case file name, ".dll" appended when no extension.
pub fn dll_key(name: &str) -> String {
    let base = name.rsplit(['\\', '/']).next().unwrap_or(name);
    let mut s = base.trim().to_ascii_lowercase();
    if !s.contains('.') {
        s.push_str(".dll");
    }
    s
}

/// Simple glob: `*` stands for any run of characters (also none), anywhere and as often as the
/// pattern has it ("api-ms-win-core-*", "api-ms-win-*-l1-1-*.dll"); every other character
/// stands for itself.
pub fn glob_match(pattern: &str, s: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(mut rest) = s.strip_prefix(parts.next().unwrap_or("")) else { return false };
    let mut parts: Vec<&str> = parts.collect();
    // no `*`: the whole name; else the last part is the tail, and the parts between the stars
    // are taken in order, each at its first place
    let Some(tail) = parts.pop() else { return rest.is_empty() };
    for part in parts {
        match rest.find(part) {
            Some(i) => rest = &rest[i + part.len()..],
            None => return false,
        }
    }
    rest.ends_with(tail)
}

/// GDK-style "copy string into caller buffer" helper: `size` includes the terminator.
pub unsafe fn write_cstr_out(src: &str, buf: *mut u8, size: usize, used: *mut usize) -> HRESULT {
    let needed = src.len() + 1;
    if !used.is_null() {
        *used = needed;
    }
    if buf.is_null() || size < needed {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    std::ptr::copy_nonoverlapping(src.as_ptr(), buf, src.len());
    *buf.add(src.len()) = 0;
    S_OK
}

/// Win32-style "copy wide string" (GetModuleFileNameW semantics): returns chars written
/// (without terminator) or `size` when truncated.
pub unsafe fn write_wide_out(src: &str, buf: *mut u16, size: u32) -> u32 {
    if buf.is_null() || size == 0 {
        return 0;
    }
    let w: Vec<u16> = src.encode_utf16().collect();
    let n = w.len().min(size as usize - 1);
    std::ptr::copy_nonoverlapping(w.as_ptr(), buf, n);
    *buf.add(n) = 0;
    if w.len() >= size as usize {
        windows_sys::Win32::Foundation::SetLastError(122); // ERROR_INSUFFICIENT_BUFFER
        size
    } else {
        n as u32
    }
}

/// Encode `src` into `buf` (capacity `size` bytes, including NUL) using Windows `codepage`
/// (`GetModuleFileNameA` semantics): returns bytes written (without terminator), or `size` with
/// `ERROR_INSUFFICIENT_BUFFER` when truncated. Truncation always stops at a complete character
/// boundary so multi-byte sequences (DBCS / UTF-8) and UTF-16 surrogate pairs are never split.
unsafe fn write_codepage_out(src: &str, codepage: u32, buf: *mut u8, size: u32) -> u32 {
    use windows_sys::Win32::Foundation::{SetLastError, ERROR_INSUFFICIENT_BUFFER};
    use windows_sys::Win32::Globalization::WideCharToMultiByte;

    if buf.is_null() || size == 0 {
        return 0;
    }
    let w: Vec<u16> = src.encode_utf16().collect();
    if w.is_empty() {
        // SAFETY: `buf` is non-null and `size >= 1`.
        *buf = 0;
        return 0;
    }
    let Ok(w_len) = i32::try_from(w.len()) else {
        // SAFETY: `buf` is non-null and `size >= 1`.
        *buf = 0;
        return 0;
    };
    let needed = WideCharToMultiByte(
        codepage,
        0,
        w.as_ptr(),
        w_len,
        std::ptr::null_mut(),
        0,
        std::ptr::null(),
        std::ptr::null_mut(),
    );
    if needed <= 0 {
        // SAFETY: `buf` is non-null and `size >= 1`.
        *buf = 0;
        return 0;
    }
    let cap = (size as usize - 1).min(i32::MAX as usize);
    if (needed as usize) <= cap {
        let written = WideCharToMultiByte(
            codepage,
            0,
            w.as_ptr(),
            w_len,
            buf,
            needed,
            std::ptr::null(),
            std::ptr::null_mut(),
        );
        if written <= 0 {
            // SAFETY: `buf` is non-null and `size >= 1`.
            *buf = 0;
            return 0;
        }
        // SAFETY: `written <= needed <= cap <= size - 1`, so `buf.add(written)` is in bounds.
        *buf.add(written as usize) = 0;
        return written as u32;
    }

    let written = if cap == 0 {
        0
    } else {
        let mut bounds = Vec::with_capacity(src.len() + 1);
        bounds.push(0usize);
        let mut u16_pos = 0usize;
        for ch in src.chars() {
            u16_pos += ch.len_utf16();
            bounds.push(u16_pos);
        }
        let mut lo = 0usize;
        let mut hi = bounds.len() - 1;
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            let n = WideCharToMultiByte(
                codepage,
                0,
                w.as_ptr(),
                bounds[mid] as i32,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            if n >= 0 && (n as usize) <= cap {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let fit_u16 = bounds[lo];
        if fit_u16 == 0 {
            0
        } else {
            let n = WideCharToMultiByte(
                codepage,
                0,
                w.as_ptr(),
                fit_u16 as i32,
                buf,
                cap as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            n.max(0) as usize
        }
    };
    // SAFETY: `written <= cap <= size - 1`, so `buf.add(written)` is in bounds.
    *buf.add(written) = 0;
    SetLastError(ERROR_INSUFFICIENT_BUFFER);
    size
}

/// Win32-style "copy ANSI string" (GetModuleFileNameA semantics): converts `src` to `CP_ACP`,
/// NUL-terminates, and returns bytes written (without terminator) or `size` when truncated.
pub unsafe fn write_ansi_out(src: &str, buf: *mut u8, size: u32) -> u32 {
    write_codepage_out(src, windows_sys::Win32::Globalization::CP_ACP, buf, size)
}

/// Wrapper to keep raw pointers in statics.
#[derive(Clone, Copy, Debug)]
pub struct SendPtr(pub *mut c_void);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dll_keys() {
        assert_eq!(dll_key("KERNELX"), "kernelx.dll");
        assert_eq!(dll_key("G:\\Bin\\Foo.DLL"), "foo.dll");
        assert_eq!(dll_key("api-ms-win-core-file-l1-1-0.dll"), "api-ms-win-core-file-l1-1-0.dll");
    }

    #[test]
    fn globbing() {
        assert!(glob_match("api-ms-win-core-*", "api-ms-win-core-file-l1-1-0.dll"));
        assert!(!glob_match("api-ms-win-core-*", "api-ms-win-crt-runtime-l1-1-0.dll"));
        assert!(glob_match("xgameruntime*.dll", "xgameruntime.thunks.dll"));
        assert!(glob_match("kernelx.dll", "kernelx.dll"));
    }

    #[test]
    fn globbing_with_several_stars() {
        // an alias for a family of API sets: both stars have to match
        assert!(glob_match("api-ms-win-*-l1-1-*.dll", "api-ms-win-core-file-l1-1-0.dll"));
        assert!(!glob_match("api-ms-win-*-l1-1-*.dll", "api-ms-win-core-file-l1-2-0.dll"));
        assert!(glob_match("a*b*c", "abc"));
        assert!(glob_match("a*b*c", "aXbYbZc"));
        assert!(!glob_match("a*b*c", "acb"));
        assert!(glob_match("*", "anything.dll"));
        assert!(glob_match("**", ""));
        // the parts do not overlap, and the ends of the name are fixed
        assert!(!glob_match("ab*ba", "aba"));
        assert!(!glob_match("a*a", "a"));
        assert!(!glob_match("x*", "ax"));
        assert!(!glob_match("*x", "xa"));
        assert!(glob_match("*.dll", "kernelx.dll"));
        // a name without a star is the whole name
        assert!(!glob_match("kernelx.dll", "kernelx.dll.bak"));
    }

    #[test]
    fn return_specs() {
        assert_eq!(parse_ret("E_NOTIMPL"), Some(0x8000_4001));
        assert_eq!(parse_ret("0x10"), Some(16));
        assert_eq!(parse_ret("-1"), Some(u64::MAX));
        assert_eq!(parse_ret("7"), Some(7));
        assert_eq!(parse_ret("bogus"), None);
    }

    #[test]
    fn cstr_out() {
        let mut buf = [0u8; 8];
        let mut used = 0usize;
        unsafe {
            assert_eq!(write_cstr_out("abc", buf.as_mut_ptr(), 8, &mut used), S_OK);
            assert_eq!(used, 4);
            assert_eq!(&buf[..4], b"abc\0");
            assert_eq!(write_cstr_out("toolongstr", buf.as_mut_ptr(), 8, &mut used), E_NOT_SUFFICIENT_BUFFER);
            assert_eq!(used, 11);
        }
    }

    #[test]
    fn ansi_out_encodes_codepage_and_truncates_on_char_boundaries() {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError, ERROR_INSUFFICIENT_BUFFER};
        use windows_sys::Win32::Globalization::{GetACP, MultiByteToWideChar, CP_UTF8};

        unsafe fn decode_cp(bytes: &[u8], cp: u32) -> String {
            let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            if nul == 0 {
                return String::new();
            }
            let n = MultiByteToWideChar(cp, 0, bytes.as_ptr(), nul as i32, std::ptr::null_mut(), 0);
            assert!(n > 0);
            let mut w = vec![0u16; n as usize];
            MultiByteToWideChar(cp, 0, bytes.as_ptr(), nul as i32, w.as_mut_ptr(), n);
            String::from_utf16_lossy(&w)
        }

        unsafe {
            // Null / zero-size / empty edge cases.
            let mut tiny = [0xAAu8; 4];
            assert_eq!(write_ansi_out("abc", std::ptr::null_mut(), 4), 0);
            assert_eq!(write_ansi_out("abc", tiny.as_mut_ptr(), 0), 0);
            assert_eq!(write_ansi_out("", tiny.as_mut_ptr(), 1), 0);
            assert_eq!(tiny[0], 0);

            // CP1252: non-ASCII French path encodes as single bytes (0xC9 / 0xE9), not UTF-8.
            let fr = "C:\\Jeux\\Été\\game.exe";
            let mut buf = [0xCCu8; 64];
            let n = write_codepage_out(fr, 1252, buf.as_mut_ptr(), buf.len() as u32);
            assert_eq!(n, 20);
            assert_eq!(buf[n as usize], 0);
            assert_eq!(decode_cp(&buf, 1252), fr);

            // CP1250: Central European diacritics ("Č" = 0xC8, "Ž" = 0x8E) round-trip cleanly.
            let cz = "C:\\Hry\\Český_Život\\game.exe";
            buf.fill(0xCC);
            let n = write_codepage_out(cz, 1250, buf.as_mut_ptr(), buf.len() as u32);
            assert_eq!(n, 27);
            assert_eq!(buf[n as usize], 0);
            assert_eq!(decode_cp(&buf, 1250), cz);

            // CJK character outside CP1252 maps cleanly to the codepage default char ('?').
            let cjk = "C:\\Games\\漢\\game.exe";
            buf.fill(0xCC);
            let n = write_codepage_out(cjk, 1252, buf.as_mut_ptr(), buf.len() as u32);
            assert_eq!(n, 19);
            assert_eq!(buf[n as usize], 0);
            assert_eq!(decode_cp(&buf, 1252), "C:\\Games\\?\\game.exe");

            // Active CP_ACP round-trip through write_ansi_out -> from_ansi_ptr.
            let acp = GetACP();
            let sample = match acp {
                1250 => cz,
                1252 | 65001 => fr,
                _ => "C:\\Games\\Title\\game.exe",
            };
            buf.fill(0xCC);
            let n = write_ansi_out(sample, buf.as_mut_ptr(), buf.len() as u32);
            assert!(n > 0 && (n as usize) < buf.len());
            assert_eq!(buf[n as usize], 0);
            assert_eq!(from_ansi_ptr(buf.as_ptr()).as_deref(), Some(sample));

            // Active CP_ACP with CJK path: round-trips if ACP supports it, else substitutes '?' cleanly.
            buf.fill(0xCC);
            let n = write_ansi_out(cjk, buf.as_mut_ptr(), buf.len() as u32);
            assert!(n > 0 && (n as usize) < buf.len());
            assert_eq!(buf[n as usize], 0);
            let decoded_cjk = from_ansi_ptr(buf.as_ptr()).unwrap();
            assert!(
                decoded_cjk == cjk || decoded_cjk == "C:\\Games\\?\\game.exe",
                "unexpected CP_ACP ({acp}) round-trip for CJK path: {decoded_cjk:?}"
            );

            // DBCS truncation (CP932, 2 bytes per Hiragana char): never splits a lead/trail byte.
            let jp = "あいう";
            buf.fill(0xCC);
            SetLastError(0);
            assert_eq!(write_codepage_out(jp, 932, buf.as_mut_ptr(), 2), 2);
            assert_eq!(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(buf[0], 0);

            buf.fill(0xCC);
            SetLastError(0);
            assert_eq!(write_codepage_out(jp, 932, buf.as_mut_ptr(), 4), 4);
            assert_eq!(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(decode_cp(&buf, 932), "あ");
            assert_eq!(buf[2], 0);

            buf.fill(0xCC);
            SetLastError(0);
            assert_eq!(write_codepage_out(jp, 932, buf.as_mut_ptr(), 5), 5);
            assert_eq!(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(decode_cp(&buf, 932), "あい");
            assert_eq!(buf[4], 0);

            buf.fill(0xCC);
            assert_eq!(write_codepage_out(jp, 932, buf.as_mut_ptr(), 7), 6);
            assert_eq!(decode_cp(&buf, 932), jp);
            assert_eq!(buf[6], 0);

            // UTF-8 truncation with a 4-byte emoji (UTF-16 surrogate pair): never splits codepoint.
            let emoji = "a😀b";
            buf.fill(0xCC);
            SetLastError(0);
            assert_eq!(write_codepage_out(emoji, CP_UTF8, buf.as_mut_ptr(), 5), 5);
            assert_eq!(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(&buf[..2], b"a\0");

            buf.fill(0xCC);
            SetLastError(0);
            assert_eq!(write_codepage_out(emoji, CP_UTF8, buf.as_mut_ptr(), 6), 6);
            assert_eq!(GetLastError(), ERROR_INSUFFICIENT_BUFFER);
            assert_eq!(decode_cp(&buf, CP_UTF8), "a😀");
            assert_eq!(buf[5], 0);
        }
    }
}
