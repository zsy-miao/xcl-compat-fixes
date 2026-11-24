//! `minuserx.dll` / the console's ntuser API sets: the window functions titles use; the rest
//! is Windows' user32.
//!
//! Console titles create their window with the ANSI functions (CreateWindowExA,
//! RegisterClassExA) but give it DefWindowProcW (XDefiant): on Windows that makes the title
//! text show as CJK characters (8-bit text read as UTF-16). The console's user32 is Unicode
//! underneath, so the ANSI calls are made as Unicode ones here – the window is a Unicode window
//! with the text converted.
//!
//! A console title's window is the screen: no caption or frame, its client area the whole
//! window, 1920x1080 (GetClientRect gives that on every console; titles pick 4K output
//! themselves). On the PC a 1920x1080 overlapped window has a 1904x1041 client area, and a title
//! that sizes its back buffer from it (Minecraft) resolved its 1920x1080 scene past the end of
//! it – the list failed to close and the device was removed. So the title's top-level windows
//! are made frameless, the console's size, kept frameless through style changes
//! (WM_NCCALCSIZE), cloaked and kept off the taskbar: the game's frames are shown in XCL's own
//! window. Both ways of making it go through here (CreateWindowExA and the Unicode
//! CreateWindowExW of the ntuser window API set – Fortnite's stayed a second, white window).

#![allow(non_snake_case)]

use std::ffi::c_void;

use crate::util::{from_ansi_ptr, wide};

#[link(name = "user32")]
extern "system" {
    fn RegisterClassExW(c: *const WndClassEx<u16>) -> u16;
    fn SetWindowLongPtrW(hwnd: *mut c_void, index: i32, value: isize) -> isize;
    fn CallWindowProcW(proc_: isize, hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize;
    fn SetWindowPos(hwnd: *mut c_void, after: *mut c_void, x: i32, y: i32, w: i32, h: i32, flags: u32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn CreateWindowExW(ex_style: u32, class: *const u16, title: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, instance: *mut c_void, param: *const c_void) -> *mut c_void;
}

pub fn exports() -> Vec<(&'static str, usize)> {
    let mut v = crate::hle_exports![CreateWindowExA, RegisterClassExA];
    // titles that make their window the Unicode way (Fortnite) get the console's screen too
    v.push(("CreateWindowExW", create_window_ex_w as usize));
    v
}

/// WNDCLASSEXA (`u8`) / WNDCLASSEXW (`u16`): the same layout, the strings are pointers
#[repr(C)]
pub struct WndClassEx<C> {
    cb_size: u32,
    style: u32,
    wnd_proc: usize,
    cls_extra: i32,
    wnd_extra: i32,
    instance: usize,
    icon: usize,
    cursor: usize,
    background: usize,
    menu_name: *const C,
    class_name: *const C,
    icon_small: usize,
}

/// A class / menu name: an atom (below 0x10000) as it is, a string as UTF-16 (kept alive in
/// `keep`).
unsafe fn name(p: *const u8, keep: &mut Vec<Vec<u16>>) -> *const u16 {
    if (p as usize) < 0x1_0000 {
        return p as *const u16;
    }
    keep.push(wide(&from_ansi_ptr(p).unwrap_or_default()));
    keep.last().unwrap().as_ptr()
}

pub unsafe extern "system" fn RegisterClassExA(c: *const WndClassEx<u8>) -> u16 {
    if c.is_null() {
        return 0;
    }
    if (*c).cb_size != std::mem::size_of::<WndClassEx<u8>>() as u32 {
        windows_sys::Win32::Foundation::SetLastError(windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER);
        return 0;
    }
    let c = &*c;
    let mut keep = Vec::new();
    let w = WndClassEx::<u16> {
        cb_size: std::mem::size_of::<WndClassEx<u16>>() as u32,
        style: c.style,
        wnd_proc: c.wnd_proc,
        cls_extra: c.cls_extra,
        wnd_extra: c.wnd_extra,
        instance: c.instance,
        icon: c.icon,
        cursor: c.cursor,
        background: c.background,
        menu_name: name(c.menu_name, &mut keep),
        class_name: name(c.class_name, &mut keep),
        icon_small: c.icon_small,
    };
    RegisterClassExW(&w)
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "system" fn CreateWindowExA(ex_style: u32, class: *const u8, title: *const u8, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, instance: *mut c_void, param: *const c_void) -> *mut c_void {
    let mut keep = Vec::new();
    let class = name(class, &mut keep);
    let title = if title.is_null() { std::ptr::null() } else { name(title, &mut keep) };
    console_window(ex_style, class, title, style, x, y, w, h, parent, menu, instance, param)
}

/// CreateWindowExW of the console's ntuser API set: as `CreateWindowExA`, without the text
/// conversion.
#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn create_window_ex_w(ex_style: u32, class: *const u16, title: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, instance: *mut c_void, param: *const c_void) -> *mut c_void {
    console_window(ex_style, class, title, style, x, y, w, h, parent, menu, instance, param)
}

#[link(name = "dwmapi")]
extern "system" {
    fn DwmSetWindowAttribute(hwnd: *mut c_void, attr: u32, value: *const c_void, size: u32) -> i32;
}

/// the console's screen as a title's window sees it
const CONSOLE_W: i32 = 1920;
const CONSOLE_H: i32 = 1080;
const WS_CHILD: u32 = 0x4000_0000;
const WS_POPUP: u32 = 0x8000_0000;
/// caption, border, sizing frame, system menu, minimize / maximize boxes
const WS_FRAME: u32 = 0x00C0_0000 | 0x0004_0000 | 0x0008_0000 | 0x0002_0000 | 0x0001_0000;
/// dialog / window / client / static edges
const WS_EX_FRAME: u32 = 0x1 | 0x100 | 0x200 | 0x2_0000;
/// no taskbar button and no Alt+Tab entry (the title's window is never shown: the game is in
/// XCL's window) – a cloaked window kept its taskbar button, a second, empty window
const WS_EX_TOOLWINDOW: u32 = 0x80;
const WS_EX_APPWINDOW: u32 = 0x4_0000;
const GWLP_WNDPROC: i32 = -4;
const WM_NCDESTROY: u32 = 0x82;
const WM_NCCALCSIZE: u32 = 0x83;
const WM_GETMINMAXINFO: u32 = 0x24;
const DWMWA_CLOAK: u32 = 13;

/// the window procedure each console window had (hwnd → procedure)
static PROCS: std::sync::Mutex<Vec<(usize, isize)>> = std::sync::Mutex::new(Vec::new());

/// A console title's top-level window: frameless, 1920x1080, cloaked (see the module notes).
#[allow(clippy::too_many_arguments)]
unsafe fn console_window(ex_style: u32, class: *const u16, title: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, instance: *mut c_void, param: *const c_void) -> *mut c_void {
    if style & WS_CHILD != 0 {
        return CreateWindowExW(ex_style, class, title, style, x, y, w, h, parent, menu, instance, param);
    }
    let style = style & !WS_FRAME | WS_POPUP;
    let ex_style = ex_style & !WS_EX_FRAME & !WS_EX_APPWINDOW | WS_EX_TOOLWINDOW;
    let hwnd = CreateWindowExW(ex_style, class, title, style, 0, 0, CONSOLE_W, CONSOLE_H, parent, menu, instance, param);
    if hwnd.is_null() {
        return hwnd;
    }
    let cloak: i32 = 1;
    DwmSetWindowAttribute(hwnd, DWMWA_CLOAK, &cloak as *const i32 as *const c_void, 4);
    let old = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, console_wndproc as usize as isize);
    if old != 0 {
        PROCS.lock().unwrap().push((hwnd as usize, old));
        // SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
        SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, 0, 0, 0x2 | 0x1 | 0x4 | 0x10 | 0x20);
    }
    crate::xlog_info!("user: the title's window {hwnd:p} is the console's screen ({CONSOLE_W}x{CONSOLE_H}, no frame, cloaked)");
    hwnd
}

/// A console window has no non-client area whatever its style, and may be the screen's size
/// on a smaller PC monitor.
unsafe extern "system" fn console_wndproc(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize {
    let orig = PROCS.lock().unwrap().iter().find(|e| e.0 == hwnd as usize).map(|e| e.1);
    let Some(orig) = orig else { return 0 };
    match msg {
        // the client area is the whole window (wp != 0: rgrc[0] already holds the window rect)
        WM_NCCALCSIZE => 0,
        WM_GETMINMAXINFO => {
            let r = CallWindowProcW(orig, hwnd, msg, wp, lp);
            // MINMAXINFO: ptMaxSize (+8), ptMaxTrackSize (+24)
            if let Some(m) = (lp as *mut [i32; 10]).as_mut() {
                m[2] = m[2].max(CONSOLE_W);
                m[3] = m[3].max(CONSOLE_H);
                m[6] = m[6].max(CONSOLE_W);
                m[7] = m[7].max(CONSOLE_H);
            }
            r
        }
        WM_NCDESTROY => {
            PROCS.lock().unwrap().retain(|e| e.0 != hwnd as usize);
            CallWindowProcW(orig, hwnd, msg, wp, lp)
        }
        _ => CallWindowProcW(orig, hwnd, msg, wp, lp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;

    #[link(name = "user32")]
    extern "system" {
        #[link_name = "RegisterClassExA"]
        fn native_register_class(c: *const WndClassEx<u8>) -> u16;
        fn UnregisterClassW(class: *const u16, instance: usize) -> i32;
        fn GetClassInfoExW(instance: usize, class: *const u16, out: *mut WndClassEx<u16>) -> i32;
        fn DefWindowProcW(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize;
    }

    #[test]
    fn register_class_rejects_invalid_sizes_like_windows() {
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null()) as usize;
            let class_name = c"XclInvalidClassSizeTest";
            let wide_name = wide(class_name.to_str().unwrap());
            let mut c = WndClassEx {
                cb_size: 0, style: 0, wnd_proc: DefWindowProcW as usize,
                cls_extra: 0, wnd_extra: 0, instance, icon: 0, cursor: 0,
                background: 0, menu_name: std::ptr::null(),
                class_name: class_name.as_ptr() as *const u8, icon_small: 0,
            };
            let size = std::mem::size_of_val(&c) as u32;
            let mut results = Vec::new();
            for bad_size in [0, 1, size - 1, size + 1, u32::MAX] {
                c.cb_size = bad_size;
                SetLastError(0x1234_5678);
                let native_atom = native_register_class(&c);
                let native_error = GetLastError();
                if native_atom != 0 {
                    assert_ne!(UnregisterClassW(wide_name.as_ptr(), instance), 0);
                }
                SetLastError(0x1234_5678);
                let atom = RegisterClassExA(&c);
                let error = GetLastError();
                if atom != 0 {
                    assert_ne!(UnregisterClassW(wide_name.as_ptr(), instance), 0);
                }
                eprintln!("cbSize={bad_size}: Windows atom={native_atom:#x}, error={native_error}; XCL atom={atom:#x}, error={error}");
                results.push((bad_size, native_atom, native_error, atom, error));
            }
            for (bad_size, native_atom, native_error, atom, error) in results {
                assert_eq!(native_atom, 0, "Windows must reject cbSize={bad_size}");
                assert_eq!(atom, 0, "XCL must reject cbSize={bad_size}");
                assert_eq!(error, native_error, "cbSize={bad_size}");
            }
        }
    }

    #[test]
    fn register_class_accepts_valid_size_and_preserves_attributes() {
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null()) as usize;
            let class_name = c"XclValidClassSizeTest";
            let wide_name = wide(class_name.to_str().unwrap());
            let c = WndClassEx {
                cb_size: std::mem::size_of::<WndClassEx<u8>>() as u32,
                style: 0x3, wnd_proc: DefWindowProcW as usize,
                cls_extra: 8, wnd_extra: 16, instance, icon: 0, cursor: 0,
                background: 0, menu_name: std::ptr::null(),
                class_name: class_name.as_ptr() as *const u8, icon_small: 0,
            };
            let atom = RegisterClassExA(&c);
            assert_ne!(atom, 0, "valid class: error {}", GetLastError());
            let mut info: WndClassEx<u16> = std::mem::zeroed();
            info.cb_size = std::mem::size_of_val(&info) as u32;
            let found = GetClassInfoExW(instance, wide_name.as_ptr(), &mut info);
            assert_ne!(UnregisterClassW(wide_name.as_ptr(), instance), 0);
            assert_ne!(found, 0);
            assert_eq!(info.style, c.style);
            assert_eq!(info.wnd_proc, c.wnd_proc);
            assert_eq!(info.cls_extra, c.cls_extra);
            assert_eq!(info.wnd_extra, c.wnd_extra);
            assert_eq!(info.instance, instance);
        }
    }

    #[test]
    fn register_class_checks_size_before_reading_remaining_fields() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        unsafe {
            // Only the first eight bytes are readable; the remaining fields lie in
            // the uncommitted second page. Reject cbSize without touching them.
            let base = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            let committed = VirtualAlloc(base, 0x1000, MEM_COMMIT, PAGE_READWRITE);
            if committed.is_null() {
                VirtualFree(base, 0, MEM_RELEASE);
                panic!("commit test page failed");
            }
            let c = (base as *mut u8).add(0x1000 - 8).cast::<WndClassEx<u8>>();
            c.cast::<u32>().write(8);
            SetLastError(0x1234_5678);
            let atom = RegisterClassExA(c);
            let error = GetLastError();
            VirtualFree(base, 0, MEM_RELEASE);
            assert_eq!(atom, 0);
            assert_eq!(error, windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER);
        }
    }
}
