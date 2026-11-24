//! xcl.exe – command line front-end and (in host mode) the process the game runs in.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Our own TLS callback: ntdll calls it for every thread start/exit in the process.
/// In host mode it initialises the game's static TLS for the new thread and forwards
/// the notification to the game's TLS callbacks / DllMains.
unsafe extern "system" fn xcl_tls_callback(_h: *mut c_void, reason: u32, _r: *mut c_void) {
    if xcl::host::HOST_MODE.load(Ordering::Acquire) {
        xcl::loader::on_tls_event(reason);
    }
}

#[used]
#[link_section = ".CRT$XLY"]
static XCL_TLS_CALLBACK: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) = xcl_tls_callback;

const HELP: &str = "\
XCL – Xbox Series X|S → Windows compatibility layer

Usage:
  xcl run <game folder | file.exe> [-- game arguments]
      Runs the game. The game folder is only read; writes, game saves and logs go to
      %LOCALAPPDATA%\\XCL\\<title>\\ (or %XCL_DATA_ROOT%\\<title>\\).
  xcl inspect <file.exe|file.dll> [--json]
      Analysis without running: imports and what XCL has for them (HLE / Windows / stub).
  xcl disasm <file> [--offset N] [--length N] [--all] [--hex] [--fssm FILE]
      Disassembler for console GPU code (RDNA2) – e.g. for shaders saved in <data>\\shaders\\.
      Console containers (XDAT/XBCH/FSSM) are unpacked; --fssm saves the unpacked FSSM.
  xcl shaders <file | folder | package.rpf> [--top N] [--json] [--recomp] [--compile N]
      Shader corpus statistics: formats, stages, RDNA2 instruction histogram (priorities
      for the recompiler). RPF9 packages are scanned for shader records.
      --recomp: translates every RDNA2 program → HLSL and shows what is missing;
      --compile N: additionally compiles N complete translations with DXC.
      --export-hlsl DIR [--count N] [--min-kb N] [--record-hash HEX]: writes deterministic
          compute HLSL for investigation; an exact hash bypasses the size threshold.
      --measure-pso HEX [--repeat N]: times cache-disabled driver compilation of one exact
          compute record with XCL's real root signature and static samplers.
      --direct [--sample N] [--seed S] [--validate] [--compare]: direct DXIL translator
      coverage over a seeded sample (default 200 programs, 0 = all); also
      --timeout, --max-bytes, --threads, --json, --hash H, --dump DIR.
  xcl rpf <package.rpf> [--top N] [--find NAME] [--extract NAME --out DIR]
      Lists or extracts readable inline records. Archives larger than 1 GB use a
      bounded-memory header probe; --probe requests it explicitly and --full overrides it.
      Use --probe only for header summaries, without --find, --extract, --out or --full.
  xcl --version
";

fn main() {
    // keep the TLS callback from being discarded by the linker
    unsafe { std::ptr::read_volatile(&XCL_TLS_CALLBACK) };

    if let Some(boot) = std::env::var_os("XCL_HOST") {
        std::env::remove_var("XCL_HOST");
        std::process::exit(xcl::host::host_main(&PathBuf::from(boot)));
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => {
            let rest: Vec<String> = match args.iter().position(|a| a == "--") {
                Some(i) => args[i + 1..].to_vec(),
                None => args[2..].to_vec(),
            };
            xcl::launcher::run(&PathBuf::from(&args[1]), &rest)
        }
        Some("inspect") if args.len() >= 2 => {
            let json = args.iter().any(|a| a == "--json");
            xcl::inspect::run(&PathBuf::from(&args[1]), json)
        }
        Some("disasm") if args.len() >= 2 => xcl::disasm::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("shaders") if args.len() >= 2 => xcl::shaders::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("rpf") if args.len() >= 2 => xcl::rpf::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("--version") | Some("-V") => {
            println!("xcl {}", env!("CARGO_PKG_VERSION"));
            0
        }
        None if started_by_double_click() => run_picked_game(),
        _ => {
            print!("{HELP}");
            if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
                0
            } else {
                2
            }
        }
    };
    std::process::exit(code);
}

/// xcl.exe opened from Explorer: the console is its own (from a terminal, the shell shares it).
fn started_by_double_click() -> bool {
    let mut ids = [0u32; 4];
    unsafe { windows_sys::Win32::System::Console::GetConsoleProcessList(ids.as_mut_ptr(), ids.len() as u32) == 1 }
}

/// No command line: a small menu (choose a game / run the last one again), then the game runs
/// in this console. On an error the console stays open until Enter, so the message can be read.
fn run_picked_game() -> i32 {
    let last_file = xcl::launcher::data_root().join("last-game.txt");
    let last = std::fs::read_to_string(&last_file).ok().map(|s| PathBuf::from(s.trim())).filter(|p| p.exists());
    let target = {
        let _ui = Ui::enter();
        let picked = match menu(last.as_deref()) {
            Choice::Again => last.clone(),
            Choice::Cancel => None,
            Choice::Choose | Choice::NoMenu => pick_game_file(),
        };
        // MicrosoftGame.config stands for its folder: XCL picks the exe the config names
        picked.map(|file| match file.file_name() {
            Some(n) if n.eq_ignore_ascii_case("MicrosoftGame.config") => file.parent().map(PathBuf::from).unwrap_or(file),
            _ => file,
        })
    };
    let Some(target) = target else { return 0 };
    let _ = std::fs::create_dir_all(xcl::launcher::data_root());
    let _ = std::fs::write(&last_file, target.to_string_lossy().as_bytes());
    println!("xcl run \"{}\"", target.display());
    let code = xcl::launcher::run(&target, &[]);
    if code != 0 {
        println!("\nXCL ended with code {code}. Press Enter to close.");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
    code
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `name` of system DLL `dll`, looked up when the menu needs it: xcl.exe does not import the
/// dialog libraries (the game runs in this executable, and must not get them).
unsafe fn system_proc(dll: &str, name: &str) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    let lib = LoadLibraryW(wide(dll).as_ptr());
    if lib.is_null() {
        return None;
    }
    GetProcAddress(lib, format!("{name}\0").as_ptr()).map(|f| f as usize)
}

/// While the menu is open: COM for the file dialog, and the common controls 6 (the task
/// dialog, the current look) through an activation context of its own – not a manifest of
/// xcl.exe, which would apply to the game too.
struct Ui {
    actctx: Option<(windows_sys::Win32::Foundation::HANDLE, usize)>,
}

impl Ui {
    fn enter() -> Ui {
        use windows_sys::Win32::System::ApplicationInstallationAndServicing::{ActivateActCtx, CreateActCtxW, ACTCTXW};
        const MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency><dependentAssembly>
    <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
  </dependentAssembly></dependency>
</assembly>
"#;
        unsafe {
            windows_sys::Win32::System::Com::CoInitializeEx(std::ptr::null(), windows_sys::Win32::System::Com::COINIT_APARTMENTTHREADED as u32);
        }
        let path = std::env::temp_dir().join("xcl-menu.manifest");
        let actctx = std::fs::write(&path, MANIFEST).ok().and_then(|()| unsafe {
            let source = wide(&path.to_string_lossy());
            let mut c: ACTCTXW = std::mem::zeroed();
            c.cbSize = std::mem::size_of::<ACTCTXW>() as u32;
            c.lpSource = source.as_ptr();
            let h = CreateActCtxW(&c);
            if h == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                return None;
            }
            let mut cookie = 0usize;
            (ActivateActCtx(h, &mut cookie) != 0).then_some((h, cookie))
        });
        Ui { actctx }
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        use windows_sys::Win32::System::ApplicationInstallationAndServicing::{DeactivateActCtx, ReleaseActCtx};
        unsafe {
            if let Some((h, cookie)) = self.actctx {
                DeactivateActCtx(0, cookie);
                ReleaseActCtx(h);
            }
            windows_sys::Win32::System::Com::CoUninitialize();
        }
    }
}

enum Choice {
    Choose,
    Again,
    Cancel,
    /// no task dialog on this system: straight to the file dialog
    NoMenu,
}

/// The menu: a task dialog with "Choose a game…" and, after a first run, "Run again".
fn menu(last: Option<&std::path::Path>) -> Choice {
    use windows_sys::Win32::UI::Controls::*;
    const CHOOSE: i32 = 100;
    const AGAIN: i32 = 101;
    type TaskDialogIndirect = unsafe extern "system" fn(*const TASKDIALOGCONFIG, *mut i32, *mut i32, *mut i32) -> i32;
    let Some(f) = (unsafe { system_proc("comctl32.dll", "TaskDialogIndirect") }) else { return Choice::NoMenu };
    let title = wide("XCL");
    let heading = wide("Run an Xbox Series game");
    let content = wide("XCL runs the game from its unpacked files. Choose the game's exe or its MicrosoftGame.config.");
    let again = last.map(|p| wide(&format!("Run again\n{}", p.display())));
    let choose = wide("Choose a game…\nThe game's exe or MicrosoftGame.config");
    let mut buttons = Vec::new();
    if let Some(a) = &again {
        buttons.push(TASKDIALOG_BUTTON { nButtonID: AGAIN, pszButtonText: a.as_ptr() });
    }
    buttons.push(TASKDIALOG_BUTTON { nButtonID: CHOOSE, pszButtonText: choose.as_ptr() });
    let mut c: TASKDIALOGCONFIG = unsafe { std::mem::zeroed() };
    c.cbSize = std::mem::size_of::<TASKDIALOGCONFIG>() as u32;
    c.hwndParent = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    c.dwFlags = TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
    c.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    c.pszWindowTitle = title.as_ptr();
    c.pszMainInstruction = heading.as_ptr();
    c.pszContent = content.as_ptr();
    c.cButtons = buttons.len() as u32;
    c.pButtons = buttons.as_ptr();
    c.nDefaultButton = if again.is_some() { AGAIN } else { CHOOSE };
    let mut pressed = 0i32;
    let hr = unsafe { std::mem::transmute::<usize, TaskDialogIndirect>(f)(&c, &mut pressed, std::ptr::null_mut(), std::ptr::null_mut()) };
    match (hr, pressed) {
        (h, _) if h < 0 => Choice::NoMenu,
        (_, CHOOSE) => Choice::Choose,
        (_, AGAIN) => Choice::Again,
        _ => Choice::Cancel,
    }
}

/// The standard Windows open dialog (None: cancelled).
fn pick_game_file() -> Option<PathBuf> {
    use windows_sys::Win32::UI::Controls::Dialogs::*;
    type GetOpenFileName = unsafe extern "system" fn(*mut OPENFILENAMEW) -> i32;
    let f = unsafe { system_proc("comdlg32.dll", "GetOpenFileNameW") }?;
    // pairs of (description, pattern), each ending in NUL, the list in one more
    let filter: Vec<u16> = "Xbox game (*.exe, MicrosoftGame.config)\0*.exe;MicrosoftGame.config\0All files\0*.*\0\0".encode_utf16().collect();
    let title = wide("XCL – choose a game");
    let mut file = vec![0u16; 32768];
    let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
    ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
    ofn.hwndOwner = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    ofn.lpstrFilter = filter.as_ptr();
    ofn.lpstrFile = file.as_mut_ptr();
    ofn.nMaxFile = file.len() as u32;
    ofn.lpstrTitle = title.as_ptr();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if unsafe { std::mem::transmute::<usize, GetOpenFileName>(f)(&mut ofn) } == 0 {
        return None;
    }
    let n = file.iter().position(|&c| c == 0).unwrap_or(file.len());
    Some(PathBuf::from(String::from_utf16_lossy(&file[..n])))
}
