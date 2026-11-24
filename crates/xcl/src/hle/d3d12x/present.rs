//! Presentation for the real backend.
//!
//! Console titles do not use DXGI swap chains: they render into their own back buffers
//! and call `PresentX` – command queue slot 30, `PresentX(planeCount, planes, parameters)`
//! with plane 0 `{ u64 frame token; u32 resource count @8; ID3D12Resource** @0x10 }` (as
//! observed on Fortnite). XCL presents that resource in its own window at once and tells the
//! frame clock (`frame`) the token flipped.
//!
//! Until a title's first `PresentX`, a fallback guesses the frame: XCL remembers which
//! resource every RTV descriptor points to and the first render target of each command list,
//! and after ExecuteCommandLists presents the latest back-buffer-like render target – on a
//! call of another console-only queue method, or rate-limited after submissions.
//!
//! The back buffer is copied on the title's own queue (implicit promotion from
//! COMMON/PRESENT state, which consoles require before PresentX).

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicIsize, AtomicU64, Ordering};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

use windows::core::{Interface, HSTRING, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromPoint, MONITORINFO, MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY, MONITOR_FROM_FLAGS};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::CreateEventW;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::gpu::fence::{self, FenceError};
use crate::stubs::StubInfo;

static RTVS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static LISTS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static CANDIDATE: Mutex<Option<(usize, usize)>> = Mutex::new(None); // (queue, resource) – resource AddRef'd
static PRESENTS: AtomicU64 = AtomicU64::new(0);
static LAST_FRAME_DUMP: Mutex<Option<FrameDumpSample>> = Mutex::new(None);
static BOUNDARIES: AtomicU64 = AtomicU64::new(0);
static LAST_BOUNDARY_MS: AtomicU64 = AtomicU64::new(0);
static HWND_RAW: AtomicIsize = AtomicIsize::new(0);
/// the title called PresentX: the fallback guessing is off
static PRESENTX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PRESENTX_LOCK: Mutex<()> = Mutex::new(());

/// command queue #30 PresentX(planeCount, planes, parameters): plane 0's first resource is
/// the frame; S_OK always (a console present does not fail for the title).
pub unsafe extern "system" fn present_x(queue: *mut c_void, plane_count: u32, planes: *const u8, _params: *const u8) -> i32 {
    if plane_count == 0 || planes.is_null() || !crate::gpu::shader::readable(planes as usize, 0x18) {
        return 0;
    }
    let token = (planes as *const u64).read_unaligned();
    let _present = PRESENTX_LOCK.lock().unwrap();
    present_x_frame(queue, plane_count, planes);
    // The title can use flip statistics to decide when to reuse its buffer.
    // Queue our copy before reporting the flip.
    super::frame::mark_flipped(token);
    0
}

unsafe fn present_x_frame(queue: *mut c_void, plane_count: u32, planes: *const u8) {
    let frames = BOUNDARIES.fetch_add(1, Ordering::Relaxed) + 1;
    if frames % 300 == 0 {
        // the title's frame rate, for comparing runs (every 300 frames)
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now = std::time::Instant::now();
        if let Some(t) = LAST.lock().unwrap().replace(now) {
            crate::xlog_info!("D3D12.X: {frames} frames presented, {:.1} fps over the last 300", 300.0 / (now - t).as_secs_f64());
        }
    }
    super::real::stop_if_device_removed();
    if !PRESENTX.swap(true, Ordering::Relaxed) {
        crate::xlog_info!("D3D12.X: PresentX – the title presents its own frames ({plane_count} plane(s))");
    }
    {
        // a change in the number of planes (the console's display composes them – a UI
        // plane over the game's): logged with the planes' raw words
        static LAST: AtomicU64 = AtomicU64::new(1);
        if LAST.swap(plane_count as u64, Ordering::Relaxed) != plane_count as u64 {
            let n = 8 * plane_count as usize;
            let words: Vec<u64> = if crate::gpu::shader::readable(planes as usize, 8 * n) { (0..n).map(|k| (planes as *const u64).add(k).read_unaligned()).collect() } else { Vec::new() };
            crate::xlog_info!("D3D12.X: PresentX now with {plane_count} plane(s): {words:x?}");
        }
    }
    let count = (planes.add(8) as *const u32).read_unaligned();
    let list = (planes.add(0x10) as *const usize).read_unaligned();
    if count == 0 || !crate::gpu::shader::readable(list, 8) || !enabled() {
        return;
    }
    let res = (list as *const usize).read_unaligned();
    if res == 0 {
        return;
    }
    {
        let mut p = PRESENTED.lock().unwrap();
        if !p.contains(&res) {
            p.push(res);
        }
    }
    if trace_frame() {
        super::list_x::dump_pending_geometry();
    }
    if trace_frame() {
        static K: AtomicU64 = AtomicU64::new(0);
        let k = K.fetch_add(1, Ordering::Relaxed);
        if k == 2 || k == 300 {
            super::console_heap::census();
        }
    }
    if trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 3 || n % 50 == 0 {
            let u = USES.lock().unwrap().as_ref().and_then(|m| m.get(&res).copied()).unwrap_or_default();
            let d = ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)).map(|r| r.GetDesc()).map(|d| (d.Width, d.Height, d.Format.0, d.Flags.0));
            let busiest: Vec<String> = {
                let g = USES.lock().unwrap();
                let mut v: Vec<(usize, [u64; 4])> = g.as_ref().map(|m| m.iter().map(|(&k, &u)| (k, u)).collect()).unwrap_or_default();
                v.sort_by_key(|(_, u)| std::cmp::Reverse(u[0] + u[1] + u[2]));
                v.iter().take(6).map(|(k, u)| format!("{k:#x}: rt {} uav {} copy {} clear {}", u[0], u[1], u[2], u[3])).collect()
            };
            crate::xlog_info!("frame trace: PresentX #{n} of {res:#x} {:?} – rt binds {}, UAV views {}, copy destination {}, clears {}; most used: {}", d, u[0], u[1], u[2], u[3], busiest.join("; "));
        }
    }
    present_now(Duration::ZERO, Some((queue as usize, res)));
}

thread_local! {
    static IN_PRESENT: Cell<bool> = const { Cell::new(false) };
}

fn now_ms() -> u64 {
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub fn counters() -> (u64, u64) {
    (PRESENTS.load(Ordering::Relaxed), BOUNDARIES.load(Ordering::Relaxed))
}

fn enabled() -> bool {
    !crate::state::config().present.eq_ignore_ascii_case("off")
}

unsafe fn add_ref(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(1));
    f(p);
}

unsafe fn release(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(2));
    f(p);
}

pub unsafe fn note_rtv(handle: usize, res: *mut c_void) {
    if !enabled() {
        return;
    }
    let mut g = RTVS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !res.is_null() {
        add_ref(res as usize);
    }
    let old = if res.is_null() { m.remove(&handle) } else { m.insert(handle, res as usize) };
    if let Some(o) = old {
        release(o);
    }
}

/// `XCL_TRACE_FRAME=1`: what console code does with each resource it later presents –
/// render-target binds, UAV views, copy destinations, clears (logged at the first presents)
static USES: Mutex<Option<HashMap<usize, [u64; 4]>>> = Mutex::new(None);
pub const USE_RT: usize = 0;
pub const USE_UAV: usize = 1;
pub const USE_COPY: usize = 2;
pub const USE_CLEAR: usize = 3;

pub fn trace_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_TRACE_FRAME").is_some())
}

/// `res` was used as `what` (USE_*)
pub fn note_use(res: usize, what: usize) {
    if trace_frame() && res != 0 {
        USES.lock().unwrap().get_or_insert_with(HashMap::new).entry(res).or_default()[what] += 1;
    }
}

/// resources the title presented
static PRESENTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub fn is_presented(res: usize) -> bool {
    PRESENTED.lock().unwrap().contains(&res)
}

/// the resource an RTV handle names (render-target binds)
pub fn rtv_resource(handle: usize) -> Option<usize> {
    RTVS.lock().unwrap().as_ref()?.get(&handle).copied()
}

/// The resource bound as render target 0 on `list` (`XCL_TRACE_FRAME`).
pub fn rt0_resource(list: usize) -> Option<usize> {
    let h = *LISTS.lock().unwrap().as_ref()?.get(&list)?;
    rtv_resource(h)
}

pub fn note_rt0(list: usize, handle: usize) {
    if !enabled() {
        return;
    }
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(list, handle);
}

/// Format the swap chain uses for a back buffer of `f` (None = not presentable).
pub fn swap_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

unsafe fn backbuffer_like(res: usize) -> Option<D3D12_RESOURCE_DESC> {
    let raw = res as *mut c_void;
    let r = ID3D12Resource::from_raw_borrowed(&raw)?;
    let d = r.GetDesc();
    let ok = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Width >= 640
        && d.Height >= 360
        && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1
        && swap_format(d.Format).is_some();
    ok.then_some(d)
}

/// queues the title submitted work to (AddRef'd: they stay valid for `drain`)
static QUEUES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// The queues the title submitted work to.
pub fn known_queues() -> Vec<usize> {
    QUEUES.lock().unwrap().clone()
}

static EXECUTES: AtomicU64 = AtomicU64::new(0);

/// (PresentX calls, ExecuteCommandLists calls) so far
pub fn activity() -> (u64, u64) {
    (PRESENTS.load(Ordering::Relaxed), EXECUTES.load(Ordering::Relaxed))
}

pub unsafe fn note_execute(queue: *mut c_void, lists: &[*mut c_void]) {
    EXECUTES.fetch_add(1, Ordering::Relaxed);
    {
        let mut q = QUEUES.lock().unwrap();
        if !q.contains(&(queue as usize)) {
            add_ref(queue as usize);
            q.push(queue as usize);
        }
    }
    if !enabled() || IN_PRESENT.with(|c| c.get()) || PRESENTX.load(Ordering::Relaxed) {
        return;
    }
    let handle = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        lists.iter().rev().find_map(|l| m.get(&(*l as usize)).copied())
    };
    let Some(h) = handle else { return };
    let res = RTVS.lock().unwrap().as_ref().and_then(|m| m.get(&h).copied());
    let Some(res) = res else { return };
    if backbuffer_like(res).is_none() {
        return;
    }
    {
        let mut c = CANDIDATE.lock().unwrap();
        if c.map(|(_, r)| r) != Some(res) {
            add_ref(res);
            if let Some((_, old)) = c.replace((queue as usize, res)) {
                release(old);
            }
        }
    }
    // no console present call seen recently → present on submissions (rate limited)
    if now_ms().saturating_sub(LAST_BOUNDARY_MS.load(Ordering::Relaxed)) > 500 {
        present_now(Duration::from_millis(15), None);
    }
}

/// StubHook for console-only queue methods (PresentX, frame events, …).
pub fn on_queue_console_method(_info: &StubInfo, _caller: usize) {
    if !enabled() || PRESENTX.load(Ordering::Relaxed) {
        return;
    }
    BOUNDARIES.fetch_add(1, Ordering::Relaxed);
    LAST_BOUNDARY_MS.store(now_ms(), Ordering::Relaxed);
    unsafe { present_now(Duration::from_millis(4), None) };
}

// ---------------------------------------------------------------------------------------
// window
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    // keyboard and mouse as the gamepad: raw mouse input, the cursor held in the window
    if crate::hle::kbm::on_message(hwnd.0, msg, wp.0, lp.0) {
        return LRESULT(0);
    }
    match msg {
        WM_CLOSE => {
            crate::xlog_info!("XCL window closed – ending the game process");
            crate::report::write_final(None);
            crate::log::flush();
            windows::Win32::System::Threading::ExitProcess(0);
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }
        WM_TIMER if wp.0 == COMPILE_TIMER => {
            show_compiling(hwnd);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wp, lp),
    }
}

/// The game's name in the XCL window's title: its display name from MicrosoftGame.config,
/// else its package identity.
fn window_name() -> String {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let Some(b) = crate::state::boot() else { return "XCL".into() };
        crate::launcher::display_name(&b.game_root).unwrap_or_else(|| b.title.clone())
    })
    .clone()
}

/// the XCL window's timer for `show_compiling`
const COMPILE_TIMER: usize = 0x5C0;

struct FrameRate {
    sampled_at: Instant,
    presented: u64,
    fps: f64,
}

impl FrameRate {
    fn new(now: Instant, presented: u64) -> Self {
        Self {
            sampled_at: now,
            presented,
            fps: 0.0,
        }
    }

    fn sample(&mut self, now: Instant, presented: u64) -> f64 {
        let elapsed = now.duration_since(self.sampled_at);
        if elapsed >= Duration::from_secs(1) {
            self.fps = presented.saturating_sub(self.presented) as f64 / elapsed.as_secs_f64();
            self.sampled_at = now;
            self.presented = presented;
        }
        self.fps
    }
}

/// While XCL compiles shaders (DXC, programs the disk cache did not have – the title draws
/// nothing with those pipelines meanwhile: after an XCL update the screen stays black while
/// thousands recompile), the window's title says so:
/// "XCL – <game> | N FPS | compiling shaders: N running, M done[, Q pipelines queued][, F failed]".
/// It stays 2 s after the last compile (compiles in the title's own calls come one by one),
/// then only the game name and presented frame rate remain.
unsafe fn show_compiling(hwnd: HWND) {
    static SHOWN: Mutex<Option<String>> = Mutex::new(None);
    static LAST_ACTIVE: Mutex<Option<Instant>> = Mutex::new(None);
    static FRAME_RATE: Mutex<Option<FrameRate>> = Mutex::new(None);
    let now = Instant::now();
    let presented = PRESENTS.load(Ordering::Relaxed);
    let fps = FRAME_RATE
        .lock()
        .unwrap()
        .get_or_insert_with(|| FrameRate::new(now, presented))
        .sample(now, presented);
    let (_, failed, queued) = super::pso_async::counts();
    let (running, done) = crate::gpu::recomp::dxc::compile_counts();
    let active = running > 0 || queued > 0;
    let recent = {
        let mut last = LAST_ACTIVE.lock().unwrap();
        if active {
            *last = Some(Instant::now());
        }
        last.is_some_and(|t| t.elapsed() < Duration::from_secs(2))
    };
    let title = format!("XCL – {} | {fps:.1} FPS", window_name());
    let text = if recent {
        let queued = if queued > 0 { format!(", {queued} pipelines queued") } else { String::new() };
        let failed = if failed > 0 { format!(", {failed} failed") } else { String::new() };
        format!("{title} | compiling shaders: {running} running, {done} done{queued}{failed}")
    } else {
        title
    };
    let mut shown = SHOWN.lock().unwrap();
    if shown.as_deref() != Some(text.as_str()) {
        let _ = SetWindowTextW(hwnd, &HSTRING::from(text.as_str()));
        *shown = Some(text);
    }
}

fn window_thread() {
    unsafe {
        let Ok(inst) = GetModuleHandleW(PCWSTR::null()) else { return };
        let class = windows::core::w!("XCLGameWindow");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: inst.into(),
            lpszClassName: class,
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            ..Default::default()
        };
        RegisterClassW(&wc);
        let cfg = crate::state::config();
        let title = window_name();
        let mut rc = RECT { left: 0, top: 0, right: cfg.window_width as i32, bottom: cfg.window_height as i32 };
        let _ = AdjustWindowRect(&mut rc, WS_OVERLAPPEDWINDOW, false);
        let (w, h) = (rc.right - rc.left, rc.bottom - rc.top);
        let (x, y) = window_position(&cfg.window_pos, w, h);
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            &HSTRING::from(format!("XCL – {title}")),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            x,
            y,
            w,
            h,
            None,
            None,
            Some(inst.into()),
            None,
        );
        let Ok(hwnd) = hwnd else {
            crate::xlog_error!("could not create the XCL window");
            return;
        };
        HWND_RAW.store(hwnd.0 as isize, Ordering::Release);
        crate::xlog_info!("XCL window created ({}x{})", cfg.window_width, cfg.window_height);
        crate::hle::kbm::attach(hwnd.0);
        SetTimer(Some(hwnd), COMPILE_TIMER, 500, None);
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// The XCL window's top-left from `window_pos` (`XCL_WINDOW_POS` overrides): "auto" keeps
/// CW_USEDEFAULT (Windows may pick any monitor), "center" centers on the primary monitor's
/// work area, "x,y" is clamped into the nearest monitor's work area — a multi-monitor desktop
/// otherwise gets the window on whichever monitor Windows' cascade picks.
unsafe fn window_position(setting: &str, w: i32, h: i32) -> (i32, i32) {
    let env = std::env::var("XCL_WINDOW_POS").ok();
    let mode = env.as_deref().unwrap_or(setting).trim();
    if mode.is_empty() || mode.eq_ignore_ascii_case("auto") {
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    }
    let work = |pt: POINT, flags: u32| -> Option<RECT> {
        let mon = MonitorFromPoint(pt, MONITOR_FROM_FLAGS(flags));
        if mon.is_invalid() {
            return None;
        }
        let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
        GetMonitorInfoW(mon, &mut mi).as_bool().then_some(mi.rcWork)
    };
    if mode.eq_ignore_ascii_case("center") {
        let Some(r) = work(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY.0) else {
            return (CW_USEDEFAULT, CW_USEDEFAULT);
        };
        return (r.left + ((r.right - r.left) - w).max(0) / 2, r.top + ((r.bottom - r.top) - h).max(0) / 2);
    }
    let Some((xs, ys)) = mode.split_once(',') else {
        crate::xlog_warn!("window_pos: expected \"auto\", \"center\" or \"x,y\", got {mode:?} – using auto");
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    };
    let (Ok(x), Ok(y)) = (xs.trim().parse::<i32>(), ys.trim().parse::<i32>()) else {
        crate::xlog_warn!("window_pos: expected \"auto\", \"center\" or \"x,y\", got {mode:?} – using auto");
        return (CW_USEDEFAULT, CW_USEDEFAULT);
    };
    // clamp into the work area of the monitor nearest the requested top-left
    let Some(r) = work(POINT { x, y }, MONITOR_DEFAULTTONEAREST.0) else {
        return (x, y);
    };
    (x.clamp(r.left, (r.right - w).max(r.left)), y.clamp(r.top, (r.bottom - h).max(r.top)))
}

fn ensure_window() -> Option<HWND> {
    static START: Once = Once::new();
    START.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-window".into()).spawn(window_thread);
    });
    let t0 = Instant::now();
    while HWND_RAW.load(Ordering::Acquire) == 0 {
        if t0.elapsed() > Duration::from_secs(3) {
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Some(HWND(HWND_RAW.load(Ordering::Acquire) as *mut c_void))
}

// ---------------------------------------------------------------------------------------
// presenter
// ---------------------------------------------------------------------------------------

struct Presenter {
    queue: usize,
    dev: ID3D12Device,
    sc: Option<IDXGISwapChain3>,
    w: u32,
    h: u32,
    fmt: DXGI_FORMAT,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    copy_source: Option<ID3D12Resource>,
    dumps: Vec<FrameDump>,
    disabled: bool,
    value: u64,
    event: HANDLE,
    last: Instant,
}

struct FrameDump {
    _source: ID3D12Resource,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    format: DXGI_FORMAT,
    tag: String,
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

unsafe fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::transmute_copy(res),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

unsafe fn new_presenter(queue: usize) -> Option<Presenter> {
    let dev_raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&dev_raw)?;
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
    list.Close().ok()?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
    let event = CreateEventW(None, false, false, None).ok()?;
    Some(Presenter {
        queue, dev: dev.clone(), sc: None, w: 0, h: 0, fmt: DXGI_FORMAT_UNKNOWN, alloc, list, fence,
        copy_source: None, dumps: Vec::new(), disabled: false, value: 0, event,
        last: Instant::now() - Duration::from_secs(1),
    })
}

unsafe fn ensure_swapchain(p: &mut Presenter, hwnd: HWND, w: u32, h: u32, fmt: DXGI_FORMAT) -> bool {
    if let Some(sc) = &p.sc {
        if p.w == w && p.h == h && p.fmt == fmt {
            return true;
        }
        if sc.ResizeBuffers(2, w, h, fmt, DXGI_SWAP_CHAIN_FLAG(0)).is_ok() {
            p.w = w;
            p.h = h;
            p.fmt = fmt;
            crate::xlog_info!("present: swap chain {w}x{h} {:?}", fmt);
            return true;
        }
        return false;
    }
    let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory2>(DXGI_CREATE_FACTORY_FLAGS(0)) else { return false };
    let qraw = p.queue as *mut c_void;
    let Some(queue) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return false };
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: w,
        Height: h,
        Format: fmt,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2,
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        ..Default::default()
    };
    match factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None) {
        Ok(sc1) => {
            let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
            p.sc = sc1.cast::<IDXGISwapChain3>().ok();
            p.w = w;
            p.h = h;
            p.fmt = fmt;
            crate::xlog_info!("present: swap chain {w}x{h} {:?} – the game image will appear in the XCL window", fmt);
            p.sc.is_some()
        }
        Err(e) => {
            crate::xlog_error!("present: CreateSwapChainForHwnd: {e}");
            false
        }
    }
}

unsafe fn present_now(min_interval: Duration, frame: Option<(usize, usize)>) {
    if IN_PRESENT.with(|c| c.replace(true)) {
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_PRESENT.with(|c| c.set(false));
        }
    }
    let _reset = Reset;
    let (queue, src, mut guard) = if let Some((queue, res)) = frame {
        let raw = res as *mut c_void;
        let Some(src) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
        (queue, src.clone(), PRESENTER.lock().unwrap())
    } else {
        let Ok(guard) = PRESENTER.try_lock() else { return };
        if PRESENTX.load(Ordering::Relaxed) {
            return;
        }
        let candidate = CANDIDATE.lock().unwrap();
        let Some((queue, res)) = *candidate else { return };
        let raw = res as *mut c_void;
        let Some(src) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
        (queue, src.clone(), guard)
    };
    // PresentX on another queue than the presenter was made for (the fallback presented once from
    // a queue before the title's first PresentX, issue #103): the swap chain belongs to a queue –
    // a new presenter for the title's present queue, after the old one's copy is done
    if frame.is_some() && guard.as_ref().is_some_and(|p| p.queue != queue) {
        if let Some(old) = guard.as_ref() {
            let _ = fence::wait(&old.fence, old.value, &old.dev, Some(old.event), 1000);
        }
        crate::xlog_info!("D3D12.X: PresentX from another queue ({queue:#x}) – the XCL window's swap chain moves to it");
        *guard = None;
    }
    if guard.is_none() {
        *guard = new_presenter(queue);
    }
    let Some(p) = guard.as_mut() else { return };
    if p.disabled || p.last.elapsed() < min_interval {
        return;
    }
    let Some(desc) = backbuffer_like(src.as_raw() as usize) else { return };
    let Some(fmt) = swap_format(desc.Format) else { return };
    let Some(hwnd) = ensure_window() else { return };
    if p.queue != queue {
        return; // swap chain is bound to the first queue we presented from
    }
    // wait for the previous copy so the allocator can be reused
    if let Err(e) = fence::wait(&p.fence, p.value, &p.dev, Some(p.event), 100) {
        p.disabled = matches!(&e, FenceError::Removed(_));
        if !matches!(&e, FenceError::Timeout(_)) {
            crate::xlog_error!("present: previous copy: {e}");
        }
        return;
    }
    p.copy_source = None;
    p.dumps.clear();
    if !ensure_swapchain(p, hwnd, desc.Width as u32, desc.Height, fmt) {
        return;
    }
    let Some(sc) = p.sc.clone() else { return };
    let Ok(bb) = sc.GetBuffer::<ID3D12Resource>(sc.GetCurrentBackBufferIndex()) else { return };
    if p.alloc.Reset().is_err() || p.list.Reset(&p.alloc, None).is_err() {
        return;
    }
    let b1 = [transition(&bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST)];
    p.list.ResourceBarrier(&b1);
    p.list.CopyResource(&bb, &src);
    let due = frame_dump_due();
    let mut dumps = Vec::new();
    if due {
        if let Some((buffer, footprint)) = readback_copy(&p.list, &src, &desc) {
            dumps.push(FrameDump { _source: src.clone(), buffer, footprint, format: desc.Format, tag: "frame".into() });
        }
    }
    // with `XCL_TRACE_FRAME`: also the render target the title binds most (its scene / UI)
    // (and the targets the watched one's contents went on to: rt_<n>_<k>)
    if due && trace_frame() {
        let busiest = busiest_rt();
        let followed = FOLLOWED.lock().unwrap();
        let targets = busiest
            .into_iter()
            .chain(followed.iter().copied())
            .filter_map(|r| {
                let rr = r as *mut c_void;
                let res = ID3D12Resource::from_raw_borrowed(&rr)?.clone();
                let d = res.GetDesc();
                if d.SampleDesc.Count > 1 {
                    return None;
                }
                let (buffer, footprint) = readback_copy(&p.list, &res, &d)?;
                Some((res, buffer, footprint, d.Format))
            });
        for (k, (res, buffer, footprint, format)) in targets.enumerate() {
            let tag = if k == 0 { "rt".into() } else { format!("rt{k}") };
            dumps.push(FrameDump { _source: res, buffer, footprint, format, tag });
        }
    }
    let b2 = [transition(&bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)];
    p.list.ResourceBarrier(&b2);
    if p.list.Close().is_err() {
        return;
    }
    let qraw = queue as *mut c_void;
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&qraw) else { return };
    let Ok(cl) = p.list.cast::<ID3D12CommandList>() else { return };
    q.ExecuteCommandLists(&[Some(cl)]);
    p.copy_source = Some(src);
    p.dumps = dumps;
    p.value += 1;
    if let Err(e) = q.Signal(&p.fence, p.value) {
        // The copy is already submitted. Without its signal, keep its resources and allocator.
        p.disabled = true;
        let e = fence::api_error(&p.dev, "Signal", e);
        crate::xlog_error!("present: submit copy: {e}");
        return;
    }
    if let Err(e) = sc.Present(0, DXGI_PRESENT(0)).ok() {
        let e = fence::api_error(&p.dev, "Present", e);
        p.disabled = matches!(&e, FenceError::Removed(_));
        crate::xlog_error!("present: {e}");
        return;
    }
    p.last = Instant::now();
    if !p.dumps.is_empty() {
        match fence::wait(&p.fence, p.value, &p.dev, Some(p.event), 2000) {
            Ok(()) => {
                for dump in p.dumps.drain(..) {
                    write_frame(&dump.buffer, &dump.footprint, dump.format, &dump.tag);
                }
            }
            Err(e) => {
                p.disabled = matches!(&e, FenceError::Removed(_));
                crate::xlog_error!("present: frame dump copy: {e}");
                if p.disabled {
                    return;
                }
            }
        }
    }
    let n = PRESENTS.fetch_add(1, Ordering::Relaxed);
    if n == 0 {
        crate::xlog_info!("present: first game frame shown ({}x{})", desc.Width, desc.Height);
    }
}

/// `XCL_FRAME_DUMP=1`: every 120th shown frame is written to the logs folder (`XCL_FRAME_DUMP=n`,
/// n > 1: every n-th; a title loading at a few frames a second shows few)
/// (`frame_<n>.bmp`, a quarter of each side; `XCL_FRAME_DUMP_FULL=1`: full size) – what the title presented, before the window.
fn frame_dump_due() -> bool {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| match std::env::var("XCL_FRAME_DUMP") {
        Ok(v) => v.parse().ok().filter(|&n| n > 1).unwrap_or(120),
        Err(_) => 0,
    });
    every > 0 && PRESENTS.load(Ordering::Relaxed) % every == 20.min(every - 1)
}

/// Frames shown so far.
pub fn presents() -> u64 {
    PRESENTS.load(Ordering::Relaxed)
}

/// Content sampled by an existing `XCL_FRAME_DUMP` readback, not proof of gameplay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameDumpSample {
    /// Zero-based presentation index, as in `frame_<n>.bmp`.
    pub frame: u64,
    pub width: u32,
    pub height: u32,
    pub format: i32,
    /// One texel every `sample_step` pixels in each direction (normally 4).
    pub sample_step: u32,
    pub sampled_pixels: u64,
    /// Samples whose decoded 8-bit R+G+B exceeds 30; None for an undecoded format.
    pub non_dark_samples: Option<u64>,
}

pub fn last_frame_dump() -> Option<FrameDumpSample> {
    LAST_FRAME_DUMP.lock().ok().and_then(|s| *s)
}

/// a copy of subresource 0 of `src` into a new readback buffer, recorded on `list`
unsafe fn readback_copy(list: &ID3D12GraphicsCommandList, src: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let dev: ID3D12Device = {
        let mut d: Option<ID3D12Device> = None;
        src.GetDevice(&mut d).ok()?;
        d?
    };
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    dev.GetCopyableFootprints(desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
    let mut rb: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut rb,
    )
    .ok()?;
    let rb = rb?;
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
    let s = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(src), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    list.CopyTextureRegion(&dst, 0, 0, 0, &s, None);
    Some((rb, fp))
}

/// texel → 8-bit RGB for the formats titles present in (None: not decoded)
fn rgb(format: DXGI_FORMAT, t: u32) -> Option<[u8; 3]> {
    Some(match format {
        DXGI_FORMAT_R10G10B10A2_UNORM => [(t >> 2 & 0xFF) as u8, (t >> 12 & 0xFF) as u8, (t >> 22 & 0xFF) as u8],
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => [t as u8, (t >> 8) as u8, (t >> 16) as u8],
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => [(t >> 16) as u8, (t >> 8) as u8, t as u8],
        // HDR: tone-mapped for viewing
        DXGI_FORMAT_R11G11B10_FLOAT => {
            let tm = |v: f32| ((v / (1.0 + v)).sqrt() * 255.0) as u8;
            [tm(small_float(t & 0x7FF, 6)), tm(small_float(t >> 11 & 0x7FF, 6)), tm(small_float(t >> 22, 5))]
        }
        _ => return None,
    })
}

/// the largest render target the draws `XCL_TRACE_SHADERS` picked drew into (AddRef'd): the
/// one the `XCL_TRACE_FRAME` dumps show instead of the busiest
static WATCHED_RT: Mutex<Option<usize>> = Mutex::new(None);
/// targets that what the watched one holds went on to (drawn or copied from it; AddRef'd)
static FOLLOWED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub fn watched_rt() -> Option<usize> {
    *WATCHED_RT.lock().unwrap()
}

/// The watched target or one its contents went on to.
pub fn is_watched(r: usize) -> bool {
    watched_rt() == Some(r) || FOLLOWED.lock().unwrap().contains(&r)
}

/// `r` now holds (some of) what the watched target held: its readers are traced too.
pub unsafe fn follow(r: usize) {
    let mut g = FOLLOWED.lock().unwrap();
    if r == 0 || g.contains(&r) || g.len() >= 6 || watched_rt() == Some(r) {
        return;
    }
    add_ref(r);
    g.push(r);
}

pub unsafe fn watch_rt(rt: usize) {
    let area = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map_or(0, |x| {
        let d = x.GetDesc();
        if d.SampleDesc.Count > 1 { 0 } else { d.Width * d.Height as u64 }
    });
    let mut g = WATCHED_RT.lock().unwrap();
    if *g == Some(rt) || area(rt) <= g.map_or(0, area) {
        return;
    }
    add_ref(rt);
    if let Some(old) = g.replace(rt) {
        release(old);
    }
}

/// the single-sampled resource bound as a render target most often (`XCL_TRACE_FRAME`; a
/// multisampled one cannot be copied to a buffer – the present list would not close)
fn busiest_rt() -> Option<usize> {
    if let Some(r) = *WATCHED_RT.lock().unwrap() {
        return Some(r);
    }
    let single = |r: usize| unsafe { ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| x.GetDesc().SampleDesc.Count <= 1) };
    USES.lock().unwrap().as_ref()?.iter().filter(|(&r, u)| u[USE_RT] > 0 && single(r)).max_by_key(|(_, u)| u[USE_RT]).map(|(&r, _)| r)
}

/// an 11- / 10-bit unsigned float (5-bit exponent) -> f32
fn small_float(v: u32, mantissa_bits: u32) -> f32 {
    let (e, m) = (v >> mantissa_bits, v & ((1 << mantissa_bits) - 1));
    let m = m as f32 / (1u32 << mantissa_bits) as f32;
    match e {
        0 => m * 2f32.powi(-14),
        31 => 65504.0,
        _ => (1.0 + m) * 2f32.powi(e as i32 - 15),
    }
}

unsafe fn write_frame(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, format: DXGI_FORMAT, tag: &str) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let (w, h, pitch) = (fp.Footprint.Width as usize, fp.Footprint.Height as usize, fp.Footprint.RowPitch as usize);
    // every 4th texel of every 4th row; `XCL_FRAME_DUMP_FULL=1`: every texel (1-pixel detail)
    let step = if std::env::var_os("XCL_FRAME_DUMP_FULL").is_some() { 1 } else { 4 };
    let (ow, oh) = ((w / step).max(1), (h / step).max(1));
    let row = (ow * 3).div_ceil(4) * 4;
    let mut bmp = Vec::with_capacity(54 + row * oh);
    let size = (54 + row * oh) as u32;
    bmp.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40, ow as u32, oh as u32] {
        bmp.extend_from_slice(&v.to_le_bytes());
    }
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    for _ in 0..6 {
        bmp.extend_from_slice(&0u32.to_le_bytes());
    }
    let (mut lit, mut decoded) = (0usize, true);
    let mut raw: HashMap<u32, usize> = HashMap::new();
    for y in (0..oh).rev() {
        let start = bmp.len();
        for x in 0..ow {
            let t = ((p as *const u8).add(fp.Offset as usize + step * y * pitch + 4 * step * x) as *const u32).read_unaligned();
            *raw.entry(t).or_default() += 1;
            let c = rgb(format, t).unwrap_or_else(|| {
                decoded = false;
                [0; 3]
            });
            if c.iter().map(|&v| v as u32).sum::<u32>() > 30 {
                lit += 1;
            }
            bmp.extend_from_slice(&[c[2], c[1], c[0]]);
        }
        bmp.resize(start + row, 0);
    }
    rb.Unmap(0, None);
    let n = PRESENTS.load(Ordering::Relaxed);
    // `XCL_TRACE_FRAME` also dumps intermediate targets; only the presented image belongs here.
    if tag == "frame" {
        if let Ok(mut last) = LAST_FRAME_DUMP.lock() {
            *last = Some(FrameDumpSample {
                frame: n,
                width: w as u32,
                height: h as u32,
                format: format.0,
                sample_step: step as u32,
                sampled_pixels: (ow * oh) as u64,
                non_dark_samples: decoded.then_some(lit as u64),
            });
        }
    }
    let Some(boot) = crate::state::boot() else { return };
    let path = boot.logs_dir().join(format!("{tag}_{n}.bmp"));
    let _ = std::fs::write(&path, &bmp);
    crate::xlog_info!(
        "present: {tag} {n} ({w}x{h}, format {}) written to {} – {lit} of {} samples not black{}",
        format.0,
        path.display(),
        ow * oh,
        if decoded { "" } else { " (format not decoded)" }
    );
    let mut top: Vec<(u32, usize)> = raw.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    crate::xlog_info!("present: {tag} {n} raw texels (most common): {}", top.iter().take(6).map(|(v, c)| format!("{v:#010x}×{c}")).collect::<Vec<_>>().join(" "));
}

impl Drop for Presenter {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_uses_frame_and_time_deltas() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 1_000);
        assert_eq!(
            rate.sample(start + Duration::from_millis(1_500), 1_030),
            20.0
        );
        assert_eq!(
            rate.sample(start + Duration::from_millis(2_750), 1_080),
            40.0
        );
    }

    #[test]
    fn frame_rate_waits_for_a_full_sample() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 0);
        assert_eq!(rate.sample(start, 0), 0.0);
        assert_eq!(rate.sample(start + Duration::from_millis(500), 15), 0.0);
        assert_eq!(rate.sample(start + Duration::from_secs(1), 30), 30.0);
        assert_eq!(rate.sample(start + Duration::from_millis(1_500), 35), 30.0);
        assert_eq!(rate.sample(start + Duration::from_secs(2), 40), 10.0);
    }

    #[test]
    fn frame_rate_reports_stalls_and_recovery() {
        let start = Instant::now();
        let mut rate = FrameRate::new(start, 0);
        assert_eq!(rate.sample(start + Duration::from_secs(1), 60), 60.0);
        assert_eq!(rate.sample(start + Duration::from_secs(2), 60), 0.0);
        assert_eq!(rate.sample(start + Duration::from_secs(4), 70), 5.0);
    }

    // XCL_WINDOW_POS is process-global; one test covers all modes in sequence.
    #[test]
    fn window_position_modes() {
        unsafe {
            std::env::remove_var("XCL_WINDOW_POS");
            assert_eq!(window_position("auto", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("nonsense", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));
            assert_eq!(window_position("12,abc", 1280, 720), (CW_USEDEFAULT, CW_USEDEFAULT));

            let (cx, cy) = window_position("center", 1280, 720);
            if cx != CW_USEDEFAULT {
                // centered on the primary work area: inside it, and not off its left/top
                let mi = {
                    let mon = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
                    let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
                    assert!(GetMonitorInfoW(mon, &mut mi).as_bool());
                    mi
                };
                assert!(cx >= mi.rcWork.left && cy >= mi.rcWork.top);
                assert!(cx + 1280 <= mi.rcWork.right || mi.rcWork.right - mi.rcWork.left < 1280);
            }

            // a position far outside every monitor clamps into the nearest work area
            let (x, y) = window_position("100000,100000", 1280, 720);
            let mon = MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONEAREST);
            assert!(!mon.is_invalid());

            // the env override wins over the setting
            std::env::set_var("XCL_WINDOW_POS", "center");
            assert_eq!(window_position("auto", 1280, 720), window_position("center", 1280, 720));
            std::env::remove_var("XCL_WINDOW_POS");
        }
    }
}

#[cfg(test)]
mod frame_dump_tests {
    use super::*;
    use super::super::real::D3d12Stats;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    #[test]
    fn reports_without_frame_dumps_remain_readable() {
        let stats = D3d12Stats { presents: 100, ..Default::default() };
        let mut json = serde_json::to_value(&stats).unwrap();
        assert!(json["last_frame_dump"].is_null(), "presents alone do not establish content");
        json.as_object_mut().unwrap().remove("last_frame_dump");
        let old: D3d12Stats = serde_json::from_value(json).unwrap();
        assert_eq!(old.presents, 100);
        assert_eq!(old.last_frame_dump, None);
    }

    #[test]
    fn mapped_frame_dump_distinguishes_dark_bright_and_undecoded_content() {
        unsafe {
            // WARP needs no physical GPU. Fill a CPU-visible buffer with the same padded
            // footprint used by a readback, so this exercises write_frame's actual decoder.
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).unwrap();
            let dev = dev.unwrap();
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 512 + 256 * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer,
            ).unwrap();
            let buffer = buffer.unwrap();
            let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 512,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 8, Height: 8, Depth: 1, RowPitch: 256,
                },
            };
            let fill = |red: u8| {
                let mut p: *mut c_void = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let bytes = std::slice::from_raw_parts_mut(p as *mut u8, desc.Width as usize);
                bytes.fill(255); // padding must not be counted as bright pixels
                for y in 0..8 {
                    for x in 0..8 {
                        let start = 512 + y * 256 + x * 4;
                        bytes[start..start + 4].copy_from_slice(&[red, 0, 0, 255]);
                    }
                }
                buffer.Unmap(0, None);
            };
            fill(30);
            write_frame(&buffer, &fp, fp.Footprint.Format, "frame");
            let dark = last_frame_dump().unwrap();
            assert_eq!((dark.width, dark.height, dark.format), (8, 8, 28));
            assert_eq!(dark.sampled_pixels, (8 / dark.sample_step).pow(2) as u64);
            assert_eq!(dark.non_dark_samples, Some(0));

            fill(31);
            write_frame(&buffer, &fp, fp.Footprint.Format, "rt");
            assert_eq!(last_frame_dump(), Some(dark), "intermediate targets must not replace frame content");
            write_frame(&buffer, &fp, fp.Footprint.Format, "frame");
            let bright = last_frame_dump().unwrap();
            assert_eq!(bright.non_dark_samples, Some(bright.sampled_pixels));
            write_frame(&buffer, &fp, DXGI_FORMAT_R32_FLOAT, "frame");
            let json = serde_json::to_value(last_frame_dump().unwrap()).unwrap();
            assert!(json["non_dark_samples"].is_null(), "undecoded is different from dark");
        }
    }
}
