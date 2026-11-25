//! GameInput.dll: how a GDK title reads its controllers.
//!
//! The title gets the console's GameInput interfaces (the public GameInput.h of the GDK, v0:
//! IGameInput, IGameInputDevice, IGameInputReading as COM objects with the methods in that
//! header's order) with one device: a gamepad that is always connected. A reading is the
//! state at the moment it is asked for – the PC's first XInput controller (else a PlayStation
//! controller, `playstation`), and the keyboard
//! while a window of this process has the focus (Enter / Space A, Escape / Backspace B,
//! arrows the D-pad, Tab View, P Menu). Windows' own GameInput.dll is not used: its
//! interfaces are not the console's and it reads only for a foreground app of its own.
//! Device callbacks report the gamepad (at once for a blocking enumeration, from another
//! thread for an asynchronous one); reading callbacks get a reading when the state changes
//! (polled at 250 Hz). Methods not listed log their slot once and fail.
//!
//! `XCL_PRESS_A_AT=s1,s2…`: press A for 200 ms at those seconds after start (tests).

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use windows_sys::core::GUID;

use crate::util::{E_NOINTERFACE, E_NOTIMPL, E_POINTER, HRESULT, S_OK};

const GAMEINPUT_E_DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
const GAMEINPUT_E_READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;

// Verified GameInput v0 interface GUIDs (Windows SDK / GDK `GameInput.h`).
const IID_IUNKNOWN: GUID = GUID::from_u128(0x00000000_0000_0000_c000_000000000046);
const IID_IGAMEINPUT: GUID = GUID::from_u128(0x11be2a7e_4254_445a_9c09_ffc40f006918);
const IID_IGAMEINPUT_READING: GUID = GUID::from_u128(0x2156947a_e1fa_4de0_a30b_d812931dbd8d);
const IID_IGAMEINPUT_DEVICE: GUID = GUID::from_u128(0x31dd86fb_4c1b_408a_868f_439b3cd47125);

// GameInputKind
const KIND_CONTROLLER: u32 = 0xE;
const KIND_GAMEPAD: u32 = 0x4_0000;
const KIND_UI_NAVIGATION: u32 = 0x100_0000;
const KINDS: u32 = KIND_CONTROLLER | KIND_GAMEPAD | KIND_UI_NAVIGATION;
// GameInputDeviceStatus: connected, input enabled, output enabled, synchronized
const STATUS: u32 = 0x1 | 0x2 | 0x4 | 0x40;
// GameInputEnumerationKind
const ENUM_BLOCKING: u32 = 2;

/// GameInputGamepadState
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct GamepadState {
    pub buttons: u32,
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_x: f32,
    pub left_y: f32,
    pub right_x: f32,
    pub right_y: f32,
}

// GameInputGamepadButtons
pub const MENU: u32 = 0x1;
pub const VIEW: u32 = 0x2;
pub const A: u32 = 0x4;
pub const B: u32 = 0x8;
pub const X: u32 = 0x10;
pub const Y: u32 = 0x20;
pub const DPAD_UP: u32 = 0x40;
pub const DPAD_DOWN: u32 = 0x80;
pub const DPAD_LEFT: u32 = 0x100;
pub const DPAD_RIGHT: u32 = 0x200;
pub const LEFT_SHOULDER: u32 = 0x400;
pub const RIGHT_SHOULDER: u32 = 0x800;
pub const LEFT_STICK: u32 = 0x1000;
pub const RIGHT_STICK: u32 = 0x2000;

/// XINPUT_GAMEPAD button bits → GameInputGamepadButtons
pub fn buttons_from_xinput(w: u16) -> u32 {
    const MAP: [(u16, u32); 14] = [
        (0x0001, DPAD_UP),
        (0x0002, DPAD_DOWN),
        (0x0004, DPAD_LEFT),
        (0x0008, DPAD_RIGHT),
        (0x0010, MENU),
        (0x0020, VIEW),
        (0x0040, LEFT_STICK),
        (0x0080, RIGHT_STICK),
        (0x0100, LEFT_SHOULDER),
        (0x0200, RIGHT_SHOULDER),
        (0x1000, A),
        (0x2000, B),
        (0x4000, X),
        (0x8000, Y),
    ];
    MAP.iter().filter(|(x, _)| w & x != 0).fold(0, |b, (_, g)| b | g)
}

/// GameInputUiNavigationButtons from gamepad buttons
pub fn ui_navigation(b: u32) -> u32 {
    const MAP: [(u32, u32); 12] = [
        (MENU, 0x1),
        (VIEW, 0x2),
        (A, 0x4),
        (B, 0x8),
        (DPAD_UP, 0x10),
        (DPAD_DOWN, 0x20),
        (DPAD_LEFT, 0x40),
        (DPAD_RIGHT, 0x80),
        (X, 0x100),
        (Y, 0x200),
        (LEFT_SHOULDER, 0x4000),
        (RIGHT_SHOULDER, 0x8000),
    ];
    MAP.iter().filter(|(g, _)| b & g != 0).fold(0, |u, (_, n)| u | n)
}

// ---------------------------------------------------------------------------------------
// the state: XInput controller 0..3 (the first connected), the keyboard, scripted presses
// ---------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Default)]
struct XInputState {
    packet: u32,
    buttons: u16,
    left_trigger: u8,
    right_trigger: u8,
    lx: i16,
    ly: i16,
    rx: i16,
    ry: i16,
}

type GetState = unsafe extern "system" fn(u32, *mut XInputState) -> u32;
type SetState = unsafe extern "system" fn(u32, *const [u16; 2]) -> u32;

fn xinput() -> Option<(GetState, SetState)> {
    static F: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    let f = *F.get_or_init(|| unsafe {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
        let h = LoadLibraryA(b"xinput1_4.dll\0".as_ptr());
        if h.is_null() {
            crate::xlog_warn!("GameInput: xinput1_4.dll not loaded – controllers of the PC are not read");
            return None;
        }
        let g = GetProcAddress(h, b"XInputGetState\0".as_ptr())? as usize;
        let s = GetProcAddress(h, b"XInputSetState\0".as_ptr())? as usize;
        Some((g, s))
    });
    f.map(|(g, s)| unsafe { (std::mem::transmute::<usize, GetState>(g), std::mem::transmute::<usize, SetState>(s)) })
}

/// the XInput user index that answered last
static PAD: AtomicU32 = AtomicU32::new(0);
/// a controller answered the last poll (connect / disconnect are logged)
static PAD_CONNECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// microseconds since start of the last search over the four XInput slots while none answered:
/// XInputGetState on an empty slot is slow (it asks the driver each time), so with no
/// controller connected the slots are searched at most every `SEARCH_US`
static LAST_SEARCH: AtomicU64 = AtomicU64::new(0);
const SEARCH_US: u64 = 1_000_000;

/// The PC's controller (the XInput slot that answers first: an Xbox pad, or any pad Windows
/// exposes through XInput) as a GameInput gamepad state; None without one.
fn xinput_pad() -> Option<GamepadState> {
    let (get, _) = xinput()?;
    let connected = PAD_CONNECTED.load(Ordering::Relaxed);
    let first = PAD.load(Ordering::Relaxed);
    // with a controller: its slot answers first (one call); without: all four, now and then
    if !connected {
        let now = now_us();
        let last = LAST_SEARCH.load(Ordering::Relaxed);
        if last != 0 && now.saturating_sub(last) < SEARCH_US {
            return None;
        }
        LAST_SEARCH.store(now, Ordering::Relaxed);
    }
    for k in (0..4).map(|i| (first + i) % 4) {
        let mut x = XInputState::default();
        let r = unsafe { get(k, &mut x) };
        if r != 0 && r != 1167 {
            // not ERROR_DEVICE_NOT_CONNECTED: worth a line (once)
            static ODD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ODD.swap(true, Ordering::Relaxed) {
                crate::xlog_warn!("GameInput: XInputGetState({k}) → {r}");
            }
        }
        if r == 0 {
            if !connected || k != first {
                crate::xlog_info!("GameInput: the PC's controller (XInput slot {k}) is the title's gamepad");
            }
            PAD.store(k, Ordering::Relaxed);
            PAD_CONNECTED.store(true, Ordering::Relaxed);
            let axis = |v: i16| (v as f32 / 32767.0).max(-1.0);
            return Some(GamepadState {
                buttons: buttons_from_xinput(x.buttons),
                left_trigger: x.left_trigger as f32 / 255.0,
                right_trigger: x.right_trigger as f32 / 255.0,
                left_x: axis(x.lx),
                left_y: axis(x.ly),
                right_x: axis(x.rx),
                right_y: axis(x.ry),
            });
        }
    }
    if connected {
        crate::xlog_info!(
            "GameInput: the PC's controller was disconnected{}",
            if super::kbm::enabled() { " (keyboard and mouse still play)" } else { "" }
        );
        PAD_CONNECTED.store(false, Ordering::Relaxed);
        LAST_SEARCH.store(now_us(), Ordering::Relaxed);
    }
    None
}

fn start() -> Instant {
    static T: OnceLock<Instant> = OnceLock::new();
    *T.get_or_init(Instant::now)
}

/// A button by its name in `XCL_PRESS_AT`.
fn button_named(name: &str) -> Option<u32> {
    Some(match name.to_ascii_uppercase().as_str() {
        "A" => A,
        "B" => B,
        "X" => X,
        "Y" => Y,
        "UP" => DPAD_UP,
        "DOWN" => DPAD_DOWN,
        "LEFT" => DPAD_LEFT,
        "RIGHT" => DPAD_RIGHT,
        "MENU" => MENU,
        "VIEW" => VIEW,
        "LB" => LEFT_SHOULDER,
        "RB" => RIGHT_SHOULDER,
        _ => return None,
    })
}

/// `XCL_PRESS_AT=s:BUTTON,…` ("14:A,18.5:DOWN,20:A"; buttons A B X Y UP DOWN LEFT RIGHT MENU
/// VIEW LB RB) and `XCL_PRESS_A_AT=s,…`: buttons held for 200 ms at those seconds after the
/// host started (the log's times) – a title's menus walked without a person at the controls.
fn parse_presses(script: &str, a_only: &str) -> Vec<(f64, u32)> {
    let mut v: Vec<(f64, u32)> = a_only.split(',').filter_map(|s| s.trim().parse().ok()).map(|s| (s, A)).collect();
    for item in script.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match item.split_once(':').and_then(|(s, b)| Some((s.trim().parse::<f64>().ok()?, button_named(b.trim())?))) {
            Some(p) => v.push(p),
            None => crate::xlog_warn!("GameInput: XCL_PRESS_AT entry {item:?} not understood (seconds:BUTTON)"),
        }
    }
    v
}

fn scripted() -> u32 {
    static AT: OnceLock<Vec<(f64, u32)>> = OnceLock::new();
    let at = AT.get_or_init(|| {
        let var = |n: &str| std::env::var(n).unwrap_or_default();
        parse_presses(&var("XCL_PRESS_AT"), &var("XCL_PRESS_A_AT"))
    });
    // seconds since the host started, as the log's times (not since the title first polled)
    let t = crate::log::uptime();
    at.iter().filter(|&&(s, _)| t >= s && t < s + 0.2).fold(0, |b, &(_, g)| b | g)
}

#[link(name = "user32")]
extern "system" {
    fn GetAsyncKeyState(key: i32) -> i16;
    fn GetForegroundWindow() -> *mut c_void;
    fn GetWindowThreadProcessId(window: *mut c_void, pid: *mut u32) -> u32;
}

unsafe fn keyboard() -> u32 {
    let w = GetForegroundWindow();
    if w.is_null() {
        return 0;
    }
    let mut pid = 0u32;
    GetWindowThreadProcessId(w, &mut pid);
    if pid != std::process::id() {
        return 0;
    }
    const KEYS: [(i32, u32); 10] = [(0x0D, A), (0x20, A), (0x1B, B), (0x08, B), (0x26, DPAD_UP), (0x28, DPAD_DOWN), (0x25, DPAD_LEFT), (0x27, DPAD_RIGHT), (0x09, VIEW), (0x50, MENU)];
    KEYS.iter().filter(|(k, _)| GetAsyncKeyState(*k) as u16 & 0x8000 != 0).fold(0, |b, (_, g)| b | g)
}

/// Merge host keyboard/mouse input (only when `keyboard_mouse` is enabled) and scripted
/// test presses (`XCL_PRESS_AT` / `XCL_PRESS_A_AT`) into `s`.
fn merge_host_inputs(
    s: &mut GamepadState,
    keyboard_mouse: bool,
    read_keyboard: impl FnOnce() -> u32,
    apply_kbm: impl FnOnce(&mut GamepadState),
    read_scripted: impl FnOnce() -> u32,
) {
    if keyboard_mouse {
        s.buttons |= read_keyboard();
        apply_kbm(s);
    }
    s.buttons |= read_scripted();
}

/// the gamepad now
pub fn poll() -> GamepadState {
    let mut s = xinput_pad().or_else(super::playstation::pad).unwrap_or_default();
    merge_host_inputs(
        &mut s,
        super::kbm::enabled(),
        || {
            // SAFETY: `keyboard()` queries foreground window and async key states via Win32.
            unsafe { keyboard() }
        },
        super::kbm::apply,
        scripted,
    );
    {
        // the first presses of each button: worth a line (is input arriving at all?)
        static SEEN: AtomicU32 = AtomicU32::new(0);
        let new = s.buttons & !SEEN.fetch_or(s.buttons, Ordering::Relaxed);
        if new != 0 {
            crate::xlog_info!("GameInput: gamepad buttons {new:#x} pressed for the first time");
        }
    }
    s
}

fn now_us() -> u64 {
    start().elapsed().as_micros() as u64 + 1
}

// ---------------------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------------------

#[repr(C)]
struct Object {
    vtable: *const usize,
}

// static objects: their vtables are leaked tables
unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[repr(C)]
struct Reading {
    vtable: *const usize,
    refs: AtomicU32,
    state: GamepadState,
    sequence: u64,
    timestamp: u64,
}

/// GameInputDeviceInfo, as far as a title looks to tell a gamepad: an Xbox controller that
/// reads as a gamepad (and controller, UI navigation) with the four rumble motors
#[repr(C)]
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage_page: u16,
    usage_id: u16,
    hardware_version: [u16; 4],
    firmware_version: [u16; 4],
    device_id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    rest: [u8; 896],
}

const DEVICE_ID: [u8; 32] = *b"XCL gamepad 0                   ";

/// The gamepad's APP_LOCAL_DEVICE_ID (XUser's device association names it).
pub fn gamepad_device_id() -> [u8; 32] {
    DEVICE_ID
}

fn device_info() -> &'static DeviceInfo {
    static I: OnceLock<Box<DeviceInfo>> = OnceLock::new();
    I.get_or_init(|| {
        Box::new(DeviceInfo {
            size: std::mem::size_of::<DeviceInfo>() as u32,
            vendor: 0x045E,
            product: 0x0B12,
            revision: 1,
            interface: 0,
            collection: 0,
            usage_page: 1,
            usage_id: 5,
            hardware_version: [1, 0, 0, 0],
            firmware_version: [1, 0, 0, 0],
            device_id: DEVICE_ID,
            root_id: DEVICE_ID,
            // GameInputFamilyXboxOne
            family: 1,
            // GameInputDeviceCapabilities: synchronization
            capabilities: 8,
            supported_input: KINDS,
            // low / high frequency, left / right trigger
            rumble_motors: 0xF,
            rest: [0; 896],
        })
    })
}

/// a vtable of `n` slots: logging stubs that fail, `set` filled in
fn vtable(name: &str, n: usize, set: &[(usize, usize)]) -> *const usize {
    let mut v: Vec<usize> = (0..n).map(|i| crate::stubs::make_stub_info("GameInput.dll", &format!("{name}::#{i}"), E_NOTIMPL as u32 as u64).0).collect();
    for &(i, f) in set {
        v[i] = f;
    }
    Box::leak(v.into_boxed_slice()).as_ptr()
}

fn device() -> *mut Object {
    static D: OnceLock<usize> = OnceLock::new();
    *D.get_or_init(|| {
        let v = vtable(
            "IGameInputDevice",
            24,
            &[
                (0, device_qi as usize),
                (1, add_ref as usize),
                (2, release as usize),
                (3, get_device_info as usize),
                (4, get_device_status as usize),
                (5, get_battery_state as usize),
                (7, motor_powered_on as usize),
                (8, nothing_u32_f32 as usize),
                (10, set_rumble_state as usize),
                (11, nothing_u32 as usize),
                (12, nothing as usize),
                (13, nothing as usize),
            ],
        );
        Box::leak(Box::new(Object { vtable: v })) as *mut Object as usize
    }) as *mut Object
}

fn input() -> *mut Object {
    static G: OnceLock<usize> = OnceLock::new();
    *G.get_or_init(|| {
        let v = vtable(
            "IGameInput",
            24,
            &[
                (0, input_qi as usize),
                (1, add_ref as usize),
                (2, release as usize),
                (3, get_current_timestamp as usize),
                (4, get_current_reading as usize),
                (5, other_reading as usize),
                (6, other_reading as usize),
                (7, get_temporal_reading as usize),
                (8, register_reading_callback as usize),
                (9, register_device_callback as usize),
                (10, register_other_callback5 as usize),
                (11, register_other_callback4 as usize),
                (12, stop_callback as usize),
                (13, unregister_callback as usize),
                (16, find_device_from_id as usize),
                (20, enable_oem as usize),
                (21, set_focus_policy as usize),
            ],
        );
        Box::leak(Box::new(Object { vtable: v })) as *mut Object as usize
    }) as *mut Object
}

fn reading_vtable() -> *const usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        let mut set = vec![
            (0, reading_qi as usize),
            (1, reading_add_ref as usize),
            (2, reading_release as usize),
            (3, reading_kind as usize),
            (4, reading_sequence as usize),
            (5, reading_timestamp as usize),
            (6, reading_device as usize),
            (7, no_state::<7> as usize),
            (22, reading_gamepad as usize),
            (24, reading_ui_navigation as usize),
        ];
        // counts and array states: none (each logged on its first call – which views of
        // the device a title reads)
        let counts: [usize; 10] = [
            zero_count::<8> as usize,
            zero_count::<9> as usize,
            zero_count::<10> as usize,
            zero_count::<11> as usize,
            zero_count::<12> as usize,
            zero_count::<13> as usize,
            zero_count::<14> as usize,
            zero_count::<15> as usize,
            zero_count::<17> as usize,
            zero_count::<18> as usize,
        ];
        for (i, f) in [8, 9, 10, 11, 12, 13, 14, 15, 17, 18].into_iter().zip(counts) {
            set.push((i, f));
        }
        // mouse, motion, arcade / flight stick, racing wheel: not this device
        let states: [usize; 5] = [no_state::<16> as usize, no_state::<19> as usize, no_state::<20> as usize, no_state::<21> as usize, no_state::<23> as usize];
        for (i, f) in [16, 19, 20, 21, 23].into_iter().zip(states) {
            set.push((i, f));
        }
        vtable("IGameInputReading", 32, &set) as usize
    }) as *const usize
}

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn new_reading(state: GamepadState) -> *mut Reading {
    Box::into_raw(Box::new(Reading { vtable: reading_vtable(), refs: AtomicU32::new(1), state, sequence: SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1, timestamp: now_us() }))
}

// ---- shared methods ----

fn guid_eq(a: &GUID, b: &GUID) -> bool {
    (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4)
}

/// Validate `iid` against `IID_IUNKNOWN` and `own`, enforcing COM out-pointer semantics.
///
/// # Safety
/// `out` must be null or valid for writes; `iid` must be null or valid for reads as a `GUID`.
unsafe fn qi_for(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void, own: &GUID, on_ok: impl FnOnce()) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    if !iid.is_null() && (guid_eq(&*iid, own) || guid_eq(&*iid, &IID_IUNKNOWN)) {
        on_ok();
        *out = this;
        return S_OK;
    }
    if !iid.is_null() {
        // refused interfaces are worth a line each (a title built for a newer GameInput asks
        // for its interfaces this way – before #100 it got the v0 object for any IID)
        static SEEN: Mutex<Vec<u128>> = Mutex::new(Vec::new());
        let i = &*iid;
        let g = (i.data1 as u128) << 96 | (i.data2 as u128) << 80 | (i.data3 as u128) << 64 | u64::from_be_bytes(i.data4) as u128;
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&g) && seen.len() < 16 {
            seen.push(g);
            crate::xlog_warn!("GameInput: QueryInterface({g:032x}) refused (E_NOINTERFACE)");
        }
    }
    *out = std::ptr::null_mut();
    E_NOINTERFACE
}

unsafe extern "system" fn input_qi(this: *mut Object, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    qi_for(this as *mut c_void, iid, out, &IID_IGAMEINPUT, || {})
}
unsafe extern "system" fn device_qi(this: *mut Object, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    qi_for(this as *mut c_void, iid, out, &IID_IGAMEINPUT_DEVICE, || {})
}
unsafe extern "system" fn add_ref(_this: *mut c_void) -> u32 {
    1
}
unsafe extern "system" fn release(_this: *mut c_void) -> u32 {
    1
}
unsafe extern "system" fn nothing(_this: *mut c_void) {}
unsafe extern "system" fn nothing_u32(_this: *mut c_void, _a: u32) {}
unsafe extern "system" fn nothing_u32_f32(_this: *mut c_void, _a: u32, _b: f32) {}

// ---- IGameInputDevice ----

unsafe extern "system" fn get_device_info(_this: *mut Object) -> *const DeviceInfo {
    device_info()
}
unsafe extern "system" fn get_device_status(_this: *mut Object) -> u32 {
    STATUS
}
/// GameInputBatteryState: unknown (wired)
unsafe extern "system" fn get_battery_state(_this: *mut Object, state: *mut [u32; 5]) {
    if !state.is_null() {
        *state = [0, 0, 0, 0, 0];
    }
}
unsafe extern "system" fn motor_powered_on(_this: *mut Object, _motor: u32) -> bool {
    false
}
/// GameInputRumbleParams {low, high, left trigger, right trigger}: the XInput motors
unsafe extern "system" fn set_rumble_state(_this: *mut Object, p: *const [f32; 4]) {
    let Some((_, set)) = xinput() else { return };
    let (lo, hi) = if p.is_null() { (0.0, 0.0) } else { ((*p)[0], (*p)[1]) };
    let speed = |v: f32| (v.clamp(0.0, 1.0) * 65535.0) as u16;
    set(PAD.load(Ordering::Relaxed), &[speed(lo), speed(hi)]);
}

// ---- IGameInput ----

unsafe extern "system" fn get_current_timestamp(_this: *mut Object) -> u64 {
    now_us()
}

fn ours(device: *mut c_void) -> bool {
    device.is_null() || device == self::device() as *mut c_void
}

/// A GameInput query of a kind not seen before: worth a line (which input a title reads how).
fn note_query(what: &str, kind: u32) {
    static SEEN: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
    let what: &'static str = match what {
        "GetCurrentReading" => "GetCurrentReading",
        "GetNext/PreviousReading" => "GetNext/PreviousReading",
        _ => "RegisterReadingCallback",
    };
    let mut g = SEEN.lock().unwrap();
    if g.len() < 32 && !g.contains(&(what, kind)) {
        g.push((what, kind));
        crate::xlog_info!("GameInput: {what} for kinds {kind:#x} first asked");
    }
}

unsafe extern "system" fn get_current_reading(_this: *mut Object, kind: u32, dev: *mut c_void, out: *mut *mut Reading) -> HRESULT {
    note_query("GetCurrentReading", kind);
    if out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    if kind & KINDS == 0 || !ours(dev) {
        return GAMEINPUT_E_READING_NOT_FOUND;
    }
    *out = new_reading(poll());
    S_OK
}

/// the readings before and after one are not kept
unsafe extern "system" fn other_reading(_this: *mut Object, _reference: *mut c_void, kind: u32, _dev: *mut c_void, out: *mut *mut Reading) -> HRESULT {
    note_query("GetNext/PreviousReading", kind);
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
    GAMEINPUT_E_READING_NOT_FOUND
}

unsafe extern "system" fn get_temporal_reading(this: *mut Object, _timestamp: u64, dev: *mut c_void, out: *mut *mut Reading) -> HRESULT {
    get_current_reading(this, KIND_GAMEPAD, dev, out)
}

static TOKENS: AtomicU64 = AtomicU64::new(0);

type ReadingCallback = unsafe extern "system" fn(u64, *mut c_void, *mut Reading, bool);
type DeviceCallback = unsafe extern "system" fn(u64, *mut c_void, *mut Object, u64, u32, u32);

enum CallbackKind {
    Reading { ctx: usize, cb: ReadingCallback },
    Device,
}

#[derive(Default)]
struct CallbackState {
    stopped: bool,
    in_flight: usize,
    running_threads: Vec<ThreadId>,
}

struct CallbackEntry {
    token: u64,
    kind: CallbackKind,
    state: Mutex<CallbackState>,
    done: Condvar,
}

struct CallbackGuard<'a> {
    entry: &'a CallbackEntry,
    tid: ThreadId,
}

impl Drop for CallbackGuard<'_> {
    fn drop(&mut self) {
        let mut st = self.entry.state.lock().unwrap_or_else(|e| e.into_inner());
        st.in_flight = st.in_flight.saturating_sub(1);
        if let Some(pos) = st.running_threads.iter().position(|&t| t == self.tid) {
            st.running_threads.swap_remove(pos);
        }
        self.entry.done.notify_all();
    }
}

impl CallbackEntry {
    fn new(token: u64, kind: CallbackKind) -> Self {
        Self {
            token,
            kind,
            state: Mutex::new(CallbackState::default()),
            done: Condvar::new(),
        }
    }

    fn try_enter(&self) -> Option<CallbackGuard<'_>> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.stopped {
            return None;
        }
        let tid = std::thread::current().id();
        st.in_flight += 1;
        st.running_threads.push(tid);
        Some(CallbackGuard { entry: self, tid })
    }

    /// Prevent future invocations; returns `true` when no invocation is currently in flight.
    fn stop_and_is_idle(&self) -> bool {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.stopped = true;
        st.in_flight == 0
    }

    /// Stop future invocations and wait up to `timeout_us` for in-flight invocations on other
    /// threads to finish. Re-entrant calls from inside the callback itself return `true`
    /// immediately rather than deadlocking.
    fn unregister(&self, timeout_us: u64) -> bool {
        let me = std::thread::current().id();
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.stopped = true;
        let other_in_flight = |s: &CallbackState| s.running_threads.iter().any(|&t| t != me);
        if !other_in_flight(&st) {
            return true;
        }
        if timeout_us == 0 {
            return false;
        }
        let deadline = Instant::now().checked_add(Duration::from_micros(timeout_us));
        while other_in_flight(&st) {
            match deadline {
                Some(end) => {
                    let Some(rem) = end.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    let (guard, wait_res) = self.done.wait_timeout(st, rem).unwrap_or_else(|e| e.into_inner());
                    st = guard;
                    if other_in_flight(&st) && wait_res.timed_out() {
                        return false;
                    }
                }
                None => {
                    // `timeout_us` overflows `Instant` (e.g. `u64::MAX`): wait indefinitely.
                    st = self.done.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
        true
    }
}

/// Registered callbacks (reading callbacks and pending/in-flight async device callbacks).
static CALLBACKS: Mutex<Vec<Arc<CallbackEntry>>> = Mutex::new(Vec::new());

/// RegisterReadingCallback(device, kind, analog threshold, context, callback, &token)
unsafe extern "system" fn register_reading_callback(_this: *mut Object, dev: *mut c_void, kind: u32, _threshold: f32, context: *mut c_void, cb: Option<ReadingCallback>, token: *mut u64) -> HRESULT {
    note_query("RegisterReadingCallback", kind);
    let t = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    if !token.is_null() {
        *token = t;
    }
    if let Some(cb) = cb.filter(|_| kind & KINDS != 0 && ours(dev)) {
        let entry = Arc::new(CallbackEntry::new(t, CallbackKind::Reading { ctx: context as usize, cb }));
        CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).push(entry);
        start_poller();
    }
    S_OK
}

fn dispatch_reading_callbacks(s: GamepadState) {
    let cbs: Vec<Arc<CallbackEntry>> = CALLBACKS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|c| matches!(c.kind, CallbackKind::Reading { .. }))
        .cloned()
        .collect();
    for entry in cbs {
        let CallbackKind::Reading { ctx, cb } = entry.kind else { continue };
        let Some(_guard) = entry.try_enter() else { continue };
        let r = new_reading(s);
        // SAFETY: `cb` and `ctx` were registered by the caller; `_guard` keeps
        // `UnregisterCallback` blocked until `cb` returns so `ctx` stays valid.
        unsafe {
            cb(entry.token, ctx as *mut c_void, r, false);
            reading_release(r);
        }
    }
}

/// A thread that calls the reading callbacks when the gamepad's state changes.
fn start_poller() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("xcl-gameinput".into()).spawn(|| {
            crate::eh::arm_current_thread();
            let mut last = GamepadState::default();
            loop {
                std::thread::sleep(Duration::from_millis(4));
                let s = poll();
                if s == last {
                    continue;
                }
                last = s;
                dispatch_reading_callbacks(s);
            }
        });
    });
}

/// RegisterDeviceCallback(device, kind, status filter, enumeration kind, context, callback,
/// &token): the gamepad, connected – at once for a blocking enumeration, from another
/// thread for an asynchronous one, not at all without enumeration (it never comes or goes)
unsafe extern "system" fn register_device_callback(_this: *mut Object, dev: *mut c_void, kind: u32, _filter: u32, enumeration: u32, context: *mut c_void, cb: Option<DeviceCallback>, token: *mut u64) -> HRESULT {
    let t = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    if !token.is_null() {
        *token = t;
    }
    crate::xlog_info!("GameInput: device callback for kinds {kind:#x} (enumeration {enumeration})");
    let Some(cb) = cb.filter(|_| kind & KINDS != 0 && ours(dev) && enumeration != 0) else { return S_OK };
    if enumeration == ENUM_BLOCKING {
        cb(t, context, device(), now_us(), STATUS, 0);
    } else {
        let entry = Arc::new(CallbackEntry::new(t, CallbackKind::Device));
        CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::clone(&entry));
        let ctx = context as usize;
        let spawned = std::thread::Builder::new().name("xcl-gameinput-enum".into()).spawn(move || {
            crate::eh::arm_current_thread();
            if let Some(_guard) = entry.try_enter() {
                // SAFETY: `cb` and `ctx` were registered by the caller; `_guard` keeps
                // `UnregisterCallback` blocked until `cb` returns so `ctx` stays valid.
                unsafe { cb(t, ctx as *mut c_void, device(), now_us(), STATUS, 0) };
            }
            CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| c.token != t);
        });
        if spawned.is_err() {
            CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| c.token != t);
        }
    }
    S_OK
}

/// RegisterSystemButtonCallback(device, buttons, context, callback, &token)
unsafe extern "system" fn register_other_callback5(_this: *mut Object, _a: *mut c_void, _b: u32, _c: *mut c_void, _d: *mut c_void, token: *mut u64) -> HRESULT {
    if !token.is_null() {
        *token = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    }
    S_OK
}

/// RegisterKeyboardLayoutCallback(device, context, callback, &token)
unsafe extern "system" fn register_other_callback4(_this: *mut Object, _a: *mut c_void, _c: *mut c_void, _d: *mut c_void, token: *mut u64) -> HRESULT {
    if !token.is_null() {
        *token = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    }
    S_OK
}

unsafe extern "system" fn stop_callback(_this: *mut Object, token: u64) {
    let entry = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|c| c.token == token).cloned();
    if let Some(entry) = entry {
        if entry.stop_and_is_idle() {
            CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| c.token != token);
        }
    }
}

unsafe extern "system" fn unregister_callback(_this: *mut Object, token: u64, timeout_us: u64) -> bool {
    let entry = CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|c| c.token == token).cloned();
    let Some(entry) = entry else {
        return true;
    };
    if entry.unregister(timeout_us) {
        CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| c.token != token);
        true
    } else {
        false
    }
}

/// FindDeviceFromId(&APP_LOCAL_DEVICE_ID, &device)
unsafe extern "system" fn find_device_from_id(_this: *mut Object, id: *const [u8; 32], out: *mut *mut Object) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    if !id.is_null() && *id == DEVICE_ID {
        *out = device();
        return S_OK;
    }
    *out = std::ptr::null_mut();
    GAMEINPUT_E_DEVICE_NOT_FOUND
}

unsafe extern "system" fn enable_oem(_this: *mut Object, _v: u16, _p: u16, _i: u8, _c: u8) -> HRESULT {
    S_OK
}

unsafe extern "system" fn set_focus_policy(_this: *mut Object, _policy: u32) {}

// ---- IGameInputReading ----

unsafe extern "system" fn reading_qi(this: *mut Reading, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    qi_for(this as *mut c_void, iid, out, &IID_IGAMEINPUT_READING, || {
        reading_add_ref(this);
    })
}
unsafe extern "system" fn reading_add_ref(this: *mut Reading) -> u32 {
    (*this).refs.fetch_add(1, Ordering::AcqRel) + 1
}
unsafe extern "system" fn reading_release(this: *mut Reading) -> u32 {
    let left = (*this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        drop(Box::from_raw(this));
    }
    left
}
unsafe extern "system" fn reading_kind(_this: *mut Reading) -> u32 {
    KINDS
}
unsafe extern "system" fn reading_sequence(this: *mut Reading, _kind: u32) -> u64 {
    (*this).sequence
}
unsafe extern "system" fn reading_timestamp(this: *mut Reading) -> u64 {
    (*this).timestamp
}
unsafe extern "system" fn reading_device(_this: *mut Reading, out: *mut *mut Object) {
    if !out.is_null() {
        *out = device();
    }
}
/// IGameInputReading method names by slot (the logged ones)
fn reading_method(slot: usize) -> &'static str {
    match slot {
        7 => "GetRawReport",
        8 => "GetControllerAxisCount",
        9 => "GetControllerAxisState",
        10 => "GetControllerButtonCount",
        11 => "GetControllerButtonState",
        12 => "GetControllerSwitchCount",
        13 => "GetControllerSwitchState",
        14 => "GetKeyCount",
        15 => "GetKeyState",
        16 => "GetMouseState",
        17 => "GetTouchCount",
        18 => "GetTouchState",
        19 => "GetMotionState",
        20 => "GetArcadeStickState",
        21 => "GetFlightStickState",
        23 => "GetRacingWheelState",
        _ => "?",
    }
}

fn note_reading_call(slot: usize) {
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if SEEN.fetch_or(1 << slot, Ordering::Relaxed) & (1 << slot) == 0 {
        crate::xlog_info!("GameInput: IGameInputReading::{} first called (answered: none)", reading_method(slot));
    }
}

unsafe extern "system" fn zero_count<const SLOT: usize>(_this: *mut Reading, _a: u32, _b: *mut c_void) -> u32 {
    note_reading_call(SLOT);
    0
}
unsafe extern "system" fn no_state<const SLOT: usize>(_this: *mut Reading, _out: *mut c_void) -> bool {
    note_reading_call(SLOT);
    false
}
unsafe extern "system" fn reading_gamepad(this: *mut Reading, out: *mut GamepadState) -> bool {
    if out.is_null() {
        return false;
    }
    *out = (*this).state;
    true
}
/// GameInputUiNavigationState {buttons}
unsafe extern "system" fn reading_ui_navigation(this: *mut Reading, out: *mut u32) -> bool {
    if out.is_null() {
        return false;
    }
    *out = ui_navigation((*this).state.buttons);
    true
}

/// GameInputCreate(&gameInput)
pub unsafe extern "system" fn GameInputCreate(out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    start();
    *out = input() as *mut c_void;
    crate::xlog_info!("GameInput: GameInputCreate – one gamepad (XInput controller / keyboard)");
    S_OK
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![GameInputCreate]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xinput_buttons_map_to_gameinput() {
        // A + B + D-pad up + start
        assert_eq!(buttons_from_xinput(0x1000 | 0x2000 | 0x0001 | 0x0010), A | B | DPAD_UP | MENU);
        assert_eq!(ui_navigation(A | B | DPAD_LEFT), 0x4 | 0x8 | 0x40);
        assert_eq!(std::mem::offset_of!(DeviceInfo, supported_input), 104);
    }

    #[test]
    fn readings_come_and_go() {
        unsafe {
            let g = input();
            let mut r = std::ptr::null_mut();
            assert_eq!(get_current_reading(g, KIND_GAMEPAD, std::ptr::null_mut(), &mut r), S_OK);
            let mut s = GamepadState { buttons: 99, ..Default::default() };
            assert!(reading_gamepad(r, &mut s));
            assert_ne!(s.buttons, 99);
            let mut d = std::ptr::null_mut();
            reading_device(r, &mut d);
            assert_eq!(d, device());
            assert_eq!(reading_release(r), 0);
            assert_eq!(get_current_reading(g, 0x10, std::ptr::null_mut(), &mut r), GAMEINPUT_E_READING_NOT_FOUND, "keyboard kind");
        }
    }

    #[test]
    fn press_script_parses_buttons_and_times() {
        let v = parse_presses("14:A, 18.5:down,20:Rb,junk,3:NOPE", "1,2.5");
        assert_eq!(v, vec![(1.0, A), (2.5, A), (14.0, A), (18.5, DPAD_DOWN), (20.0, RIGHT_SHOULDER)]);
        assert!(parse_presses("", "").is_empty());
    }

    #[test]
    fn keyboard_mouse_config_gates_keyboard_and_kbm_but_keeps_scripted() {
        let mut s = GamepadState { buttons: B, left_x: 0.25, ..Default::default() };
        let mut keyboard_called = false;
        let mut kbm_called = false;
        let mut scripted_called = false;

        merge_host_inputs(
            &mut s,
            false,
            || {
                keyboard_called = true;
                MENU
            },
            |st| {
                kbm_called = true;
                st.left_x = 1.0;
                st.buttons |= X;
            },
            || {
                scripted_called = true;
                A
            },
        );
        assert!(!keyboard_called, "keyboard() must not run when keyboard_mouse = false");
        assert!(!kbm_called, "kbm::apply() must not run when keyboard_mouse = false");
        assert!(scripted_called, "scripted() must still run when keyboard_mouse = false");
        assert_eq!(s.buttons, B | A);
        assert_eq!(s.left_x, 0.25);

        let mut s2 = GamepadState::default();
        merge_host_inputs(
            &mut s2,
            true,
            || MENU,
            |st| {
                st.left_x = -1.0;
                st.buttons |= Y;
            },
            || A,
        );
        assert_eq!(s2.buttons, MENU | Y | A);
        assert_eq!(s2.left_x, -1.0);
    }

    #[test]
    fn query_interface_validates_iids_and_refcounts() {
        type QiFn = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        const UNKNOWN_IID: GUID = GUID::from_u128(0xdeadbeef_1234_5678_9abc_def012345678);

        unsafe fn call_qi(obj: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
            let vtbl = *(obj as *const *const usize);
            let f: QiFn = std::mem::transmute(*vtbl);
            f(obj, iid, out)
        }

        unsafe {
            let sentinel = 0x1234usize as *mut c_void;

            // IGameInput
            let g = input() as *mut c_void;
            let mut out = sentinel;
            assert_eq!(call_qi(g, &IID_IUNKNOWN, &mut out), S_OK);
            assert_eq!(out, g);
            out = sentinel;
            assert_eq!(call_qi(g, &IID_IGAMEINPUT, &mut out), S_OK);
            assert_eq!(out, g);
            for bad in [&IID_IGAMEINPUT_DEVICE, &IID_IGAMEINPUT_READING, &UNKNOWN_IID] {
                out = sentinel;
                assert_eq!(call_qi(g, bad, &mut out), E_NOINTERFACE);
                assert!(out.is_null(), "failed QueryInterface must zero *out");
            }
            out = sentinel;
            assert_eq!(call_qi(g, std::ptr::null(), &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!(call_qi(g, &IID_IGAMEINPUT, std::ptr::null_mut()), E_POINTER);

            // IGameInputDevice
            let d = device() as *mut c_void;
            out = sentinel;
            assert_eq!(call_qi(d, &IID_IUNKNOWN, &mut out), S_OK);
            assert_eq!(out, d);
            out = sentinel;
            assert_eq!(call_qi(d, &IID_IGAMEINPUT_DEVICE, &mut out), S_OK);
            assert_eq!(out, d);
            for bad in [&IID_IGAMEINPUT, &IID_IGAMEINPUT_READING, &UNKNOWN_IID] {
                out = sentinel;
                assert_eq!(call_qi(d, bad, &mut out), E_NOINTERFACE);
                assert!(out.is_null());
            }
            out = sentinel;
            assert_eq!(call_qi(d, std::ptr::null(), &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!(call_qi(d, &IID_IGAMEINPUT_DEVICE, std::ptr::null_mut()), E_POINTER);

            // IGameInputReading (refcounted)
            let r = new_reading(GamepadState::default());
            let rp = r as *mut c_void;
            out = sentinel;
            assert_eq!(call_qi(rp, &IID_IUNKNOWN, &mut out), S_OK);
            assert_eq!(out, rp);
            assert_eq!((*r).refs.load(Ordering::Acquire), 2);
            out = sentinel;
            assert_eq!(call_qi(rp, &IID_IGAMEINPUT_READING, &mut out), S_OK);
            assert_eq!(out, rp);
            assert_eq!((*r).refs.load(Ordering::Acquire), 3);
            for bad in [&IID_IGAMEINPUT, &IID_IGAMEINPUT_DEVICE, &UNKNOWN_IID] {
                out = sentinel;
                assert_eq!(call_qi(rp, bad, &mut out), E_NOINTERFACE);
                assert!(out.is_null());
                assert_eq!((*r).refs.load(Ordering::Acquire), 3, "failed QI must not AddRef");
            }
            out = sentinel;
            assert_eq!(call_qi(rp, std::ptr::null(), &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!((*r).refs.load(Ordering::Acquire), 3);
            assert_eq!(call_qi(rp, &IID_IGAMEINPUT_READING, std::ptr::null_mut()), E_POINTER);
            assert_eq!((*r).refs.load(Ordering::Acquire), 3);

            assert_eq!(reading_release(r), 2);
            assert_eq!(reading_release(r), 1);
            assert_eq!(reading_release(r), 0);
        }
    }

    struct CallbackGate {
        entered: (Mutex<bool>, Condvar),
        proceed: (Mutex<bool>, Condvar),
        calls: AtomicU32,
        finished: std::sync::atomic::AtomicBool,
    }

    impl CallbackGate {
        fn new() -> Self {
            Self {
                entered: (Mutex::new(false), Condvar::new()),
                proceed: (Mutex::new(false), Condvar::new()),
                calls: AtomicU32::new(0),
                finished: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn on_callback(&self) {
            let n = self.calls.fetch_add(1, Ordering::AcqRel);
            if n == 0 {
                {
                    let mut e = self.entered.0.lock().unwrap();
                    *e = true;
                    self.entered.1.notify_all();
                }
                let mut p = self.proceed.0.lock().unwrap();
                while !*p {
                    p = self.proceed.1.wait(p).unwrap();
                }
                self.finished.store(true, Ordering::Release);
            }
        }

        fn wait_entered(&self) {
            let mut e = self.entered.0.lock().unwrap();
            while !*e {
                e = self.entered.1.wait(e).unwrap();
            }
        }

        fn release(&self) {
            let mut p = self.proceed.0.lock().unwrap();
            *p = true;
            self.proceed.1.notify_all();
        }
    }

    unsafe extern "system" fn gated_reading_cb(_token: u64, ctx: *mut c_void, _reading: *mut Reading, _overflow: bool) {
        let gate = &*(ctx as *const CallbackGate);
        gate.on_callback();
    }

    unsafe extern "system" fn gated_device_cb(_token: u64, ctx: *mut c_void, _dev: *mut Object, _ts: u64, _cur: u32, _prev: u32) {
        let gate = &*(ctx as *const CallbackGate);
        gate.on_callback();
    }

    static READING_CB_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn unregister_reading_callback_waits_for_in_flight_and_prevents_uaf() {
        let _serial = READING_CB_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gate = Arc::new(CallbackGate::new());
        let g = input();
        let mut token = 0u64;
        unsafe {
            assert_eq!(
                register_reading_callback(g, std::ptr::null_mut(), KIND_GAMEPAD, 0.0, Arc::as_ptr(&gate) as *mut c_void, Some(gated_reading_cb), &mut token),
                S_OK
            );
        }

        let dispatch_thread = std::thread::spawn(|| {
            dispatch_reading_callbacks(GamepadState { buttons: A, ..Default::default() });
        });
        gate.wait_entered();

        // Zero and short timeouts must fail while the callback is still in flight.
        unsafe {
            assert!(!unregister_callback(g, token, 0));
            assert!(!unregister_callback(g, token, 15_000));
        }
        assert!(!gate.finished.load(Ordering::Acquire));

        let gate_rel = Arc::clone(&gate);
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            gate_rel.release();
        });

        // Unregister must wait until the in-flight invocation completes before returning true.
        let ok = unsafe { unregister_callback(g, token, 5_000_000) };
        assert!(ok);
        assert!(gate.finished.load(Ordering::Acquire), "callback must finish before UnregisterCallback returns true");
        releaser.join().unwrap();
        dispatch_thread.join().unwrap();

        // Subsequent dispatches must never invoke the unregistered callback again.
        let calls_before = gate.calls.load(Ordering::Acquire);
        dispatch_reading_callbacks(GamepadState { buttons: B, ..Default::default() });
        assert_eq!(gate.calls.load(Ordering::Acquire), calls_before);
    }

    #[test]
    fn stop_then_unregister_still_waits_for_in_flight_callback() {
        let _serial = READING_CB_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gate = Arc::new(CallbackGate::new());
        let g = input();
        let mut token = 0u64;
        unsafe {
            assert_eq!(
                register_reading_callback(g, std::ptr::null_mut(), KIND_GAMEPAD, 0.0, Arc::as_ptr(&gate) as *mut c_void, Some(gated_reading_cb), &mut token),
                S_OK
            );
        }

        let dispatch_thread = std::thread::spawn(|| {
            dispatch_reading_callbacks(GamepadState { buttons: X, ..Default::default() });
        });
        gate.wait_entered();

        // StopCallback must not discard tracking while an invocation is still running.
        unsafe { stop_callback(g, token) };
        assert!(!gate.finished.load(Ordering::Acquire));

        let gate_rel = Arc::clone(&gate);
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            gate_rel.release();
        });

        // u64::MAX (UINT64_MAX) must wait indefinitely without overflowing Instant.
        assert!(unsafe { unregister_callback(g, token, u64::MAX) });
        assert!(gate.finished.load(Ordering::Acquire));
        releaser.join().unwrap();
        dispatch_thread.join().unwrap();
    }

    #[test]
    fn self_unregister_from_inside_callback_does_not_deadlock() {
        let _serial = READING_CB_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        struct SelfCtx {
            calls: AtomicU32,
            unreg_ok: std::sync::atomic::AtomicBool,
        }
        unsafe extern "system" fn self_unreg_cb(token: u64, ctx: *mut c_void, _reading: *mut Reading, _overflow: bool) {
            let c = &*(ctx as *const SelfCtx);
            c.calls.fetch_add(1, Ordering::AcqRel);
            let ok = unregister_callback(input(), token, u64::MAX);
            c.unreg_ok.store(ok, Ordering::Release);
        }

        let ctx = Box::new(SelfCtx {
            calls: AtomicU32::new(0),
            unreg_ok: std::sync::atomic::AtomicBool::new(false),
        });
        let g = input();
        let mut token = 0u64;
        unsafe {
            assert_eq!(
                register_reading_callback(g, std::ptr::null_mut(), KIND_GAMEPAD, 0.0, &*ctx as *const SelfCtx as *mut c_void, Some(self_unreg_cb), &mut token),
                S_OK
            );
        }

        dispatch_reading_callbacks(GamepadState { buttons: Y, ..Default::default() });
        assert_eq!(ctx.calls.load(Ordering::Acquire), 1);
        assert!(ctx.unreg_ok.load(Ordering::Acquire));

        // Must not fire again after self-unregister.
        dispatch_reading_callbacks(GamepadState { buttons: MENU, ..Default::default() });
        assert_eq!(ctx.calls.load(Ordering::Acquire), 1);
    }

    #[test]
    fn unregister_async_device_callback_waits_for_enum_thread() {
        let gate = Arc::new(CallbackGate::new());
        let g = input();
        let mut token = 0u64;
        unsafe {
            assert_eq!(
                register_device_callback(g, std::ptr::null_mut(), KIND_GAMEPAD, STATUS, 1, Arc::as_ptr(&gate) as *mut c_void, Some(gated_device_cb), &mut token),
                S_OK
            );
        }

        gate.wait_entered();
        assert!(!unsafe { unregister_callback(g, token, 0) });
        assert!(!gate.finished.load(Ordering::Acquire));

        let gate_rel = Arc::clone(&gate);
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            gate_rel.release();
        });

        assert!(unsafe { unregister_callback(g, token, 5_000_000) });
        assert!(gate.finished.load(Ordering::Acquire));
        releaser.join().unwrap();
    }
}
