//! The XGameRuntime thunk protocol – how GDK titles and GDK libraries (XCurl, libHttpClient)
//! reach the runtime. They link a static thunk library that loads `xgameruntime.dll`, calls
//! `InitializeApiImpl(Ex/Ex2)` and gets each API family as a COM-style object from
//! `QueryApiImpl(clsid, iid)`; an API call is a call through a slot of its vtable (0..2 are
//! IUnknown). The by-name exports in `mod.rs` are for titles that import them directly.
//!
//! XCL answers itself for what must stay offline and XCL-managed – XUser (one local user, its
//! devices), XGameSave (saves in the XCL data folder), persistent local storage, XPackage (a
//! fully installed package), XGame (title id from the game's MicrosoftGame.Config), XSystem and
//! its analytics (a Series X / S), runtime features, display deferrals, game events (offline:
//! dropped), the XGameUi dialogs that do not take a user –, refuses XStore, app broadcast and
//! app capture (the thunk then returns its "API not available" error), and passes everything
//! else to the PC's own XGameRuntime (`System32\xgameruntime.dll`: the same classes under the
//! same GUIDs) – XTaskQueue / XAsync, XThread, XGameInvite… (XNetworking is always XCL's). Once
//! the PC runtime is in use, the queues and async blocks are its own, so XCL's async APIs run on
//! its XAsync (`xasync::use_pc`); without `System32\xgameruntime.dll`, XAsync / XTaskQueue /
//! XThread fall back to XCL's own. Nothing that takes an XCL user handle goes to the PC runtime: it does not
//! validate foreign handles.
//!
//! Layout: `gdk_table` (extracted from the public PC GDK's thunk library) gives every
//! function's (class, interface, slot); XCL's objects are built from it by
//! function name. A class's interface ids that do not share slots share one vtable (later GDK
//! versions append functions under new interface ids); an interface id the table does not know
//! (another GDK version) gets the class's main vtable. The slot maps agree with what the thunks
//! in the game's DLLs use (argument and result widths) and with the names in the PC runtime's
//! slots. Functions XCL does not implement are logged and listed in the report by name
//! (`xgameruntime.unimplemented`).

#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use super::gamesave::*;
use super::*;
use crate::util::{E_INVALIDARG, HRESULT, S_OK};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Guid(pub u32, pub u16, pub u16, pub [u8; 8]);

impl Guid {
    pub const fn from_u128(v: u128) -> Guid {
        Guid((v >> 96) as u32, (v >> 80) as u16, (v >> 64) as u16, (v as u64).to_be_bytes())
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let d = &self.3;
        write!(f, "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{}", self.0, self.1, self.2, d[0], d[1], d[2..].iter().map(|b| format!("{b:02x}")).collect::<String>())
    }
}

const fn g(v: u128) -> Guid {
    Guid::from_u128(v)
}

pub const XUSER: Guid = g(0x01acd177_91f9_4763_a38e_ccbb55ce32e0);
/// interface of XUserGetGamertag (with a gamertag component): its own vtable
#[cfg_attr(not(test), allow(dead_code))]
const XUSER_GAMERTAG_IID: Guid = g(0xcef4fac0_7676_4a94_a119_4c43f9eb5b74);
/// older console GDK XUser IID (same base slot layout)
const XUSER_V2_IID: Guid = g(0x1bf2f8c5_d507_4e52_bb05_f726d0e71161);
const XUSER_DEVICE: Guid = g(0x7d824997_10dc_45ab_86b7_2737767c0bf1);
pub const XGAMESAVE: Guid = g(0x704c3f58_e629_4cc2_b197_30511b996fe2);
/// the interface the game's libHttpClient asks for (GDK table: 0b9d556e)
#[cfg_attr(not(test), allow(dead_code))]
const XPACKAGE_OLD_IID: Guid = g(0xe2a4734b_2f4a_456d_aa8f_d065e04fb209);
/// console GDK 19041 XPackage v1 IID: XPackageEnumeratePackages is at slot 28
const XPACKAGE_V1_IID: Guid = g(0xbe3f3e9f_c4b3_424d_9a95_d1af060559a5);
/// console GDK 19041/22000 XPackage v2 IID: EnumeratePackages @ 35, RegisterPackageInstalled @ 36
const XPACKAGE_V2_IID: Guid = g(0xf92d8712_2b27_4d8a_bf01_11a6f8e3eb42);
pub const XPACKAGE: Guid = g(0xaf406016_e850_4aa8_a88d_2f3dcb9dac7e);
pub const XPLS: Guid = g(0xf4faf4d4_2d04_4fce_b3e0_474a713a3e84);
const XPLS_V1_IID: Guid = g(0xd29411df_0794_4553_8b27_95fc02d0f75d);
pub const XGAME: Guid = g(0x973a344e_24bf_4d0f_8457_56c534892b29);
const XGAME_V1_IID: Guid = g(0x50849859_0ad8_4f81_80e4_5bc78626f852);
pub const XSYSTEM: Guid = g(0xe349bd1a_fc20_4e40_b99c_4178cc6b409f);
const XSYSTEM_V2_IID: Guid = g(0x67ce4bfc_b1d1_4ac7_bc3a_cb9219a97a85);
/// console GDK XSystem v3 IID: XSystemGetAppSpecificDeviceId is at slot 4 (not slot 5)
const XSYSTEM_V3_IID: Guid = g(0x6fd71f09_7513_49f0_89bc_bfaf5df6f852);
const XANALYTICS: Guid = g(0xb884675d_b738_4a9c_815d_9a9a1e0c6c9b);
const XFEATURE: Guid = g(0x8836fe87_edb9_4fe3_8dad_05f0d2cd5b40);
const XDISPLAY: Guid = g(0x1b339674_328d_4283_a200_3171f18d3639);
const XGAMEEVENT: Guid = g(0xbbfbdcc7_bfe7_409b_a5ca_edf054960b4d);
pub const XNETWORKING: Guid = g(0x37e56907_2f10_41e8_b72f_36edb185331a);
pub const XERROR: Guid = g(0x8ca467f7_22e8_4096_8456_bb8aa13f79d8);
pub const XACCESSIBILITY: Guid = g(0x3e241977_6237_41e9_8559_102c2d9983f1);
pub const XGAMEUI: Guid = g(0xdfcd4649_4ff8_4043_ba07_35d607df98b0);
const XSTORE: Guid = g(0x0dd112ac_7c24_448c_b92b_3960fb5bd30c);
const XAPPBROADCAST: Guid = g(0x186d5592_a72d_45fb_9560_11aed0e6647a);
const XAPPCAPTURE: Guid = g(0xa4f1aee2_4bf1_4485_b008_a7c26d52ac27);
/// XAsync + XTaskQueue + XThread (iid = clsid)
pub const XASYNC: Guid = g(0x073b7dcb_1fcf_4030_94be_e3c9eb623428);

/// classes that reach Xbox Live / the Store or take XCL's user handles: not available
const REFUSED: [(Guid, &str); 3] = [(XSTORE, "XStore"), (XAPPBROADCAST, "XAppBroadcast"), (XAPPCAPTURE, "XAppCapture")];

/// rows of the GDK table: (function, class, interface, slot)
fn table() -> impl Iterator<Item = (&'static str, Guid, Guid, usize)> {
    super::gdk_table::API.iter().map(|&(n, c, i, s)| (n, g(c), g(i), s as usize))
}

/// a class's name for the log: the prefix of its first function in the GDK table
fn family(c: &Guid) -> String {
    let Some((n, ..)) = table().find(|r| r.1 == *c) else { return "?".into() };
    let end = n.char_indices().skip(1).filter(|(_, ch)| ch.is_ascii_uppercase()).nth(1).map_or(n.len(), |(k, _)| k);
    n[..end].to_string()
}

const REGDB_E_CLASSNOTREG: HRESULT = 0x8004_0154_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;

// ---------------------------------------------------------------------------------------
// the PC runtime
// ---------------------------------------------------------------------------------------

type QueryFn = unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> HRESULT;
type InitEx2 = unsafe extern "system" fn(u64, u64, u64, *mut c_void) -> HRESULT;
type InitEx = unsafe extern "system" fn(u64, u64, u64) -> HRESULT;
type Init = unsafe extern "system" fn(u64, u64) -> HRESULT;

struct Pc {
    query: QueryFn,
    init_ex2: Option<InitEx2>,
    init_ex: Option<InitEx>,
    init: Option<Init>,
    uninit: Option<unsafe extern "system" fn()>,
}

static PC: OnceLock<Option<Pc>> = OnceLock::new();

fn pc() -> Option<&'static Pc> {
    PC.get_or_init(|| unsafe {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
        let h = LoadLibraryExW(crate::util::wide("xgameruntime.dll").as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
        if h.is_null() {
            crate::xlog_warn!("XGameRuntime: no System32\\xgameruntime.dll (Gaming Services) – XCL's own XTaskQueue/XAsync are used; dialogs, text entry, game invites, game streaming and the other APIs XCL does not implement are unavailable");
            return None;
        }
        let f = |n: &str| GetProcAddress(h, format!("{n}\0").as_ptr()).map(|p| p as usize);
        let query = f("QueryApiImpl")?;
        crate::xlog_info!("XGameRuntime: APIs XCL does not implement come from the system xgameruntime.dll");
        Some(Pc {
            query: std::mem::transmute::<usize, QueryFn>(query),
            init_ex2: f("InitializeApiImplEx2").map(|p| std::mem::transmute::<usize, InitEx2>(p)),
            init_ex: f("InitializeApiImplEx").map(|p| std::mem::transmute::<usize, InitEx>(p)),
            init: f("InitializeApiImpl").map(|p| std::mem::transmute::<usize, Init>(p)),
            uninit: f("UninitializeApiImpl").map(|p| std::mem::transmute::<usize, unsafe extern "system" fn()>(p)),
        })
    })
    .as_ref()
}

unsafe fn pc_object(c: &Guid, i: &Guid) -> Option<usize> {
    let mut o: *mut c_void = std::ptr::null_mut();
    ((pc()?.query)(c, i, &mut o) >= 0 && !o.is_null()).then_some(o as usize)
}

/// After the PC runtime is initialised: XCL's async APIs on its XAsync.
unsafe fn start_pc_async() {
    if let Some(o) = pc_object(&XASYNC, &XASYNC) {
        super::xasync::use_pc(o);
    }
}

// ---------------------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------------------

pub unsafe extern "system" fn InitializeApiImplEx2(gdk: u64, os: u64, flags: u64, options: *mut c_void) -> HRESULT {
    crate::xlog_info!("XGameRuntime: InitializeApiImplEx2({gdk:#x}, {os:#x}, flags {flags:#x}) – title goes through the GDK thunks");
    pc_init(gdk, os, flags, options);
    S_OK
}

/// Initialise the PC runtime with the title's versions; options this PC runtime does not know
/// (newer GDK) fall back to the older entry points.
unsafe fn pc_init(gdk: u64, os: u64, flags: u64, options: *mut c_void) {
    let Some(p) = pc() else { return };
    let mut hr = p.init_ex2.map(|f| f(gdk, os, flags, options)).unwrap_or(-1);
    if hr < 0 {
        hr = p.init_ex.map(|f| f(gdk, os, flags)).unwrap_or(-1);
    }
    if hr < 0 {
        hr = p.init.map(|f| f(gdk >> 32, gdk & 0xFFFF_FFFF)).unwrap_or(-1);
    }
    if hr < 0 {
        crate::xlog_warn!("XGameRuntime PC: initialization → {:#x}", hr as u32);
    }
    start_pc_async();
}

pub unsafe extern "system" fn InitializeApiImplEx(gdk: u64, os: u64, flags: u64) -> HRESULT {
    crate::xlog_info!("XGameRuntime: InitializeApiImplEx({gdk:#x}, {os:#x}, flags {flags:#x})");
    pc_init(gdk, os, flags, std::ptr::null_mut());
    S_OK
}

pub unsafe extern "system" fn InitializeApiImpl(gdk: u64, os: u64) -> HRESULT {
    crate::xlog_info!("XGameRuntime: InitializeApiImpl({gdk:#x}, {os:#x})");
    pc_init(gdk << 32 | (os & 0xFFFF_FFFF), 0, 0, std::ptr::null_mut());
    S_OK
}

pub unsafe extern "system" fn UninitializeApiImpl() {
    if let Some(f) = pc().and_then(|p| p.uninit) {
        f();
    }
}

static ERROR_CB: Mutex<Option<(usize, usize)>> = Mutex::new(None);
static ERROR_OPTIONS: AtomicUsize = AtomicUsize::new(0);

pub unsafe extern "system" fn XErrorReport(hr: HRESULT, message: *const u8) {
    let text = crate::util::from_cstr_ptr(message).unwrap_or_default();
    crate::xlog_warn!("XErrorReport {:#x}: {text}", hr as u32);
    // Copy and release `ERROR_CB` before invoking the guest callback so re-entrant calls to
    // `XErrorSetCallback` or `XErrorReport` do not deadlock.
    let entry = *ERROR_CB.lock().unwrap();
    if let Some((cb, ctx)) = entry {
        let f: unsafe extern "system" fn(HRESULT, *const u8, *mut c_void) -> bool = std::mem::transmute(cb);
        let _ = f(hr, message, ctx as *mut c_void);
    }
}

unsafe extern "system" fn error_set_callback(cb: usize, ctx: *mut c_void) {
    *ERROR_CB.lock().unwrap() = (cb != 0).then_some((cb, ctx as usize));
}

unsafe extern "system" fn error_set_options(options: u32, _reserved: usize) {
    // Store requested options without triggering DebugBreak in-process.
    ERROR_OPTIONS.store(options as usize, Ordering::Relaxed);
}

/// Separate per-IID clones for forwarding classes (such as XGameUi) when a title queries an
/// older/unknown interface version, preventing its shorter PC vtable from poisoning the main
/// interface's cached `pc` pointer.
fn forward_alias_obj(base: &'static Obj, c: Guid, i: Guid) -> &'static Obj {
    static ALIASES: OnceLock<Mutex<BTreeMap<(Guid, Guid), &'static Obj>>> = OnceLock::new();
    let map = ALIASES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut g = map.lock().unwrap();
    if let Some(&existing) = g.get(&(c, i)) {
        return existing;
    }
    let iface = Iface {
        name: base.iface.name,
        clsid: base.iface.clsid,
        iids: vec![i],
        main: false,
        names: base.iface.names.clone(),
        slots: base.iface.slots.clone(),
        forward: base.iface.forward.clone(),
    };
    let obj = object(iface);
    obj.pc.store(unsafe { pc_object(&c, &i).unwrap_or(0) }, Ordering::Release);
    g.insert((c, i), obj);
    obj
}

pub unsafe extern "system" fn QueryApiImpl(clsid: *const Guid, iid: *const Guid, out: *mut *mut c_void) -> HRESULT {
    if clsid.is_null() || iid.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    *out = std::ptr::null_mut();
    let (c, i) = (*clsid, *iid);
    if let Some((_, name)) = REFUSED.iter().find(|r| r.0 == c) {
        note(c, i, format!("{name}: unavailable (offline)"));
        return REGDB_E_CLASSNOTREG;
    }
    if let Some(o) = own(&c, &i) {
        let known = o.iface.iids.contains(&i) || i == c;
        note(c, i, format!("{}: XCL{}", o.iface.name, if known { "" } else { " (interface version not in the GDK table: the main vtable)" }));
        let target = if !o.iface.forward.is_empty() {
            if known {
                if o.pc.load(Ordering::Acquire) == 0 {
                    o.pc.store(pc_object(&c, &i).unwrap_or(0), Ordering::Release);
                }
                o
            } else {
                forward_alias_obj(o, c, i)
            }
        } else {
            o
        };
        *out = target as *const Obj as *mut c_void;
        return S_OK;
    }
    match pc_object(&c, &i) {
        Some(p) => {
            note(c, i, format!("{}: PC", family(&c)));
            *out = p as *mut c_void;
            S_OK
        }
        None if c == XASYNC => {
            note(c, i, "XAsync: XCL (fallback – no system xgameruntime.dll)".into());
            *out = xcl_xasync_object() as *const Obj as *mut c_void;
            S_OK
        }
        None if c == XACCESSIBILITY => {
            note(c, i, "XAccessibility: XCL (fallback – no system xgameruntime.dll)".into());
            *out = xcl_xaccessibility_object() as *const Obj as *mut c_void;
            S_OK
        }
        None => {
            note(c, i, format!("{}: missing", family(&c)));
            REGDB_E_CLASSNOTREG
        }
    }
}

// ---------------------------------------------------------------------------------------
// XCL's API objects
// ---------------------------------------------------------------------------------------

/// One vtable of an XCL class.
struct Iface {
    name: &'static str,
    clsid: Guid,
    /// interface ids this vtable answers
    iids: Vec<Guid>,
    /// the class's main vtable: also for interface ids the GDK table does not know
    main: bool,
    /// slot → function (GDK table)
    names: Vec<Option<&'static str>>,
    /// (slot, function): the API function itself, without `this` (a shim drops it)
    slots: Vec<(usize, usize)>,
    /// slots passed to the PC runtime's object of the same interface
    forward: Vec<usize>,
}

/// A class XCL answers for: its functions by GDK name, and the ones passed to the PC runtime.
struct Class {
    name: &'static str,
    clsid: Guid,
    impls: Vec<(&'static str, usize)>,
    forward: &'static [&'static str],
}

/// Functions that return bool (an unimplemented one answers false).
fn returns_bool(name: &str) -> bool {
    let rest = name.trim_start_matches('X');
    let verb = rest.char_indices().skip(1).find(|(_, ch)| ch.is_ascii_uppercase()).map_or("", |(k, _)| &rest[k..]);
    verb.starts_with("Is") || verb.contains("Unregister") || name == "XPackageUpdateInstallationMonitor" || name == "XPackageUninstallPackage"
}

#[repr(C)]
struct Obj {
    vtbl: *const usize,
    iface: Iface,
    /// the PC runtime's object, for `forward`
    pc: AtomicUsize,
    /// slot → API function the shim calls
    targets: [usize; SLOTS],
}

unsafe impl Sync for Obj {}
unsafe impl Send for Obj {}

const SLOTS: usize = 96;

unsafe extern "system" fn query_interface(this: *const Obj, iid: *const Guid, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() || iid.is_null() {
        return E_INVALIDARG;
    }
    let f = &(*this).iface;
    if *iid == f.clsid || f.iids.contains(&*iid) || (f.main && own(&f.clsid, &*iid).is_some_and(|o| std::ptr::eq(o, this))) {
        *out = this as *mut c_void;
        return S_OK;
    }
    *out = std::ptr::null_mut();
    // a title asserting on this (a newer interface version of the class?) ends without a trace otherwise
    crate::xlog_warn!("XGameRuntime: {} QueryInterface({}) refused (E_NOINTERFACE) – an interface version XCL's GDK table does not know?", f.name, *iid);
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(_this: *const Obj) -> u32 {
    1
}

static UNIMPLEMENTED: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

unsafe fn unknown_slot(this: *const Obj, slot: usize) -> usize {
    let f = &(*this).iface;
    let key = match f.names.get(slot).copied().flatten() {
        Some(n) => n.to_string(),
        None => format!("{}#{slot}", f.name),
    };
    let first = {
        let mut g = UNIMPLEMENTED.lock().unwrap();
        let n = g.entry(key.clone()).or_insert(0);
        *n += 1;
        *n == 1
    };
    if first {
        crate::xlog_warn!("XGameRuntime: {key} – not supported (E_NOTIMPL)");
    }
    if f.names.get(slot).copied().flatten().is_some_and(returns_bool) {
        0
    } else {
        E_NOTIMPL as u32 as usize
    }
}

type Fwd = unsafe extern "system" fn(*const Obj, usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

/// Call slot `slot` of the PC object with the caller's arguments (up to 9; extra ones are
/// ignored by the callee).
unsafe fn forward(this: *const Obj, slot: usize, a: [usize; 9]) -> usize {
    let o = (*this).pc.load(Ordering::Acquire);
    if o == 0 {
        return unknown_slot(this, slot);
    }
    let f: Fwd = std::mem::transmute(*(*(o as *const *const usize)).add(slot));
    f(o as *const Obj, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
}

/// Slot `slot` of an XCL object: its API function with the caller's arguments after `this`.
unsafe fn call_target(this: *const Obj, slot: usize, a: [usize; 9]) -> usize {
    let f: unsafe extern "system" fn(usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize = std::mem::transmute((*this).targets[slot]);
    f(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
}

macro_rules! per_slot {
    ($($n:literal)*) => {
        const SHIMS: [Fwd; SLOTS] = [$({
            unsafe extern "system" fn f(this: *const Obj, a: usize, b: usize, c: usize, d: usize, e: usize, g: usize, h: usize, i: usize, j: usize) -> usize {
                call_target(this, $n, [a, b, c, d, e, g, h, i, j])
            }
            f
        }),*];
        const STUBS: [unsafe extern "system" fn(*const Obj) -> usize; SLOTS] = [$({
            unsafe extern "system" fn f(this: *const Obj) -> usize {
                unknown_slot(this, $n)
            }
            f
        }),*];
        const FORWARDS: [Fwd; SLOTS] = [$({
            unsafe extern "system" fn f(this: *const Obj, a: usize, b: usize, c: usize, d: usize, e: usize, g: usize, h: usize, i: usize, j: usize) -> usize {
                forward(this, $n, [a, b, c, d, e, g, h, i, j])
            }
            f
        }),*];
    };
}
per_slot!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
    40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
    80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95);

fn object(iface: Iface) -> &'static Obj {
    let mut v: Vec<usize> = (0..SLOTS).map(|k| STUBS[k] as usize).collect();
    v[0] = query_interface as usize;
    v[1] = add_ref as usize;
    v[2] = add_ref as usize;
    for &k in &iface.forward {
        v[k] = FORWARDS[k] as usize;
    }
    let mut targets = [0usize; SLOTS];
    for &(k, f) in &iface.slots {
        v[k] = SHIMS[k] as usize;
        targets[k] = f;
    }
    let vtbl = Box::leak(v.into_boxed_slice()).as_ptr();
    Box::leak(Box::new(Obj { vtbl, iface, pc: AtomicUsize::new(0), targets }))
}

/// The vtables of a class from the GDK table: interface ids whose slots do not collide share
/// one (the one holding the class id itself, or the first, is the main one).
fn layout(c: Class) -> Vec<Iface> {
    let mut by_iid: BTreeMap<Guid, Vec<(usize, &'static str)>> = BTreeMap::new();
    for (n, cl, i, slot) in table() {
        if cl == c.clsid && slot < SLOTS {
            by_iid.entry(i).or_default().push((slot, n));
        }
    }
    let mut order: Vec<Guid> = by_iid.keys().copied().collect();
    order.sort_by_key(|i| (*i != c.clsid, by_iid[i].iter().map(|r| r.0).min()));
    let mut groups: Vec<(Vec<Guid>, Vec<Option<&'static str>>)> = Vec::new();
    for i in order {
        let rows = &by_iid[&i];
        let free = groups.iter().position(|(_, names)| rows.iter().all(|&(s, _)| names[s].is_none()));
        let k = free.unwrap_or_else(|| {
            groups.push((Vec::new(), vec![None; SLOTS]));
            groups.len() - 1
        });
        groups[k].0.push(i);
        for &(s, n) in rows {
            groups[k].1[s] = Some(n);
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(k, (iids, names))| {
            let slot_of = |f: &str| names.iter().position(|n| *n == Some(f));
            Iface {
                name: c.name,
                clsid: c.clsid,
                main: k == 0,
                slots: c.impls.iter().filter_map(|&(f, p)| Some((slot_of(f)?, p))).collect(),
                forward: c.forward.iter().filter_map(|f| slot_of(f)).collect(),
                iids,
                names,
            }
        })
        .collect()
}

fn classes() -> Vec<Class> {
    vec![
            Class { name: "XUser", clsid: XUSER, impls: xuser(), forward: &[] },
            Class { name: "XUser (devices)", clsid: XUSER_DEVICE, impls: xuser_devices(), forward: &[] },
            Class { name: "XGameSave", clsid: XGAMESAVE, impls: xgamesave(), forward: &[] },
            Class { name: "XPersistentLocalStorage", clsid: XPLS, impls: xpls(), forward: &[] },
            Class { name: "XPackage", clsid: XPACKAGE, impls: xpackage(), forward: &[] },
            Class { name: "XGame", clsid: XGAME, impls: named![XGameGetXboxTitleId, XLaunchNewGame, XLaunchRestartOnCrash], forward: &[] },
            Class { name: "XSystem", clsid: XSYSTEM, impls: xsystem(), forward: &[] },
            Class { name: "XSystemAnalytics", clsid: XANALYTICS, impls: named![XSystemGetAnalyticsInfo], forward: &[] },
            Class { name: "XGameRuntimeFeature", clsid: XFEATURE, impls: named![XGameRuntimeIsFeatureAvailable], forward: &[] },
            Class {
                name: "XDisplay",
                clsid: XDISPLAY,
                impls: vec![("XLaunchUri", launch_uri as usize), ("XDisplayAcquireTimeoutDeferral", display_deferral as usize), ("XDisplayCloseTimeoutDeferralHandle", nothing as usize)],
                forward: &[],
            },
            // telemetry: offline, accepted and dropped
            Class { name: "XGameEvent", clsid: XGAMEEVENT, impls: vec![("XGameEventWrite", ok as usize)], forward: &[] },
            Class { name: "XNetworking", clsid: XNETWORKING, impls: xnetworking(), forward: &[] },
            Class { name: "XError", clsid: XERROR, impls: xerror(), forward: &[] },
            // dialogs and text entry without a user handle go to the PC runtime
            Class {
                name: "XGameUi",
                clsid: XGAMEUI,
                impls: Vec::new(),
                forward: &[
                    "XGameUiShowMessageDialogAsync",
                    "XGameUiShowMessageDialogResult",
                    "XGameUiShowErrorDialogAsync",
                    "XGameUiShowErrorDialogResult",
                    "XGameUiShowTextEntryAsync",
                    "XGameUiShowTextEntryResultSize",
                    "XGameUiShowTextEntryResult",
                    // the in-game text entry: options and text in, a PC handle out and back
                    "XGameUiTextEntryOpen",
                    "XGameUiTextEntryClose",
                    "XGameUiTextEntryGetState",
                    "XGameUiTextEntryGetExtents",
                    "XGameUiTextEntryUpdatePositionHint",
                    "XGameUiTextEntryUpdateVisibility",
                    "XGameUiSetNotificationPositionHint",
                ],
            },
    ]
}

/// Build an explicit legacy console interface vtable from `(slot, function_name)` rows.
fn legacy_iface(
    name: &'static str,
    clsid: Guid,
    iid: Guid,
    rows: &[(usize, &'static str)],
    impls: &[(&'static str, usize)],
) -> Iface {
    let mut names = vec![None; SLOTS];
    let mut slots = Vec::new();
    for &(slot, fname) in rows {
        if slot < SLOTS {
            names[slot] = Some(fname);
            if let Some(&(_, ptr)) = impls.iter().find(|&&(n, _)| n == fname) {
                slots.push((slot, ptr));
            }
        }
    }
    Iface { name, clsid, iids: vec![iid], main: false, names, slots, forward: Vec::new() }
}

/// Older console GDK (19041 / 2020) interface versions whose vtable slot indices differ
/// from the PC GDK 261000 table.
fn legacy_ifaces() -> Vec<Iface> {
    let pkg = xpackage();
    let sys = xsystem();
    let pkg_v1 = [
        (3, "XPackageGetCurrentProcessPackageIdentifier"),
        (4, "XPackageIsPackagedProcess"),
        (5, "XPackageCreateInstallationMonitor"),
        (6, "XPackageCloseInstallationMonitorHandle"),
        (7, "XPackageGetInstallationProgress"),
        (8, "XPackageUpdateInstallationMonitor"),
        (9, "XPackageRegisterInstallationProgressChanged"),
        (10, "XPackageUnregisterInstallationProgressChanged"),
        (11, "XPackageGetUserLocale"),
        (12, "XPackageFindChunkAvailability"),
        (13, "XPackageEnumerateChunkAvailability"),
        (14, "XPackageChangeChunkInstallOrder"),
        (16, "XPackageInstallChunksAsync"),
        (17, "XPackageInstallChunksResult"),
        (28, "XPackageEnumeratePackages"),
    ];
    let pkg_v2 = [
        (3, "XPackageGetCurrentProcessPackageIdentifier"),
        (4, "XPackageIsPackagedProcess"),
        (5, "XPackageCreateInstallationMonitor"),
        (6, "XPackageCloseInstallationMonitorHandle"),
        (7, "XPackageGetInstallationProgress"),
        (8, "XPackageUpdateInstallationMonitor"),
        (9, "XPackageRegisterInstallationProgressChanged"),
        (10, "XPackageUnregisterInstallationProgressChanged"),
        (11, "XPackageGetUserLocale"),
        (18, "XPackageEstimateDownloadSize"),
        (22, "XPackageUnregisterPackageInstalled"),
        (24, "XPackageGetMountPathSize"),
        (25, "XPackageGetMountPath"),
        (26, "XPackageCloseMountHandle"),
        (30, "XPackageGetWriteStats"),
        (33, "XPackageEnumerateFeatures"),
        (35, "XPackageEnumeratePackages"),
        (36, "XPackageRegisterPackageInstalled"),
    ];
    let sys_v3 = [(4, "XSystemGetAppSpecificDeviceId")];
    vec![
        legacy_iface("XPackage", XPACKAGE, XPACKAGE_V1_IID, &pkg_v1, &pkg),
        legacy_iface("XPackage", XPACKAGE, XPACKAGE_V2_IID, &pkg_v2, &pkg),
        legacy_iface("XSystem", XSYSTEM, XSYSTEM_V3_IID, &sys_v3, &sys),
    ]
}

/// Built-in XAsync / XTaskQueue / XThread COM object used as a fallback when Gaming Services
/// (`System32\xgameruntime.dll`) is not installed on the host.
fn xcl_xasync_object() -> &'static Obj {
    static O: OnceLock<&'static Obj> = OnceLock::new();
    O.get_or_init(|| {
        let mut ifaces = layout(Class { name: "XAsync", clsid: XASYNC, impls: xasync_impls(), forward: &[] });
        object(ifaces.remove(0))
    })
}

/// Built-in XAccessibility COM object used as a fallback when Gaming Services
/// (`System32\xgameruntime.dll`) is not installed on the host.
fn xcl_xaccessibility_object() -> &'static Obj {
    static O: OnceLock<&'static Obj> = OnceLock::new();
    O.get_or_init(|| {
        let mut ifaces = layout(Class {
            name: "XAccessibility",
            clsid: XACCESSIBILITY,
            impls: named![XClosedCaptionGetProperties, XClosedCaptionSetEnabled, XHighContrastGetMode],
            forward: &[],
        });
        object(ifaces.remove(0))
    })
}

fn objects() -> &'static [&'static Obj] {
    static O: OnceLock<Vec<&'static Obj>> = OnceLock::new();
    O.get_or_init(|| {
        let mut ifaces: Vec<Iface> = classes().into_iter().flat_map(layout).collect();
        // Register older compatible console IIDs whose slot layouts match the main vtable.
        for f in &mut ifaces {
            if !f.main {
                continue;
            }
            match f.clsid {
                c if c == XUSER => f.iids.push(XUSER_V2_IID),
                c if c == XSYSTEM => f.iids.push(XSYSTEM_V2_IID),
                c if c == XGAME => f.iids.push(XGAME_V1_IID),
                c if c == XPLS => f.iids.push(XPLS_V1_IID),
                c if c == XNETWORKING => f.iids.push(XNETWORKING),
                _ => {}
            }
        }
        ifaces.extend(legacy_ifaces());
        ifaces.into_iter().map(object).collect()
    })
}

/// XCL's object for (clsid, iid): the vtable holding that interface, else the class's main one.
fn own(c: &Guid, i: &Guid) -> Option<&'static Obj> {
    let all = objects();
    all.iter().copied().find(|o| o.iface.clsid == *c && o.iface.iids.contains(i)).or_else(|| all.iter().copied().find(|o| o.iface.clsid == *c && o.iface.main))
}

/// `(name, function)` pairs for functions named like their GDK function
macro_rules! named {
    ($($f:ident),* $(,)?) => {
        vec![$( (stringify!($f), $f as usize) ),*]
    };
}
use named;

fn xuser() -> Vec<(&'static str, usize)> {
    let mut v = named![
        XUserDuplicateHandle,
        XUserCloseHandle,
        XUserCompare,
        XUserGetMaxUsers,
        XUserAddAsync,
        XUserAddResult,
        XUserGetLocalId,
        XUserGetId,
        XUserGetIsGuest,
        XUserGetState,
        XUserGetGamertag,
        XUserGetAgeGroup,
        XUserCheckPrivilege,
        XUserRegisterForChangeEvent,
        XUserUnregisterForChangeEvent,
        XUserGetTokenAndSignatureAsync,
        XUserGetTokenAndSignatureResultSize,
        XUserGetTokenAndSignatureResult,
        XUserGetTokenAndSignatureUtf16Async,
        XUserGetTokenAndSignatureUtf16ResultSize,
        XUserGetTokenAndSignatureUtf16Result,
    ];
    v.extend([
        ("XUserFindUserByLocalId", find_user as usize),
        ("XUserFindUserById", find_user as usize),
        ("XUserResolvePrivilegeWithUiAsync", resolve_privilege_async as usize),
        ("XUserResolvePrivilegeWithUiResult", empty_result as usize),
        ("XUserResolveIssueWithUiAsync", resolve_issue_async as usize),
        ("XUserResolveIssueWithUiResult", empty_result as usize),
        ("XUserResolveIssueWithUiUtf16Async", resolve_issue_async as usize),
        ("XUserResolveIssueWithUiUtf16Result", empty_result as usize),
        ("XUserGetSignOutDeferral", sign_out_deferral as usize),
        ("XUserCloseSignOutDeferralHandle", nothing as usize),
        ("XUserAddByIdWithUiAsync", add_by_id_async as usize),
        ("XUserAddByIdWithUiResult", XUserAddResult as usize),
        ("XUserIsStoreUser", no as usize),
        ("XUserPlatformRemoteConnectSetEventHandlers", ok as usize),
        ("XUserPlatformRemoteConnectCancelPrompt", ok as usize),
        ("XUserPlatformSpopPromptSetEventHandlers", ok as usize),
        ("XUserPlatformSpopPromptComplete", ok as usize),
        ("XUserIsSignOutPresent", no as usize),
        ("XUserSignOutAsync", sign_out_async as usize),
        ("XUserSignOutResult", empty_result as usize),
        ("XUserGetGamerPictureAsync", gamer_picture_async as usize),
        ("XUserGetGamerPictureResultSize", XAsyncGetResultSize as usize),
        ("XUserGetGamerPictureResult", gamer_picture_result as usize),
    ]);
    v
}

/// device association: every controller belongs to the single local signed-in user
fn xuser_devices() -> Vec<(&'static str, usize)> {
    vec![
        ("XUserRegisterForDeviceAssociationChanged", XUserRegisterForDeviceAssociationChanged as usize),
        ("XUserUnregisterForDeviceAssociationChanged", XUserUnregisterForDeviceAssociationChanged as usize),
        ("XUserRegisterForDefaultAudioEndpointUtf16Changed", register_token4 as usize),
        ("XUserUnregisterForDefaultAudioEndpointUtf16Changed", yes as usize),
        ("XUserFindForDevice", XUserFindForDevice as usize),
        ("XUserGetDefaultAudioEndpointUtf16", default_audio_endpoint as usize),
        ("XUserFindControllerForUserWithUiAsync", XUserFindControllerForUserWithUiAsync as usize),
        ("XUserFindControllerForUserWithUiResult", XUserFindControllerForUserWithUiResult as usize),
    ]
}

fn xgamesave() -> Vec<(&'static str, usize)> {
    named![
        XGameSaveInitializeProvider,
        XGameSaveInitializeProviderAsync,
        XGameSaveInitializeProviderResult,
        XGameSaveCloseProvider,
        XGameSaveGetRemainingQuota,
        XGameSaveGetRemainingQuotaAsync,
        XGameSaveGetRemainingQuotaResult,
        XGameSaveDeleteContainer,
        XGameSaveDeleteContainerAsync,
        XGameSaveDeleteContainerResult,
        XGameSaveGetContainerInfo,
        XGameSaveEnumerateContainerInfo,
        XGameSaveEnumerateContainerInfoByName,
        XGameSaveCreateContainer,
        XGameSaveCloseContainer,
        XGameSaveEnumerateBlobInfo,
        XGameSaveEnumerateBlobInfoByName,
        XGameSaveReadBlobData,
        XGameSaveReadBlobDataAsync,
        XGameSaveReadBlobDataResult,
        XGameSaveCreateUpdate,
        XGameSaveCloseUpdate,
        XGameSaveSubmitBlobWrite,
        XGameSaveSubmitBlobDelete,
        XGameSaveSubmitUpdate,
        XGameSaveSubmitUpdateAsync,
        XGameSaveSubmitUpdateResult,
        XGameSaveFilesGetFolderWithUiAsync,
        XGameSaveFilesGetFolderWithUiResult,
        XGameSaveFilesGetRemainingQuota,
    ]
}

fn xpls() -> Vec<(&'static str, usize)> {
    let mut v = named![XPersistentLocalStorageGetPathSize, XPersistentLocalStorageGetPath, XPersistentLocalStorageGetSpaceInfo];
    // no prompt to show: the space asked for is granted
    v.extend([("XPersistentLocalStoragePromptUserForSpaceAsync", prompt_space_async as usize), ("XPersistentLocalStoragePromptUserForSpaceResult", empty_result as usize)]);
    v
}

fn xpackage() -> Vec<(&'static str, usize)> {
    let mut v = named![
        XPackageGetCurrentProcessPackageIdentifier,
        XPackageGetIdentifier,
        XPackageIsPackagedProcess,
        XPackageCreateInstallationMonitor,
        XPackageGetInstallationProgress,
        XPackageRegisterInstallationProgressChanged,
        XPackageUnregisterInstallationProgressChanged,
        XPackageGetUserLocale,
        XPackageFindChunkAvailability,
        XPackageEstimateDownloadSize,
        XPackageGetMountPathSize,
        XPackageGetMountPath,
        XPackageCloseMountHandle,
        XPackageGetWriteStats,
        XPackageGetPackageKind,
    ];
    v.extend([
        ("XPackageCloseInstallationMonitorHandle", nothing as usize),
        ("XPackageUpdateInstallationMonitor", no as usize),
        ("XPackageEnumerateChunkAvailability", ok as usize),
        ("XPackageChangeChunkInstallOrder", ok as usize),
        ("XPackageInstallChunks", install_chunks as usize),
        ("XPackageInstallChunksAsync", install_chunks_async as usize),
        ("XPackageInstallChunksResult", install_chunks_result as usize),
        ("XPackageUninstallChunks", ok as usize),
        ("XPackageRegisterPackageInstalled", register_token4 as usize),
        ("XPackageUnregisterPackageInstalled", yes as usize),
        ("XPackageEnumerateFeatures", ok as usize),
        ("XPackageUninstallPackage", no as usize),
        ("XPackageMountWithUiAsync", mount_with_ui_async as usize),
        ("XPackageMountWithUiResult", mount_with_ui_result as usize),
        ("XPackageEnumeratePackages", enumerate_packages as usize),
    ]);
    v
}

fn xsystem() -> Vec<(&'static str, usize)> {
    let mut v = named![XSystemGetConsoleId, XSystemGetXboxLiveSandboxId, XSystemGetAppSpecificDeviceId];
    v.extend([("XSystemHandleTrack", ok as usize), ("XSystemIsHandleValid", yes as usize), ("XSystemAllowFullDownloadBandwidth", nothing as usize)]);
    v
}

fn xnetworking() -> Vec<(&'static str, usize)> {
    named![
        XNetworkingQueryPreferredLocalUdpMultiplayerPort,
        XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync,
        XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult,
        XNetworkingRegisterPreferredLocalUdpMultiplayerPortChanged,
        XNetworkingUnregisterPreferredLocalUdpMultiplayerPortChanged,
        XNetworkingQuerySecurityInformationForUrlAsync,
        XNetworkingQuerySecurityInformationForUrlAsyncResultSize,
        XNetworkingQuerySecurityInformationForUrlAsyncResult,
        XNetworkingQuerySecurityInformationForUrlUtf16Async,
        XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize,
        XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult,
        XNetworkingVerifyServerCertificate,
        XNetworkingGetConnectivityHint,
        XNetworkingRegisterConnectivityHintChanged,
        XNetworkingUnregisterConnectivityHintChanged,
    ]
}

fn xerror() -> Vec<(&'static str, usize)> {
    vec![
        ("XErrorReport", XErrorReport as usize),
        ("XErrorSetCallback", error_set_callback as usize),
        ("XErrorSetOptions", error_set_options as usize),
    ]
}

fn xasync_impls() -> Vec<(&'static str, usize)> {
    named![
        XAsyncGetStatus,
        XAsyncGetResultSize,
        XAsyncCancel,
        XAsyncRun,
        XAsyncBegin,
        XAsyncSchedule,
        XAsyncComplete,
        XAsyncGetResult,
        XTaskQueueCreate,
        XTaskQueueCreateComposite,
        XTaskQueueGetPort,
        XTaskQueueDuplicateHandle,
        XTaskQueueDispatch,
        XTaskQueueCloseHandle,
        XTaskQueueSubmitCallback,
        XTaskQueueSubmitDelayedCallback,
        XTaskQueueRegisterWaiter,
        XTaskQueueUnregisterWaiter,
        XTaskQueueTerminate,
        XTaskQueueRegisterMonitor,
        XTaskQueueUnregisterMonitor,
        XTaskQueueGetCurrentProcessTaskQueue,
        XTaskQueueSetCurrentProcessTaskQueue,
        XThreadSetTimeSensitive,
        XThreadVerifyNotTimeSensitive,
        XThreadAssertNotTimeSensitive,
        XThreadIsTimeSensitive,
    ]
}

// ---------------------------------------------------------------------------------------
// functions only the thunk path has
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn nothing(_a: usize) {}

unsafe extern "system" fn no(_a: usize) -> bool {
    false
}

unsafe extern "system" fn yes() -> bool {
    true
}

unsafe extern "system" fn ok() -> HRESULT {
    S_OK
}

const E_GAMEUSER_DEFERRAL_NOT_AVAILABLE: HRESULT = 0x8924_5103_u32 as i32;
const E_GAMEUSER_USER_NOT_FOUND: HRESULT = 0x8924_5104_u32 as i32;

static NEXT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Register*(queue, context, callback, token): a token, nothing will be called
unsafe extern "system" fn register_token4(_q: usize, _ctx: usize, _cb: usize, token: *mut u64) -> HRESULT {
    if token.is_null() {
        return E_INVALIDARG;
    }
    *token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    S_OK
}

static EMPTY_ID: u8 = 0;

/// an async call that completes at once with no result
unsafe fn completed(b: *mut XAsyncBlock) -> HRESULT {
    begin_with(b, &EMPTY_ID as *const u8 as usize, || Ok(Vec::new()))
}

/// the Result of such a call: its status
unsafe extern "system" fn empty_result(b: *mut XAsyncBlock) -> HRESULT {
    XAsyncGetStatus(b, false)
}

unsafe extern "system" fn resolve_privilege_async(_user: usize, _options: u32, _privilege: u32, b: *mut XAsyncBlock) -> HRESULT {
    completed(b)
}

unsafe extern "system" fn resolve_issue_async(_user: usize, _url: usize, b: *mut XAsyncBlock) -> HRESULT {
    completed(b)
}

unsafe extern "system" fn sign_out_async(_user: usize, b: *mut XAsyncBlock) -> HRESULT {
    completed(b)
}

unsafe extern "system" fn prompt_space_async(_bytes: u64, b: *mut XAsyncBlock) -> HRESULT {
    completed(b)
}

unsafe extern "system" fn sign_out_deferral(out: *mut usize) -> HRESULT {
    if !out.is_null() {
        *out = 0;
    }
    E_GAMEUSER_DEFERRAL_NOT_AVAILABLE
}

/// XUserAddByIdWithUiAsync: the local user by either of its ids (XUserAddByIdWithUiResult =
/// XUserAddResult)
unsafe extern "system" fn add_by_id_async(id: u64, b: *mut XAsyncBlock) -> HRESULT {
    let mut h = std::ptr::null_mut();
    if find_user(id, &mut h) < 0 {
        return E_GAMEUSER_USER_NOT_FOUND;
    }
    XUserCloseHandle(h);
    XUserAddAsync(0, b)
}

/// Opening links on the title's behalf would act on the PC: refused.
unsafe extern "system" fn launch_uri(_user: usize, uri: *const u8) -> HRESULT {
    crate::xlog_warn!("XLaunchUri({}) – refused", crate::util::from_cstr_ptr(uri).unwrap_or_default());
    E_NOTIMPL
}

/// nothing to defer on a PC
unsafe extern "system" fn display_deferral(out: *mut usize) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = 1;
    S_OK
}

static PICTURE_ID: u8 = 0;

/// XUserGetGamerPictureAsync(user, size, async): a plain picture (PNG) of the requested size
/// (XUserGamerPictureSize: 64, 208, 424, 1080 pixels) – there is no profile offline.
unsafe extern "system" fn gamer_picture_async(user: *mut c_void, size: u32, b: *mut XAsyncBlock) -> HRESULT {
    let side = match size {
        0 => 64,
        1 => 208,
        2 => 424,
        3 => 1080,
        _ => return E_INVALIDARG,
    };
    if !super::valid_user(user) {
        return E_INVALIDARG;
    }
    begin_with(b, &PICTURE_ID as *const u8 as usize, move || Ok(png_square(side, [0x5A, 0x6E, 0x82])))
}

unsafe extern "system" fn gamer_picture_result(b: *mut XAsyncBlock, size: usize, buf: *mut c_void, used: *mut usize) -> HRESULT {
    XAsyncGetResult(b, &PICTURE_ID as *const u8 as *const c_void, size, buf, used)
}

/// A `side` × `side` RGB PNG of one colour.
fn png_square(side: usize, rgb: [u8; 3]) -> Vec<u8> {
    fn crc32(data: &[u8]) -> u32 {
        let mut c = !0u32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |ty: &[u8; 4], data: &[u8]| {
        let mut c = ty.to_vec();
        c.extend_from_slice(data);
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(side as u32).to_be_bytes());
    ihdr.extend_from_slice(&(side as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, no filter method / interlace
    chunk(b"IHDR", &ihdr);
    let mut rows = Vec::with_capacity(side * (1 + 3 * side));
    for _ in 0..side {
        rows.push(0); // filter: none
        for _ in 0..side {
            rows.extend_from_slice(&rgb);
        }
    }
    chunk(b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(&rows, 6));
    chunk(b"IEND", &[]);
    out
}

/// XUserGetDefaultAudioEndpointUtf16(user, kind, count, id, used): Windows' default
/// communications device (kind 0 render, 1 capture) – what a PC title gets for its user.
unsafe extern "system" fn default_audio_endpoint(_user: u64, kind: u32, count: usize, id: *mut u16, used: *mut usize) -> HRESULT {
    if kind > 1 {
        return E_INVALIDARG;
    }
    // on a thread of its own: the title's thread keeps its COM apartment as it is
    let Some(s) = std::thread::spawn(move || unsafe { endpoint_id(kind) }).join().ok().flatten() else {
        return crate::util::E_NOT_FOUND;
    };
    let w: Vec<u16> = s.encode_utf16().chain([0]).collect();
    if !used.is_null() {
        *used = w.len();
    }
    if id.is_null() || count < w.len() {
        return crate::util::E_NOT_SUFFICIENT_BUFFER;
    }
    std::ptr::copy_nonoverlapping(w.as_ptr(), id, w.len());
    S_OK
}

/// The id of the default communications endpoint (0 render, 1 capture), None without one.
unsafe fn endpoint_id(kind: u32) -> Option<String> {
    use windows::Win32::Media::Audio::{eCapture, eCommunications, eRender, IMMDeviceEnumerator, MMDeviceEnumerator};
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED};
    let init = CoInitializeEx(None, COINIT_MULTITHREADED).is_ok();
    let id = (|| {
        let e: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).ok()?;
        let d = e.GetDefaultAudioEndpoint(if kind == 0 { eRender } else { eCapture }, eCommunications).ok()?;
        let p = d.GetId().ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0 as *const c_void));
        s
    })();
    if init {
        CoUninitialize();
    }
    id
}

/// XPackageInstallChunks(id, count, selectors, interval, suppress, queue, monitor): installed
unsafe extern "system" fn install_chunks(id: *const u8, count: u32, sel: *const c_void, interval: u32, _suppress: bool, queue: *mut c_void, out: *mut *const u64) -> HRESULT {
    XPackageCreateInstallationMonitor(id, count, sel, interval, queue, out)
}

static INSTALL_ID: u8 = 0;
static MOUNT_ID: u8 = 0;

unsafe extern "system" fn install_chunks_async(_id: usize, _count: u32, _sel: usize, _interval: u32, _suppress: bool, b: *mut XAsyncBlock) -> HRESULT {
    begin_with(b, &INSTALL_ID as *const u8 as usize, || Ok((&MONITOR as *const u64 as usize).to_le_bytes().to_vec()))
}

unsafe extern "system" fn install_chunks_result(b: *mut XAsyncBlock, out: *mut usize) -> HRESULT {
    XAsyncGetResult(b, &INSTALL_ID as *const u8 as *const c_void, 8, out as *mut c_void, std::ptr::null_mut())
}

unsafe extern "system" fn mount_with_ui_async(id: *const u8, b: *mut XAsyncBlock) -> HRESULT {
    let mut m: *mut c_void = std::ptr::null_mut();
    let hr = XPackageMount(id, &mut m);
    if hr < 0 {
        return hr;
    }
    let m = m as usize;
    begin_with(b, &MOUNT_ID as *const u8 as usize, move || Ok(m.to_le_bytes().to_vec()))
}

unsafe extern "system" fn mount_with_ui_result(b: *mut XAsyncBlock, out: *mut usize) -> HRESULT {
    XAsyncGetResult(b, &MOUNT_ID as *const u8 as *const c_void, 8, out as *mut c_void, std::ptr::null_mut())
}

/// XPackageDetails (XPackage.h)
#[repr(C)]
struct PackageDetails {
    package_identifier: *const u8,
    version: XVersion,
    kind: u32,
    display_name: *const u8,
    description: *const u8,
    publisher: *const u8,
    store_id: *const u8,
    installing: bool,
    index: u32,
    count: u32,
    age_restricted: bool,
    title_id: *const u8,
}

/// XPackageEnumeratePackages(kind, scope, context, callback): the game itself (no DLC).
unsafe extern "system" fn enumerate_packages(kind: u32, _scope: u32, ctx: *mut c_void, cb: Option<unsafe extern "system" fn(*mut c_void, *const PackageDetails) -> bool>) -> HRESULT {
    let Some(cb) = cb else { return E_INVALIDARG };
    if kind != 0 {
        return S_OK;
    }
    let c = |s: String| std::ffi::CString::new(s).unwrap_or_default();
    let id = c(package_identifier());
    let title = c(crate::state::boot().map(|b| b.title.clone()).unwrap_or_default());
    let title_id = c(crate::state::config().title_id.trim_start_matches("0x").to_ascii_uppercase());
    let empty = c(String::new());
    let d = PackageDetails {
        package_identifier: id.as_ptr() as *const u8,
        version: XVersion { major: 1, minor: 0, build: 0, revision: 0 },
        kind: 0,
        display_name: title.as_ptr() as *const u8,
        description: empty.as_ptr() as *const u8,
        publisher: empty.as_ptr() as *const u8,
        store_id: empty.as_ptr() as *const u8,
        installing: false,
        index: 0,
        count: 1,
        age_restricted: false,
        title_id: title_id.as_ptr() as *const u8,
    };
    cb(ctx, &d);
    S_OK
}

/// Match the OS user locale against the package's declared `MicrosoftGame.Config` `<Resources>`
/// languages: exact match (case-insensitive, `-`/`_` normalized), then primary-language subtag
/// match, then the first declared language; falls back to the raw OS locale only when none are
/// declared.
fn select_user_locale(os_locale: &str, declared: &[String]) -> String {
    if declared.is_empty() {
        return os_locale.to_string();
    }
    let norm_eq = |a: &str, b: &str| {
        a.len() == b.len()
            && a.bytes().zip(b.bytes()).all(|(x, y)| {
                let nx = if x == b'_' { b'-' } else { x.to_ascii_lowercase() };
                let ny = if y == b'_' { b'-' } else { y.to_ascii_lowercase() };
                nx == ny
            })
    };
    if let Some(exact) = declared.iter().find(|l| norm_eq(l, os_locale)) {
        return exact.clone();
    }
    fn primary(s: &str) -> &str {
        s.split(['-', '_']).next().unwrap_or("")
    }
    let os_primary = primary(os_locale);
    if !os_primary.is_empty() {
        if let Some(lang) = declared.iter().find(|l| primary(l).eq_ignore_ascii_case(os_primary)) {
            return lang.clone();
        }
    }
    declared[0].clone()
}

fn package_languages() -> &'static [String] {
    static LANGUAGES: OnceLock<Vec<String>> = OnceLock::new();
    LANGUAGES
        .get_or_init(|| {
            crate::state::boot()
                .and_then(|b| crate::manifest::load(&b.game_root))
                .map(|m| m.languages)
                .unwrap_or_default()
        })
        .as_slice()
}

unsafe extern "system" fn XPackageGetUserLocale(size: usize, buf: *mut u8) -> HRESULT {
    let mut w = [0u16; 85];
    let n = windows_sys::Win32::Globalization::GetUserDefaultLocaleName(w.as_mut_ptr(), w.len() as i32);
    let os_loc = if n > 1 { String::from_utf16_lossy(&w[..n as usize - 1]) } else { "en-US".into() };
    let loc = select_user_locale(&os_loc, package_languages());
    crate::util::write_cstr_out(&loc, buf, size, std::ptr::null_mut())
}

/// every chunk is installed (XPackageChunkAvailability::Ready = 0)
unsafe extern "system" fn XPackageFindChunkAvailability(_id: usize, _count: u32, _sel: usize, out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = 0;
    S_OK
}

unsafe extern "system" fn XPackageEstimateDownloadSize(_id: usize, _count: u32, _sel: usize, size: *mut u64, confirm: *mut bool) -> HRESULT {
    if !size.is_null() {
        *size = 0;
    }
    if !confirm.is_null() {
        *confirm = false;
    }
    S_OK
}

/// XPackageWriteStats { interval, budget, elapsed, bytesWritten }: nothing measured
unsafe extern "system" fn XPackageGetWriteStats(out: *mut [u64; 4]) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = [0; 4];
    S_OK
}

/// XPackageKind::Game
unsafe extern "system" fn XPackageGetPackageKind(_id: usize, out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = 0;
    S_OK
}

/// A stable id per console and title.
unsafe extern "system" fn XSystemGetAppSpecificDeviceId(size: usize, buf: *mut u8, used: *mut usize) -> HRESULT {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (crate::state::config().console_id.as_str(), crate::state::config().title_id.as_str()).hash(&mut h);
    crate::util::write_cstr_out(&format!("{:016X}", h.finish()), buf, size, used)
}

/// XUserFindUserByLocalId / XUserFindUserById (slots 10 / 12): the local user by either id.
unsafe extern "system" fn find_user(id: u64, out: *mut *mut c_void) -> HRESULT {
    let hr = XUserFindUserByLocalId(id, out);
    if hr >= 0 {
        return hr;
    }
    XUserFindUserById(id, out)
}

/// The local user has every privilege (no Xbox Live to deny one).
unsafe extern "system" fn XUserCheckPrivilege(_user: *mut c_void, _options: u32, _privilege: u32, has: *mut bool, reasons: *mut u32) -> HRESULT {
    if has.is_null() {
        return E_INVALIDARG;
    }
    *has = true;
    if !reasons.is_null() {
        *reasons = 0;
    }
    S_OK
}

unsafe extern "system" fn XGameSaveFilesGetRemainingQuota(_user: *mut c_void, _config: *const u8, quota: *mut i64) -> HRESULT {
    if quota.is_null() {
        return E_INVALIDARG;
    }
    *quota = 256 << 20;
    S_OK
}

/// The installation monitor of a fully installed package (XCL runs from an unpacked folder).
static MONITOR: u64 = 0x4D4F_4E49;

unsafe extern "system" fn XPackageCreateInstallationMonitor(_id: *const u8, _count: u32, _sel: *const c_void, _interval: u32, _queue: *mut c_void, out: *mut *const u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = &MONITOR;
    S_OK
}

/// XPackageInstallationProgress { totalBytes, installedBytes, launchBytes, launchable,
/// completed }: everything installed.
unsafe extern "system" fn XPackageGetInstallationProgress(_monitor: *const u64, p: *mut u8) {
    if p.is_null() {
        return;
    }
    for k in 0..3 {
        std::ptr::write_unaligned(p.add(8 * k) as *mut u64, 1);
    }
    *p.add(24) = 1;
    *p.add(25) = 1;
}

unsafe extern "system" fn XPackageRegisterInstallationProgressChanged(_monitor: *const u64, _ctx: *mut c_void, _cb: usize, token: *mut u64) -> HRESULT {
    if !token.is_null() {
        *token = 1;
    }
    S_OK
}

unsafe extern "system" fn XPackageUnregisterInstallationProgressChanged(_monitor: *const u64, _token: u64, _wait: bool) -> bool {
    true
}

/// `<TitleId>` of the game's MicrosoftGame.Config (xcl.toml `title_id` overrides).
pub unsafe extern "system" fn XGameGetXboxTitleId(out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let t = &crate::state::config().title_id;
    *out = u32::from_str_radix(t.trim_start_matches("0x"), 16).unwrap_or(0);
    if *out == 0 {
        crate::xlog_warn!("XGameGetXboxTitleId: no TitleId in MicrosoftGame.Config / xcl.toml → 0");
    }
    S_OK
}

unsafe extern "system" fn XLaunchNewGame(exe: *const u8, args: *const u8, _user: *mut c_void) {
    crate::xlog_warn!(
        "XLaunchNewGame({}, {}) – ignored (XCL runs a single title)",
        crate::util::from_cstr_ptr(exe).unwrap_or_default(),
        crate::util::from_cstr_ptr(args).unwrap_or_default()
    );
}

unsafe extern "system" fn XLaunchRestartOnCrash(_args: *const u8, _reserved: u32) -> HRESULT {
    S_OK
}

// ---------------------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------------------

static SEEN: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn note(c: Guid, i: Guid, what: String) {
    let key = format!("{c}/{i}");
    let mut g = SEEN.lock().unwrap();
    if !g.contains_key(&key) {
        crate::xlog_info!("XGameRuntime: QueryApiImpl {key} → {what}");
        g.insert(key, what);
    }
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiStats {
    /// "clsid/iid" → who answered (XCL, PC, unavailable, missing)
    pub apis: BTreeMap<String, String>,
    /// XCL objects' slots the title called that XCL does not implement: "Iface#slot" → calls
    pub unimplemented: BTreeMap<String, u64>,
}

pub fn stats() -> ApiStats {
    ApiStats { apis: SEEN.lock().unwrap().clone(), unimplemented: UNIMPLEMENTED.lock().unwrap().clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    unsafe fn query(c: Guid, i: Guid) -> (HRESULT, *mut c_void) {
        let mut o = std::ptr::null_mut();
        (QueryApiImpl(&c, &i, &mut o), o)
    }

    unsafe fn slot<T>(o: *mut c_void, k: usize) -> T {
        std::mem::transmute_copy(&*(*(o as *const *const usize)).add(k))
    }

    static DONE: AtomicU32 = AtomicU32::new(0);
    unsafe extern "system" fn done(_b: *mut XAsyncBlock) {
        DONE.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn guids_format_like_the_registry() {
        assert_eq!(XUSER.to_string(), "01acd177-91f9-4763-a38e-ccbb55ce32e0");
        let b: [u8; 16] = unsafe { std::mem::transmute(XUSER) };
        assert_eq!(&b[..4], &[0x77, 0xd1, 0xac, 0x01]);
    }

    fn slot_in_table(name: &str) -> (Guid, Guid, usize) {
        table().find(|r| r.0 == name).map(|r| (r.1, r.2, r.3)).unwrap_or_else(|| panic!("{name} not in the GDK table"))
    }

    /// XCL's objects follow the GDK table: every implemented function is in its class; one
    /// vtable for the interface versions that share slots, its own for the gamertag one; an
    /// interface id the table does not know gets the main vtable. The table agrees with the
    /// slots the thunks in the game's DLLs use (argument / result widths, see HANDOFF).
    #[test]
    fn vtables_follow_the_gdk_table() {
        for c in classes() {
            for (f, _) in c.impls.iter().map(|x| (x.0, x.1)).chain(c.forward.iter().map(|f| (*f, 0))) {
                assert_eq!(slot_in_table(f).0, c.clsid, "{f} is not a {} function", c.name);
            }
        }
        // slots the game's own thunks call (libHttpClient.GDK / XCurl), matched by signature
        for (f, slot) in [
            ("XUserAddAsync", 7),
            ("XUserGetState", 14),
            ("XUserCheckPrivilege", 20),
            ("XUserUnregisterForChangeEvent", 34),
            ("XUserIsStoreUser", 42),
            ("XGameSaveCreateContainer", 16),
            ("XGameSaveCloseContainer", 17),
            ("XGameSaveReadBlobData", 20),
            ("XPackageInstallChunksAsync", 16),
            ("XPackageEnumeratePackages", 39),
            ("XPersistentLocalStorageGetPath", 4),
            ("XSystemGetAppSpecificDeviceId", 5),
            ("XGameGetXboxTitleId", 3),
            ("XAsyncBegin", 7),
            ("XTaskQueueCreate", 12),
        ] {
            assert_eq!(slot_in_table(f).2, slot, "{f}");
        }
        let main = own(&XUSER, &XUSER).unwrap();
        assert!(std::ptr::eq(own(&XUSER, &g(0x26f3c674_a2fe_44fa_b6c4_a323bc94ff53)).unwrap(), main), "a later XUser interface: the same vtable");
        assert!(std::ptr::eq(own(&XUSER, &g(0x5131d685_4394_4ee6_8c18_bfb5d4aef1ff)).unwrap(), main));
        let gt = own(&XUSER, &XUSER_GAMERTAG_IID).unwrap();
        assert!(!std::ptr::eq(gt, main), "XUserGetGamertag has its own vtable");
        assert_eq!(gt.targets[3], XUserGetGamertag as usize);
        assert_eq!(main.targets[7], XUserAddAsync as usize);
        let pkg = own(&XPACKAGE, &XPACKAGE_OLD_IID).expect("an XPackage interface id the table does not know");
        assert!(pkg.iface.main);
        assert_eq!(pkg.targets[slot_in_table("XPackageEnumeratePackages").2], enumerate_packages as usize);
        let pls = own(&XPLS, &g(0x41a4e10c_5a7e_41d9_8c37_37bde62a07d6)).expect("persistent local storage is XCL's");
        assert_eq!(pls.targets[4], XPersistentLocalStorageGetPath as usize);
        assert!(own(&XSTORE, &XSTORE).is_none());
        assert!(returns_bool("XUserIsStoreUser") && returns_bool("XPackageUnregisterPackageInstalled") && !returns_bool("XUserGetState"));
    }

    /// The offline gamer picture is a valid PNG: chunk CRCs, size, pixels.
    #[test]
    fn gamer_picture_is_a_png() {
        let p = png_square(5, [1, 2, 3]);
        assert_eq!(&p[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let (mut at, mut chunks) = (8, Vec::new());
        while at < p.len() {
            let n = u32::from_be_bytes(p[at..at + 4].try_into().unwrap()) as usize;
            let body = &p[at + 4..at + 8 + n];
            let crc = u32::from_be_bytes(p[at + 8 + n..at + 12 + n].try_into().unwrap());
            // CRC-32 (zlib's): the same check the chunk carries
            let mut c = !0u32;
            for &b in body {
                c ^= b as u32;
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                }
            }
            assert_eq!(!c, crc);
            chunks.push((body[..4].to_vec(), body[4..].to_vec()));
            at += 12 + n;
        }
        let names: Vec<&[u8]> = chunks.iter().map(|c| c.0.as_slice()).collect();
        assert_eq!(names, [&b"IHDR"[..], b"IDAT", b"IEND"]);
        assert_eq!(&chunks[0].1, &[0, 0, 0, 5, 0, 0, 0, 5, 8, 2, 0, 0, 0]);
        let rows = miniz_oxide::inflate::decompress_to_vec_zlib(&chunks[1].1).unwrap();
        assert_eq!(rows.len(), 5 * 16);
        assert!(rows.chunks(16).all(|r| r[0] == 0 && r[1..].chunks(3).all(|px| px == [1, 2, 3])));
        // the CRC of IEND is the well-known one
        assert_eq!(&p[p.len() - 4..], &[0xAE, 0x42, 0x60, 0x82]);
    }

    /// The default communications endpoint: Windows' (or not found without an audio device);
    /// a short buffer gets the size it needs.
    #[test]
    fn default_audio_endpoint_is_windows_default() {
        unsafe {
            let mut used = 0usize;
            let hr = default_audio_endpoint(0, 0, 0, std::ptr::null_mut(), &mut used);
            if hr == crate::util::E_NOT_FOUND {
                println!("no audio endpoint – skipping");
                return;
            }
            assert_eq!(hr, crate::util::E_NOT_SUFFICIENT_BUFFER);
            let mut w = vec![0u16; used];
            assert_eq!(default_audio_endpoint(0, 0, w.len(), w.as_mut_ptr(), &mut used), S_OK);
            let id = String::from_utf16_lossy(&w[..used - 1]);
            assert!(
                id.starts_with('{') && id.contains("}.{"),
                "a render endpoint id: {id}"
            );
            assert_eq!(default_audio_endpoint(0, 2, 0, std::ptr::null_mut(), &mut used), E_INVALIDARG);
        }
    }

    /// The thunk protocol end to end on the PC runtime: XUser and XGameSave are XCL's objects,
    /// their async calls complete on the PC runtime's task queues; XStore is refused; XAsync
    /// comes from the PC runtime.
    #[test]
    fn thunk_protocol_with_xcl_objects_on_pc_queues() {
    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        super::TOKENS_HERE.with(|t| t.set(Some(true)));
        if pc().is_none() {
            println!("no system xgameruntime.dll – skipping");
            return;
        }
        unsafe {
            assert_eq!(InitializeApiImplEx2(0x6336_0000_2712, 0x65f4_0000_1edd, 2, std::ptr::null_mut()), S_OK);
            assert_eq!(query(XSTORE, XSTORE).0, REGDB_E_CLASSNOTREG);
            let (hr, xa) = query(XASYNC, XASYNC);
            assert!(hr >= 0 && !xa.is_null(), "XAsync from PC: {hr:#x}");
            let get_status: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, bool) -> HRESULT = slot(xa, 3);

            // XUserAddAsync / Result through the vtable, completed on the PC process queue
            let (hr, u) = query(XUSER, g(0x26f3c674_a2fe_44fa_b6c4_a323bc94ff53));
            assert_eq!(hr, S_OK);
            let add: unsafe extern "system" fn(*mut c_void, u32, *mut XAsyncBlock) -> HRESULT = slot(u, 7);
            let result: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut *mut c_void) -> HRESULT = slot(u, 8);
            let mut b: XAsyncBlock = std::mem::zeroed();
            b.callback = done as usize;
            assert_eq!(add(u, 1, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut user = std::ptr::null_mut();
            assert_eq!(result(u, &mut b, &mut user), S_OK);
            assert!(!user.is_null());
            let (_, gt) = query(XUSER, XUSER_GAMERTAG_IID);
            let gamertag: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, *mut u8, *mut usize) -> HRESULT = slot(gt, 3);
            let mut buf = [0u8; 64];
            let mut used = 0usize;
            assert_eq!(gamertag(gt, user, 0, 64, buf.as_mut_ptr(), &mut used), S_OK);
            assert!(used > 1);
            // gamer picture (slots 16-18): an async PNG of the requested size
            let (_, pu) = query(XUSER, XUSER);
            let pic: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut XAsyncBlock) -> HRESULT = slot(pu, 16);
            let pic_size: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut usize) -> HRESULT = slot(pu, 17);
            let pic_result: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, usize, *mut u8, *mut usize) -> HRESULT = slot(pu, 18);
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(pic(pu, user, 1, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut n = 0usize;
            assert_eq!(pic_size(pu, &mut b, &mut n), S_OK);
            let mut png = vec![0u8; n];
            assert_eq!(pic_result(pu, &mut b, n, png.as_mut_ptr(), &mut used), S_OK);
            assert_eq!((used, &png[..8], u32::from_be_bytes(png[16..20].try_into().unwrap())), (n, &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A][..], 208));
            // the in-game text entry goes to the PC runtime (its handles stay there)
            let (_, ui) = query(XGAMEUI, g(0xeaf669df_5542_4590_99a3_8dc061f837cc));
            let open = slot_in_table("XGameUiTextEntryOpen").2;
            assert_eq!(*(*(ui as *const *const usize)).add(open), FORWARDS[open] as usize, "TextEntryOpen is forwarded");
            // XUserGetTokenAndSignatureAsync/ResultSize/Result (slots 23-25) complete on PC queues
            let tok_async: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                u32,
                *const u8,
                *const u8,
                usize,
                *const c_void,
                usize,
                *const c_void,
                *mut XAsyncBlock,
            ) -> HRESULT = slot(u, 23);
            let tok_size: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut usize) -> HRESULT = slot(u, 24);
            let tok_result: unsafe extern "system" fn(
                *mut c_void,
                *mut XAsyncBlock,
                usize,
                *mut c_void,
                *mut *mut XUserGetTokenAndSignatureData,
                *mut usize,
            ) -> HRESULT = slot(u, 25);
            let mut tb: XAsyncBlock = std::mem::zeroed();
            assert_eq!(
                tok_async(u, user, 0, b"GET\0".as_ptr(), b"https://example.invalid/hotfix\0".as_ptr(), 0, std::ptr::null(), 0, std::ptr::null(), &mut tb),
                S_OK
            );
            assert_eq!(get_status(xa, &mut tb, true), S_OK);
            let mut tsz = 0usize;
            assert_eq!(tok_size(u, &mut tb, &mut tsz), S_OK);
            let mut tbuf = vec![0u8; tsz];
            let mut tptr: *mut XUserGetTokenAndSignatureData = std::ptr::null_mut();
            let mut tused = 0usize;
            assert_eq!(tok_result(u, &mut tb, tbuf.len(), tbuf.as_mut_ptr() as *mut c_void, &mut tptr, &mut tused), S_OK);
            assert_eq!(tused, tsz);
            assert_eq!(tptr as *const u8, tbuf.as_ptr());
            assert!(crate::util::from_cstr_ptr((*tptr).token).unwrap().starts_with("XBL3.0 "));
            assert!(!crate::util::from_cstr_ptr((*tptr).signature).unwrap().is_empty());

            // an unimplemented slot answers E_NOTIMPL and is reported
            let ach: unsafe extern "system" fn(*mut c_void) -> HRESULT = slot(ui, slot_in_table("XGameUiShowAchievementsAsync").2);
            assert_eq!(ach(ui), E_NOTIMPL);
            assert!(stats().unimplemented.contains_key("XGameUiShowAchievementsAsync"), "{:?}", stats().unimplemented);

            // XGameSave: provider (async), container, write a blob, read it back (async)
            let (_, s) = query(XGAMESAVE, g(0x1bfff3af_f14a_40a3_8e35_9ada906593f9));
            let init: unsafe extern "system" fn(*mut c_void, *mut c_void, *const u8, bool, *mut XAsyncBlock) -> HRESULT = slot(s, 4);
            let init_r: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut *mut Provider) -> HRESULT = slot(s, 5);
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(init(s, user, c"xcl-thunk-test".as_ptr() as *const u8, false, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut p = std::ptr::null_mut();
            assert_eq!(init_r(s, &mut b, &mut p), S_OK);
            let create: unsafe extern "system" fn(*mut c_void, *mut Provider, *const u8, *mut *mut Container) -> HRESULT = slot(s, 16);
            let mut c = std::ptr::null_mut();
            assert_eq!(create(s, p, c"slot".as_ptr() as *const u8, &mut c), S_OK);
            let mut upd = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, c"t".as_ptr() as *const u8, &mut upd), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(upd, c"b".as_ptr() as *const u8, b"thunk".as_ptr(), 5), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(upd), S_OK);
            XGameSaveCloseUpdate(upd);
            let read: unsafe extern "system" fn(*mut c_void, *mut Container, *const *const u8, u32, *mut XAsyncBlock) -> HRESULT = slot(s, 21);
            let read_r: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, usize, *mut Blob, *mut u32) -> HRESULT = slot(s, 22);
            let names = [c"b".as_ptr() as *const u8];
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(read(s, c, names.as_ptr(), 1, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut size = 0usize;
            assert_eq!(XAsyncGetResultSize(&mut b, &mut size), S_OK);
            let mut out = vec![0u64; size / 8 + 1];
            let mut n = 0u32;
            assert_eq!(read_r(s, &mut b, size, out.as_mut_ptr() as *mut Blob, &mut n), S_OK);
            assert_eq!(n, 1);
            XGameSaveCloseContainer(c);
            XGameSaveCloseProvider(p);

            // persistent local storage is XCL's (the PC runtime fails it for an unpackaged process)
            let (hr, pls) = query(XPLS, g(0x41a4e10c_5a7e_41d9_8c37_37bde62a07d6));
            assert_eq!(hr, S_OK);
            let size: unsafe extern "system" fn(*mut c_void, *mut usize) -> HRESULT = slot(pls, 3);
            let mut n = 0usize;
            assert_eq!(size(pls, &mut n), S_OK);
            assert!(n > 1);
            // completion callbacks run on the PC queue's threads, after the status
            for _ in 0..200 {
                if DONE.load(Ordering::SeqCst) >= 1 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            super::super::xasync::use_pc(0);
            assert!(DONE.load(Ordering::SeqCst) >= 1, "XAsync completion callback");
        }
    }

    #[test]
    fn legacy_console_iids_map_to_their_own_slots() {
        // SLOTS must cover the full GDK table (highest slot is 83).
        let max_slot = table().map(|r| r.3).max().unwrap_or(0);
        assert!(max_slot < SLOTS, "SLOTS={SLOTS} truncates slot {max_slot}");

        // A slot is compared with the function the main vtable binds to the same name, not with
        // `f as usize`: a function has no single address (`std::ptr::fn_addr_eq`), and the
        // windows-gnu build has two copies of `register_token4`, one in each codegen unit.
        let bound = |o: &Obj, slot: usize, name: &str| {
            assert_eq!(o.iface.names[slot], Some(name), "{} slot {slot}", o.iface.name);
            assert_ne!(o.targets[slot], 0, "{} slot {slot} ({name}) has no function", o.iface.name);
            o.targets[slot]
        };
        let pm = own(&XPACKAGE, &XPACKAGE).expect("XPackage main");
        let main_fn = |name: &str| bound(pm, slot_in_table(name).2, name);

        // XPackage v1 (be3f3e9f): XPackageEnumeratePackages is at slot 28.
        let p1 = own(&XPACKAGE, &XPACKAGE_V1_IID).expect("XPackage v1");
        assert!(!p1.iface.main);
        assert_eq!(bound(p1, 28, "XPackageEnumeratePackages"), main_fn("XPackageEnumeratePackages"));

        // XPackage v2 (f92d8712): EnumeratePackages @ 35, RegisterPackageInstalled @ 36.
        let p2 = own(&XPACKAGE, &XPACKAGE_V2_IID).expect("XPackage v2");
        assert!(!p2.iface.main);
        assert_eq!(bound(p2, 35, "XPackageEnumeratePackages"), main_fn("XPackageEnumeratePackages"));
        assert_eq!(bound(p2, 36, "XPackageRegisterPackageInstalled"), main_fn("XPackageRegisterPackageInstalled"));

        // ... and through the vtable, as a title calls it: v2's slot 36 hands out a new token
        // each time (register_token4); v1 has nothing at slot 36.
        let register = |o: &Obj, token: *mut u64| unsafe {
            slot::<Fwd>(o as *const Obj as *mut c_void, 36)(o, 0, 0, 0, token as usize, 0, 0, 0, 0, 0) as u32 as HRESULT
        };
        let (mut t1, mut t2) = (0u64, 0u64);
        assert_eq!(register(p2, &mut t1), S_OK);
        assert_eq!(register(p2, &mut t2), S_OK);
        assert!(t1 != 0 && t2 > t1, "tokens {t1}, {t2}");
        assert_eq!(register(p2, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(register(p1, &mut t1), E_NOTIMPL);

        // XPackage main: XPackageGetIdentifier at slot 27.
        bound(pm, 27, "XPackageGetIdentifier");

        // XSystem v3 (6fd71f09): XSystemGetAppSpecificDeviceId is at slot 4 (not SandboxId).
        let s3 = own(&XSYSTEM, &XSYSTEM_V3_IID).expect("XSystem v3");
        assert!(!s3.iface.main);
        let sm = own(&XSYSTEM, &XSYSTEM).expect("XSystem main");
        bound(sm, 4, "XSystemGetXboxLiveSandboxId");
        assert_eq!(bound(s3, 4, "XSystemGetAppSpecificDeviceId"), bound(sm, 5, "XSystemGetAppSpecificDeviceId"));

        // Compatible alias IIDs resolve cleanly to their main vtables.
        assert!(own(&XUSER, &XUSER_V2_IID).unwrap().iface.iids.contains(&XUSER_V2_IID));
        assert!(own(&XSYSTEM, &XSYSTEM_V2_IID).unwrap().iface.iids.contains(&XSYSTEM_V2_IID));
        assert!(own(&XGAME, &XGAME_V1_IID).unwrap().iface.iids.contains(&XGAME_V1_IID));
        assert!(own(&XPLS, &XPLS_V1_IID).unwrap().iface.iids.contains(&XPLS_V1_IID));
    }

    /// A device association callback hears once that the gamepad belongs to the signed-in
    /// user (none → the local user), on its queue's completion port – whether it was
    /// registered before or after the user reached the title (Minecraft marks its controller
    /// connected only from this callback).
    #[test]
    fn device_association_reaches_the_title() {
        use crate::hle::xgameruntime as rt;
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        static GOT: Mutex<Vec<rt::DeviceAssociationChange>> = Mutex::new(Vec::new());
        unsafe extern "system" fn heard(ctx: *mut c_void, change: *const rt::DeviceAssociationChange) {
            assert_eq!(ctx as usize, 7);
            GOT.lock().unwrap().push(*change);
        }
        unsafe {
            let mut q = std::ptr::null_mut();
            assert_eq!(rt::taskqueue::XTaskQueueCreate(0, 0, &mut q), S_OK);
            let mut token = 0u64;
            assert_eq!(rt::XUserRegisterForDeviceAssociationChanged(q, 7 as *mut c_void, heard as usize, &mut token), S_OK);
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XUserAddAsync(1, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut user = std::ptr::null_mut();
            assert_eq!(XUserAddResult(&mut b, &mut user), S_OK);
            XUserCloseHandle(user);
            while rt::taskqueue::XTaskQueueDispatch(q, rt::taskqueue::PORT_COMPLETION, 0) {}
            let want = rt::DeviceAssociationChange { device_id: crate::hle::gameinput::gamepad_device_id(), old_user: 0, new_user: 1 };
            assert_eq!(*GOT.lock().unwrap(), vec![want]);
            assert!(rt::XUserUnregisterForDeviceAssociationChanged(token, true));
            rt::taskqueue::XTaskQueueCloseHandle(q);
        }
    }

    #[test]
    /// Without `xbox_live_tokens` (the default) a token request fails as on a console that
    /// cannot reach Xbox Live: the async completes with 0x80072EFD and has no result.
    fn token_requests_fail_offline_by_default() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        super::TOKENS_HERE.with(|t| t.set(None));
        unsafe {
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XUserAddAsync(1, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut user = std::ptr::null_mut();
            assert_eq!(XUserAddResult(&mut b, &mut user), S_OK);
            let mut bt: XAsyncBlock = std::mem::zeroed();
            let (method, url) = (b"GET\0", b"https://example.playfabapi.com/\0");
            assert_eq!(super::XUserGetTokenAndSignatureAsync(user, 0, method.as_ptr(), url.as_ptr(), 0, std::ptr::null(), 0, std::ptr::null(), &mut bt), S_OK);
            assert_eq!(XAsyncGetStatus(&mut bt, true), 0x8007_2EFD_u32 as i32);
            let mut n = 0usize;
            assert_ne!(super::XUserGetTokenAndSignatureResultSize(&mut bt, &mut n), S_OK);
            XUserCloseHandle(user);
        }
    }

    #[test]
    fn token_and_signature_utf8_and_utf16_through_vtable() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        super::TOKENS_HERE.with(|t| t.set(Some(true)));
        unsafe {
            let (hr, u) = query(XUSER, XUSER);
            assert_eq!(hr, S_OK);
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XUserAddAsync(1, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut user = std::ptr::null_mut();
            assert_eq!(XUserAddResult(&mut b, &mut user), S_OK);

            // UTF-8 slots 23..=25
            let tok_async: unsafe extern "system" fn(
                *mut c_void, *mut c_void, u32, *const u8, *const u8, usize, *const c_void, usize, *const c_void, *mut XAsyncBlock,
            ) -> HRESULT = slot(u, 23);
            let tok_size: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut usize) -> HRESULT = slot(u, 24);
            let tok_res: unsafe extern "system" fn(
                *mut c_void, *mut XAsyncBlock, usize, *mut c_void, *mut *mut XUserGetTokenAndSignatureData, *mut usize,
            ) -> HRESULT = slot(u, 25);
            let mut b8: XAsyncBlock = std::mem::zeroed();
            let body = b"{\"hello\":\"world\"}";
            assert_eq!(
                tok_async(u, user, 0, b"POST\0".as_ptr(), b"https://userstats.xboxlive.com/batch\0".as_ptr(), 0, std::ptr::null(), body.len(), body.as_ptr() as *const c_void, &mut b8),
                S_OK
            );
            assert_eq!(XAsyncGetStatus(&mut b8, true), S_OK);
            let mut n8 = 0usize;
            assert_eq!(tok_size(u, &mut b8, &mut n8), S_OK);
            assert!(n8 > 32);
            let mut buf8 = vec![0u8; n8];
            let mut data8: *mut XUserGetTokenAndSignatureData = std::ptr::null_mut();
            let mut used8 = 0usize;
            assert_eq!(tok_res(u, &mut b8, buf8.len(), buf8.as_mut_ptr() as *mut c_void, &mut data8, &mut used8), S_OK);
            assert_eq!(used8, n8);
            let tok_s = crate::util::from_cstr_ptr((*data8).token).unwrap();
            let sig_s = crate::util::from_cstr_ptr((*data8).signature).unwrap();
            assert!(tok_s.starts_with("XBL3.0 x=xcl;"));
            assert_eq!((*data8).token_size, tok_s.len() + 1);
            assert_eq!((*data8).signature_size, sig_s.len() + 1);
            assert!(sig_s.len() >= 100, "base64 of 76-byte XSTS header: {sig_s}");

            // UTF-16 slots 26..=28
            let tok16_async: unsafe extern "system" fn(
                *mut c_void, *mut c_void, u32, *const u16, *const u16, usize, *const c_void, usize, *const c_void, *mut XAsyncBlock,
            ) -> HRESULT = slot(u, 26);
            let tok16_size: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut usize) -> HRESULT = slot(u, 27);
            let tok16_res: unsafe extern "system" fn(
                *mut c_void, *mut XAsyncBlock, usize, *mut c_void, *mut *mut XUserGetTokenAndSignatureUtf16Data, *mut usize,
            ) -> HRESULT = slot(u, 28);
            let method16 = crate::util::wide("GET");
            let url16 = crate::util::wide("https://social.xboxlive.com/users/me/people");
            let mut b16: XAsyncBlock = std::mem::zeroed();
            assert_eq!(
                tok16_async(u, user, 0, method16.as_ptr(), url16.as_ptr(), 0, std::ptr::null(), 0, std::ptr::null(), &mut b16),
                S_OK
            );
            assert_eq!(XAsyncGetStatus(&mut b16, true), S_OK);
            let mut n16 = 0usize;
            assert_eq!(tok16_size(u, &mut b16, &mut n16), S_OK);
            let mut buf16 = vec![0u8; n16];
            let mut data16: *mut XUserGetTokenAndSignatureUtf16Data = std::ptr::null_mut();
            let mut used16 = 0usize;
            assert_eq!(tok16_res(u, &mut b16, buf16.len(), buf16.as_mut_ptr() as *mut c_void, &mut data16, &mut used16), S_OK);
            assert_eq!(used16, n16);
            let tok16 = crate::util::from_wide_ptr((*data16).token).unwrap();
            let sig16 = crate::util::from_wide_ptr((*data16).signature).unwrap();
            assert!(tok16.starts_with("XBL3.0 x=xcl;"));
            assert_eq!((*data16).token_count, tok16.encode_utf16().count() + 1);
            assert_eq!((*data16).signature_count, sig16.encode_utf16().count() + 1);

            XUserCloseHandle(user);
        }
    }

    #[test]
    fn user_device_association_and_controller_lookup() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let (hr, dev) = query(XUSER_DEVICE, XUSER_DEVICE);
            assert_eq!(hr, S_OK);
            let find_dev: unsafe extern "system" fn(*mut c_void, *const u8, *mut *mut c_void) -> HRESULT = slot(dev, 3);
            let device_id = [0x42u8; 32];
            let mut user = std::ptr::null_mut();
            assert_eq!(find_dev(dev, device_id.as_ptr(), &mut user), S_OK);
            assert!(!user.is_null());

            let find_ctrl: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut XAsyncBlock) -> HRESULT = slot(dev, 9);
            let find_ctrl_res: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut [u8; 32]) -> HRESULT = slot(dev, 10);
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(find_ctrl(dev, user, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut out_dev = [0xFFu8; 32];
            assert_eq!(find_ctrl_res(dev, &mut b, &mut out_dev), S_OK);
            assert_eq!(out_dev, crate::hle::gameinput::gamepad_device_id());
            assert_ne!(out_dev, [0u8; 32]);

            XUserCloseHandle(user);
        }
    }

    #[test]
    fn xpackage_get_identifier_slot_27_uses_three_args() {
        unsafe {
            let (hr, pkg) = query(XPACKAGE, XPACKAGE);
            assert_eq!(hr, S_OK);
            let get_id: unsafe extern "system" fn(*mut c_void, *const u8, usize, *mut u8) -> HRESULT = slot(pkg, 27);
            let want = package_identifier();
            let mut buf = [0u8; 128];
            assert_eq!(get_id(pkg, b"G:\\\0".as_ptr(), buf.len(), buf.as_mut_ptr()), S_OK);
            assert_eq!(crate::util::from_cstr_ptr(buf.as_ptr()).as_deref(), Some(want.as_str()));

            // Null path_or_name also returns the current package identifier.
            buf.fill(0);
            assert_eq!(get_id(pkg, std::ptr::null(), buf.len(), buf.as_mut_ptr()), S_OK);
            assert_eq!(crate::util::from_cstr_ptr(buf.as_ptr()).as_deref(), Some(want.as_str()));

            // Short or null buffer returns E_NOT_SUFFICIENT_BUFFER without dereferencing `size` as a pointer.
            assert_eq!(get_id(pkg, b"G:\\\0".as_ptr(), 4, buf.as_mut_ptr()), crate::util::E_NOT_SUFFICIENT_BUFFER);
            assert_eq!(get_id(pkg, b"G:\\\0".as_ptr(), 64, std::ptr::null_mut()), crate::util::E_NOT_SUFFICIENT_BUFFER);
        }
    }

    #[test]
    fn user_locale_selection_matches_manifest_resources() {
        let declared: Vec<String> = ["en-US", "fr-FR", "zh-Hans"].into_iter().map(String::from).collect();
        // 1) Case-insensitive exact match (preserving manifest spelling)
        assert_eq!(select_user_locale("fr-FR", &declared), "fr-FR");
        assert_eq!(select_user_locale("fr-fr", &declared), "fr-FR");
        assert_eq!(select_user_locale("fr_FR", &declared), "fr-FR");
        // 2) Case-insensitive primary-language subtag match
        assert_eq!(select_user_locale("fr-CA", &declared), "fr-FR");
        assert_eq!(select_user_locale("FR_be", &declared), "fr-FR");
        assert_eq!(select_user_locale("zh-CN", &declared), "zh-Hans");
        // 3) First declared language fallback
        assert_eq!(select_user_locale("hr-HR", &declared), "en-US");
        assert_eq!(select_user_locale("", &declared), "en-US");
        // 4) Raw OS locale when manifest declares no languages
        assert_eq!(select_user_locale("hr-HR", &[]), "hr-HR");
    }

    #[test]
    fn xnetworking_and_xerror_through_vtable() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        static ERR_SEEN: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn on_error(hr: HRESULT, _msg: *const u8, _ctx: *mut c_void) -> bool {
            ERR_SEEN.store(hr as u32, Ordering::SeqCst);
            true
        }
        unsafe {
            // Both XNetworking CLSID and v2 IID resolve to XCL's XNetworking object.
            let (hr, net) = query(XNETWORKING, XNETWORKING);
            assert_eq!(hr, S_OK);
            let (hr2, net2) = query(XNETWORKING, g(0xbf2346b2_39af_4658_b5ea_44713c7e83b3));
            assert_eq!(hr2, S_OK);
            assert_eq!(net, net2);
            let get_port: unsafe extern "system" fn(*mut c_void, *mut u16) -> HRESULT = slot(net, 3);
            let mut port = 0u16;
            assert_eq!(get_port(net, &mut port), S_OK);
            assert_eq!(port, networking::DEFAULT_UDP_PORT);
            let get_hint: unsafe extern "system" fn(*mut c_void, *mut networking::XNetworkingConnectivityHint) -> HRESULT = slot(net, 15);
            let mut hint: networking::XNetworkingConnectivityHint = std::mem::zeroed();
            assert_eq!(get_hint(net, &mut hint), S_OK);
            assert!(hint.network_initialized);

            // XError vtable (slots 3..=5): callback runs and DebugBreak is not triggered.
            let (hr, err) = query(XERROR, XERROR);
            assert_eq!(hr, S_OK);
            let report: unsafe extern "system" fn(*mut c_void, HRESULT, *const u8) = slot(err, 3);
            let set_cb: unsafe extern "system" fn(*mut c_void, usize, *mut c_void) = slot(err, 4);
            let set_opt: unsafe extern "system" fn(*mut c_void, u32, usize) = slot(err, 5);
            set_opt(err, 3, 0);
            set_cb(err, on_error as usize, std::ptr::null_mut());
            report(err, crate::util::E_FAIL, b"unit test error\0".as_ptr());
            assert_eq!(ERR_SEEN.load(Ordering::SeqCst), crate::util::E_FAIL as u32);
            set_cb(err, 0, std::ptr::null_mut());
        }
    }

    #[test]
    fn xerror_report_allows_reentrant_set_callback_and_report() {
        use std::sync::mpsc;
        use std::time::Duration;
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        static TRACE: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        unsafe extern "system" fn second_cb(hr: HRESULT, _msg: *const u8, _ctx: *mut c_void) -> bool {
            TRACE.lock().unwrap().push(hr as u32);
            error_set_callback(0, std::ptr::null_mut());
            true
        }
        unsafe extern "system" fn first_cb(hr: HRESULT, _msg: *const u8, _ctx: *mut c_void) -> bool {
            TRACE.lock().unwrap().push(hr as u32);
            // Re-entrantly swap callback and report a second error while inside XErrorReport.
            error_set_callback(second_cb as usize, std::ptr::null_mut());
            XErrorReport(crate::util::E_INVALIDARG, b"nested error\0".as_ptr());
            true
        }

        TRACE.lock().unwrap().clear();
        let (tx, rx) = mpsc::channel();
        let t = std::thread::spawn(move || unsafe {
            error_set_callback(first_cb as usize, std::ptr::null_mut());
            XErrorReport(crate::util::E_FAIL, b"outer error\0".as_ptr());
            let _ = tx.send(());
        });
        rx.recv_timeout(Duration::from_secs(2))
            .expect("XErrorReport deadlocked on re-entrant XErrorSetCallback / XErrorReport");
        t.join().unwrap();
        assert_eq!(*TRACE.lock().unwrap(), vec![crate::util::E_FAIL as u32, crate::util::E_INVALIDARG as u32]);
        unsafe { error_set_callback(0, std::ptr::null_mut()) };
    }

    #[test]
    fn xcl_xasync_fallback_vtable_covers_composite_queues_and_threads() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let xa = xcl_xasync_object() as *const Obj as *mut c_void;
            let q_create: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> HRESULT = slot(xa, 12);
            let q_comp: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut *mut c_void) -> HRESULT = slot(xa, 13);
            let q_port: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut *mut c_void) -> HRESULT = slot(xa, 14);
            let q_close: unsafe extern "system" fn(*mut c_void, *mut c_void) = slot(xa, 17);
            let set_ts: unsafe extern "system" fn(*mut c_void, bool) -> HRESULT = slot(xa, 27);
            let verify_ts: unsafe extern "system" fn(*mut c_void) -> HRESULT = slot(xa, 28);
            let assert_ts: unsafe extern "system" fn(*mut c_void) = slot(xa, 29);
            let is_ts: unsafe extern "system" fn(*mut c_void) -> bool = slot(xa, 30);

            let mut base = std::ptr::null_mut();
            assert_eq!(q_create(xa, 0, 0, &mut base), S_OK);
            let (mut wp, mut cp) = (std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(q_port(xa, base, 0, &mut wp), S_OK);
            assert_eq!(q_port(xa, base, 1, &mut cp), S_OK);
            let mut comp = std::ptr::null_mut();
            assert_eq!(q_comp(xa, wp, cp, &mut comp), S_OK);
            assert!(!comp.is_null());
            q_close(xa, comp);
            q_close(xa, base);

            assert_eq!(set_ts(xa, true), S_OK);
            assert!(is_ts(xa));
            assert_eq!(verify_ts(xa), S_OK);
            assert_ts(xa);
            assert_eq!(set_ts(xa, false), S_OK);
            assert!(!is_ts(xa));
        }
    }

    #[test]
    fn xcl_xaccessibility_fallback_vtable_covers_captions_and_high_contrast() {
        let _g = accessibility::ACCESSIBILITY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let iid = g(0xd722b373_8c4d_4692_9e51_d6ad9b37aa7d);
        unsafe {
            // Built-in fallback object serves slots 3..=5 and stubs speech slots as E_NOTIMPL.
            let acc = xcl_xaccessibility_object() as *const Obj as *mut c_void;
            let get_props: unsafe extern "system" fn(*mut c_void, *mut accessibility::XClosedCaptionProperties) -> HRESULT = slot(acc, 3);
            let set_enabled: unsafe extern "system" fn(*mut c_void, u8) -> HRESULT = slot(acc, 4);
            let get_hc: unsafe extern "system" fn(*mut c_void, *mut u32) -> HRESULT = slot(acc, 5);
            let speech_pos: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT = slot(acc, 6);

            assert_eq!(set_enabled(acc, 1), S_OK);
            let mut props: accessibility::XClosedCaptionProperties = std::mem::zeroed();
            assert_eq!(get_props(acc, &mut props), S_OK);
            assert_eq!(props.enabled, 1);
            assert_eq!(props.font_scale, 1.0);
            assert_eq!(set_enabled(acc, 0), S_OK);
            assert_eq!(get_props(acc, &mut props), S_OK);
            assert_eq!(props.enabled, 0);

            let mut hc = u32::MAX;
            assert_eq!(get_hc(acc, &mut hc), S_OK);
            assert!(hc <= accessibility::XHIGH_CONTRAST_MODE_OTHER);

            assert_eq!(speech_pos(acc, 0), E_NOTIMPL);

            // QueryApiImpl succeeds whether backed by host System32\xgameruntime.dll or XCL fallback.
            let (hr, queried) = query(XACCESSIBILITY, iid);
            assert_eq!(hr, S_OK);
            assert!(!queried.is_null());
        }
    }
}
