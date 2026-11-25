//! Command lists of console code.
//!
//! The D3D12.X code compiled into a title (the SDK's inline functions) writes command records
//! straight into the list object when the list's buffer has room – +0x18 write pointer, +0x30
//! end, +0x28 the limit for dispatch records – and otherwise calls the list method the record
//! stands for, with the method's own arguments (HANDOFF: "inline writers"; e.g. 0xC000B600 +
//! a hardware topology, else #20 IASetPrimitiveTopology). A PC list object has other data at
//! those offsets. So console code gets a list object of the console's shape whose buffer never
//! has room: every inline write becomes the method call, and each method runs on the
//! (adopted) PC list behind it – with all of XCL's console slot handling.
//!
//! The object: the console fields, the PC list at `F_BACKING`; its vtable is a thunk per slot
//! (`mov rcx, [rcx + F_BACKING]` and on to the PC list's own vtable entry). QueryInterface
//! hands out the console object again, Release frees it with the PC list, and
//! ExecuteCommandLists runs the PC lists (`backing`).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows_sys::core::GUID;

use crate::util::HRESULT;

const LIST_SLOTS: usize = 600;
/// the object: console fields up to `ROOM`, no-room "buffer" at `ROOM` (with slack after it
/// for code that would write before checking), primary PC list at `F_PRIMARY` (identity /
/// refcount), currently active recording PC list at `F_BACKING` (swapped on mid-list sync split).
const OBJECT_BYTES: usize = 0x3000;
const ROOM: usize = 0x1000;
const F_PRIMARY: usize = 0x2FF0;
const F_BACKING: usize = 0x2FF8;
const F_WRITE: usize = 0x18;
const F_DISPATCH_END: usize = 0x28;
const F_END: usize = 0x30;
/// `XCL_LIST_TRACE`: the last `RING` slots called (u16 each) and a counter
const F_RING: usize = 0x2000;
const F_RING_AT: usize = 0x2100;
const RING: usize = 64;
/// Maximum number of mid-list split segments per command list recording.
const MAX_EXTRA_SEGMENTS: usize = 32;

#[derive(Default)]
struct SegmentBook {
    /// Additional PC command lists recorded after `primary` in this recording (in order).
    extra_active: Vec<usize>,
    /// Closed extra PC command lists from earlier recordings, kept for reuse on future splits.
    spares: Vec<usize>,
    /// A PC command list recording in place of `primary`: at the title's Reset the GPU had not
    /// finished primary's last submit, and the PC does not let a list be submitted again before
    /// that (#1452, debug layer id 553). A console does.
    first: Option<usize>,
}

/// Wrapper address -> split segment book (populated lazily only for lists that split).
static SEGMENTS: Mutex<Option<HashMap<usize, SegmentBook>>> = Mutex::new(None);
/// Active recording PC command list address -> `title_list` wrapper address.
static WRAPPER_OF: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

struct Vtable(usize);

/// The console list vtable: QueryInterface / AddRef / Release / Close / Reset of their own,
/// every other slot a thunk to the currently active PC list's entry (`F_BACKING`).
fn vtable() -> usize {
    static V: OnceLock<Vtable> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = vec![0usize; LIST_SLOTS];
        let trace = std::env::var_os("XCL_LIST_TRACE").is_some();
        for (slot, e) in v.iter_mut().enumerate() {
            *e = if trace { forward_traced(slot) } else { forward(slot) };
        }
        v[0] = qi as usize;
        v[1] = add_ref as usize;
        v[2] = release as usize;
        v[9] = close as usize;
        v[10] = reset as usize;
        Vtable(Box::leak(v.into_boxed_slice()).as_ptr() as usize)
    })
    .0
}

/// `mov rcx, [rcx + F_BACKING]; mov rax, [rcx]; jmp [rax + slot * 8]`
fn forward(slot: usize) -> usize {
    let mut c = vec![0x48, 0x8B, 0x89];
    c.extend_from_slice(&(F_BACKING as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8B, 0x01, 0xFF, 0xA0]);
    c.extend_from_slice(&((slot * 8) as u32).to_le_bytes());
    let p = crate::stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
    }
    p as usize
}

/// `forward` after `record(this, slot)` (argument registers, xmm0–3 kept)
fn forward_traced(slot: usize) -> usize {
    let mut c: Vec<u8> = vec![0x51, 0x52, 0x41, 0x50, 0x41, 0x51]; // push rcx, rdx, r8, r9
    c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x68]); // sub rsp, 0x68
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x7F, reg, 0x24, off]); // movdqu [rsp+off], xmmN
    }
    c.push(0xBA); // mov edx, slot
    c.extend_from_slice(&(slot as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, record
    c.extend_from_slice(&(record as usize as u64).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xD0]); // call rax
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x6F, reg, 0x24, off]); // movdqu xmmN, [rsp+off]
    }
    c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x68]); // add rsp, 0x68
    c.extend_from_slice(&[0x41, 0x59, 0x41, 0x58, 0x5A, 0x59]); // pop r9, r8, rdx, rcx
    c.extend_from_slice(&[0x48, 0x8B, 0x89]);
    c.extend_from_slice(&(F_BACKING as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8B, 0x01, 0xFF, 0xA0]);
    c.extend_from_slice(&((slot * 8) as u32).to_le_bytes());
    let p = crate::stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
    }
    p as usize
}

unsafe extern "system" fn record(this: *mut c_void, slot: u32) {
    let at = (this as usize + F_RING_AT) as *mut usize;
    let k = *at % RING;
    *((this as usize + F_RING + 2 * k) as *mut u16) = slot as u16;
    *at += 1;
}

/// #9 Close: closes the active recording segment (`F_BACKING`); a failure is logged with
/// the last methods called (`XCL_LIST_TRACE`). Prior split segments were closed on split.
unsafe extern "system" fn close(this: *mut c_void) -> HRESULT {
    let pc = *((this as usize + F_BACKING) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> HRESULT = std::mem::transmute(pc_slot(pc, 9));
    let hr = f(pc);
    let n = *((this as usize + F_RING_AT) as *const usize);
    if hr < 0 && n > 0 {
        let first = n.saturating_sub(RING);
        let slots: Vec<String> = (first..n).map(|k| (*((this as usize + F_RING + 2 * (k % RING)) as *const u16)).to_string()).collect();
        crate::xlog_warn!("D3D12.X: console list Close → {:#x}; console slots called since Reset (last {}): {}", hr as u32, slots.len(), slots.join(" "));
    }
    // a new recording starts at Reset
    *((this as usize + F_RING_AT) as *mut usize) = 0;
    hr
}

/// #10 Reset(allocator, initial_pso): resets the primary PC list (`F_PRIMARY`) via `list_x::reset_pc`
/// (calling directly rather than through the dual-dispatch vtable thunk so console state always
/// resets even though this wrapper sits in `xcl.exe`), restores `F_BACKING = F_PRIMARY`, and moves
/// any extra split segments into the per-list spare pool. While the GPU has not finished
/// primary's last submit, the new recording goes onto a spare instead (`first`). The newer console
/// layout's Reset (#139) reaches it through `list_x::reset_x` on the active PC list.
pub(super) unsafe extern "system" fn reset(this: *mut c_void, alloc: usize, pso: usize) -> HRESULT {
    let primary = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let adopted = super::real::is_adopted(primary);
    let stand_in = if adopted && super::alloc_pool::in_flight(primary as usize) {
        super::list_x::reset_stand_in(this, primary, alloc, pso)
    } else {
        None
    };
    let hr = if stand_in.is_some() {
        0
    } else if adopted {
        super::list_x::reset_pc(primary, alloc, pso)
    } else {
        let f: unsafe extern "system" fn(*mut c_void, usize, usize) -> HRESULT = std::mem::transmute(pc_slot(primary, 10));
        f(primary, alloc, pso)
    };
    if hr >= 0 {
        let active = stand_in.unwrap_or(primary as usize);
        *((this as usize + F_BACKING) as *mut usize) = active;
        let moved = {
            let mut sg = SEGMENTS.lock().unwrap();
            let m = sg.get_or_insert_with(HashMap::new);
            if stand_in.is_some() || m.contains_key(&(this as usize)) {
                let b = m.entry(this as usize).or_default();
                let mut old = std::mem::take(&mut b.extra_active);
                old.extend(std::mem::replace(&mut b.first, stand_in));
                b.spares.extend(old.iter().copied());
                old
            } else {
                Vec::new()
            }
        };
        if !moved.is_empty() || stand_in.is_some() {
            let mut wg = WRAPPER_OF.lock().unwrap();
            let m = wg.get_or_insert_with(HashMap::new);
            for &seg in moved.iter().chain([primary as usize].iter()) {
                m.remove(&seg);
            }
            m.insert(active, this as usize);
            drop(wg);
            for seg in moved {
                super::bind::reset(seg as *mut c_void, 0, 0);
                super::list_x::clear_segment_state(seg);
            }
        }
        *((this as usize + F_RING_AT) as *mut usize) = 0;
    }
    hr
}

/// The console object for the PC list `pc` (adopted; one reference moves to it).
pub unsafe fn wrap(pc: *mut c_void) -> *mut c_void {
    if pc.is_null() {
        return pc;
    }
    let layout = std::alloc::Layout::from_size_align(OBJECT_BYTES, 64).unwrap();
    let o = std::alloc::alloc_zeroed(layout);
    if o.is_null() {
        return pc;
    }
    *(o as *mut usize) = vtable();
    let room = o as usize + ROOM;
    for f in [F_WRITE, F_DISPATCH_END, F_END] {
        *(o.add(f) as *mut usize) = room;
    }
    *(o.add(F_PRIMARY) as *mut usize) = pc as usize;
    *(o.add(F_BACKING) as *mut usize) = pc as usize;
    WRAPPER_OF.lock().unwrap().get_or_insert_with(HashMap::new).insert(pc as usize, o as usize);
    o as *mut c_void
}

/// The primary PC list behind a console list object (None: not one).
pub fn backing(list: *mut c_void) -> Option<*mut c_void> {
    if list.is_null() || !crate::gpu::shader::readable(list as usize, 8) {
        return None;
    }
    unsafe { (*(list as *const usize) == vtable()).then(|| *((list as usize + F_PRIMARY) as *const *mut c_void)) }
}

/// Expand a console list object into all of its recorded PC command list segments (in exact
/// recording order). Returns `false` if `list` is not a `title_list` wrapper.
pub fn push_segments(list: *mut c_void, out: &mut Vec<*mut c_void>) -> bool {
    let Some(primary) = backing(list) else {
        return false;
    };
    let sg = SEGMENTS.lock().unwrap();
    let book = sg.as_ref().and_then(|m| m.get(&(list as usize)));
    out.push(book.and_then(|b| b.first).map_or(primary, |f| f as *mut c_void));
    if let Some(book) = book {
        for &seg in &book.extra_active {
            out.push(seg as *mut c_void);
        }
    }
    true
}

/// Look up the `title_list` wrapper currently recording into `pc`.
pub(super) fn wrapper_of(pc: *mut c_void) -> Option<*mut c_void> {
    if pc.is_null() {
        return None;
    }
    WRAPPER_OF.lock().unwrap().as_ref()?.get(&(pc as usize)).map(|&w| w as *mut c_void)
}

/// Whether `wrapper` may allocate/activate another split segment for this recording.
pub(super) fn can_split(wrapper: *mut c_void) -> bool {
    let sg = SEGMENTS.lock().unwrap();
    sg.as_ref()
        .and_then(|m| m.get(&(wrapper as usize)))
        .is_none_or(|b| b.extra_active.len() < MAX_EXTRA_SEGMENTS)
}

/// Pop a previously created closed extra PC command list from `wrapper`'s spare pool for reuse
/// (the last one whose submits the GPU has finished).
pub(super) fn pop_spare(wrapper: *mut c_void) -> Option<*mut c_void> {
    let mut sg = SEGMENTS.lock().unwrap();
    let spares = &mut sg.as_mut()?.get_mut(&(wrapper as usize))?.spares;
    let k = spares.iter().rposition(|&p| !super::alloc_pool::in_flight(p))?;
    Some(spares.remove(k) as *mut c_void)
}

/// Give back to `wrapper`'s spare pool a closed PC command list that did not become a segment.
pub(super) fn put_spare(wrapper: *mut c_void, pc: *mut c_void) {
    let mut sg = SEGMENTS.lock().unwrap();
    sg.get_or_insert_with(HashMap::new).entry(wrapper as usize).or_default().spares.push(pc as usize);
}

/// Switch `wrapper` from closed segment `old_pc` to newly opened segment `new_pc`.
pub(super) unsafe fn activate_segment(wrapper: *mut c_void, old_pc: *mut c_void, new_pc: *mut c_void) {
    *((wrapper as usize + F_BACKING) as *mut usize) = new_pc as usize;
    {
        let mut sg = SEGMENTS.lock().unwrap();
        sg.get_or_insert_with(HashMap::new)
            .entry(wrapper as usize)
            .or_default()
            .extra_active
            .push(new_pc as usize);
    }
    let mut wg = WRAPPER_OF.lock().unwrap();
    let m = wg.get_or_insert_with(HashMap::new);
    m.remove(&(old_pc as usize));
    m.insert(new_pc as usize, wrapper as usize);
}

/// For `KickoffX` on an already-split list: returns `(primary, prior_closed_segments, extra_to_reclaim)`
/// when `active_pc != primary`, switches `F_BACKING` back to `primary`, and moves all extra
/// segments (including `active_pc`) into `spares` so recording resumes cleanly on `primary`.
/// (`primary` here is the recording's first segment: `first` when one stands in.)
pub(super) unsafe fn reclaim_for_kickoff(active_pc: *mut c_void) -> Option<(*mut c_void, Vec<*mut c_void>, Vec<usize>)> {
    let wrapper = wrapper_of(active_pc)?;
    let mut sg = SEGMENTS.lock().unwrap();
    let book = sg.as_mut().and_then(|m| m.get_mut(&(wrapper as usize)))?;
    let primary = book.first.map_or_else(|| *((wrapper as usize + F_PRIMARY) as *const *mut c_void), |f| f as *mut c_void);
    if primary == active_pc {
        return None;
    }
    let (prior, reclaimed) = {
        let extra = std::mem::take(&mut book.extra_active);
        let mut prior = Vec::with_capacity(extra.len());
        prior.push(primary);
        for &seg in &extra {
            if seg != active_pc as usize {
                prior.push(seg as *mut c_void);
            }
        }
        book.spares.extend(extra.iter().copied());
        (prior, extra)
    };
    drop(sg);
    *((wrapper as usize + F_BACKING) as *mut usize) = primary as usize;
    {
        let mut wg = WRAPPER_OF.lock().unwrap();
        if let Some(m) = wg.as_mut() {
            for &seg in &reclaimed {
                m.remove(&seg);
            }
            m.insert(primary as usize, wrapper as usize);
        }
    }
    Some((primary, prior, reclaimed))
}

unsafe fn pc_slot(pc: *mut c_void, slot: usize) -> usize {
    *(*(pc as *const *const usize)).add(slot)
}

/// QueryInterface: the primary PC list answers; where it hands out itself, the console object goes out.
/// Command list interfaces of the console's headers that the PC does not know: the console's
/// ID3D12GraphicsCommandList5 / 6 (the PC's data1–data3, another tail – titles ask for them
/// and use the answer without a check). The console object has the console's whole vtable,
/// so it is the answer.
const CONSOLE_LIST_IIDS: [(u32, u16, u16, [u8; 8]); 2] = [
    (0x5505_0859, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
    (0xc382_7890, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
];

unsafe extern "system" fn qi(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let pc = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(pc_slot(pc, 0));
    if !riid.is_null() && !ppv.is_null() {
        let g = &*riid;
        if CONSOLE_LIST_IIDS.contains(&(g.data1, g.data2, g.data3, g.data4)) {
            let add_ref_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 1));
            add_ref_fn(pc);
            *ppv = this;
            return 0;
        }
    }
    let hr = f(pc, riid, ppv);
    if hr >= 0 && !ppv.is_null() && *ppv == pc {
        *ppv = this;
    }
    hr
}

/// AddRef: always references the primary PC list (`F_PRIMARY`).
unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    let pc = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 1));
    f(pc)
}

/// Release: the console object (and any extra split segment PC lists) goes with the primary PC
/// list's last reference.
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    let pc = *((this as usize + F_PRIMARY) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 2));
    let n = f(pc);
    if n == 0 {
        let book = SEGMENTS.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize)));
        {
            let mut wg = WRAPPER_OF.lock().unwrap();
            if let Some(m) = wg.as_mut() {
                m.remove(&(pc as usize));
                if let Some(ref b) = book {
                    for &seg in b.extra_active.iter().chain(b.spares.iter()).chain(b.first.iter()) {
                        m.remove(&seg);
                    }
                }
            }
        }
        if let Some(b) = book {
            for seg in b.extra_active.into_iter().chain(b.spares.into_iter()).chain(b.first) {
                let raw = seg as *mut c_void;
                super::bind::forget(raw);
                let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(raw, 2));
                rel(raw);
            }
        }
        super::bind::forget(pc);
        std::alloc::dealloc(this as *mut u8, std::alloc::Layout::from_size_align(OBJECT_BYTES, 64).unwrap());
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in PC object: slot 5 returns `this`, slots 0 / 2 count.
    #[repr(C)]
    struct Fake {
        vt: *const usize,
        refs: u32,
    }

    unsafe extern "system" fn fake_qi(this: *mut c_void, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
        (*(this as *mut Fake)).refs += 1;
        *ppv = this;
        0
    }
    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *mut Fake)).refs += 1;
        (*(this as *mut Fake)).refs
    }
    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *mut Fake)).refs -= 1;
        (*(this as *mut Fake)).refs
    }
    unsafe extern "system" fn fake_this(this: *mut c_void, a: u64, b: u64, c: u64, d: u64) -> u64 {
        this as u64 + a + 10 * b + 100 * c + 1000 * d
    }

    /// The SDK's inline writers check `write < end` (or `write + n <= end`): never room here.
    #[test]
    fn console_list_objects_have_no_room_and_forward_to_the_pc_list() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release as usize;
        vt[20] = fake_this as usize;
        let fake = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        unsafe {
            let o = wrap(fake as *mut c_void);
            assert_eq!(backing(o), Some(fake as *mut c_void));
            assert_eq!(backing(fake as *mut c_void), None);
            let (w, e) = (*((o as usize + F_WRITE) as *const usize), *((o as usize + F_END) as *const usize));
            assert!(w >= e && *((o as usize + F_DISPATCH_END) as *const usize) == w);
            let slot = |i: usize| *(*(o as *const *const usize)).add(i);
            // a method: the PC list's, with the PC list as `this` and the arguments kept
            let m: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u64) -> u64 = std::mem::transmute(slot(20));
            assert_eq!(m(o, 1, 2, 3, 4), fake as u64 + 4321);
            // QueryInterface hands out the console object – also for the console-only list IIDs
            let q: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(slot(0));
            let mut out = std::ptr::null_mut();
            assert_eq!(q(o, std::ptr::null(), &mut out), 0);
            assert_eq!(out, o);
            let (d1, d2, d3, d4) = CONSOLE_LIST_IIDS[0];
            let gcl5 = GUID { data1: d1, data2: d2, data3: d3, data4: d4 };
            let mut out = std::ptr::null_mut();
            assert_eq!(q(o, &gcl5, &mut out), 0);
            assert_eq!(out, o);
            let r: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(2));
            assert_eq!(r(o), 2);
            assert_eq!(r(o), 1);
            assert_eq!(r(o), 0); // the object is freed with it
            drop(Box::from_raw(fake));
        }
    }
}
