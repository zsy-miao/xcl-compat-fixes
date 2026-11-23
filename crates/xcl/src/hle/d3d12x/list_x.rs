//! The console command list's own methods.
//!
//! The console's ID3D12GraphicsCommandList keeps the PC methods (some moved:
//! `real::console_remap`) and holds console extensions from #60 on (as observed on Fortnite):
//! CopyMemoryToMemoryX #60, FillMemoryWith{32,64,128}BitValueX #61–#63, depth bounds #68,
//! Write{32,64}BitValueBottomOfPipeX #69 / #70, Wait{32,64}BitValueX #71 / #72,
//! DispatchX #78, DrawIndexedX #81, CopyResourceX #84, SetPredicationBufferX #85,
//! Write{32,64}BitValueTopOfPipeX #88 / #89, Write32BitValueBottomOfShaderX #90,
//! KickoffX #93, RSSetDepthBiasX #94. A title call of any other extension (GDS, hints…)
//! is logged and does nothing (`real::class_for`).
//!
//! Occlusion queries may nest on the console; the PC allows one active per list. When one
//! begins while another is active, the outer one ends there on the PC and resolves as visible;
//! the inner one is a real query (Fortnite nests its per-object queries in a dummy outer one).
//!
//! Console GPU addresses in unified memory are the title's own addresses; memory operations
//! go to that window's buffer (`real::uma_window_buffer`), which is in COMMON between them.
//! An index buffer at such an address is given to the PC as the window buffer's GPU address.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::device_x::{self, DepthBias};
use super::real;

const PC_GET_TYPE: usize = 8;
const PC_CLOSE: usize = 9;
const PC_RESET: usize = 10;
const PC_COPY_BUFFER_REGION: usize = 15;
const PC_COPY_TEXTURE_REGION: usize = 16;
const PC_COPY_RESOURCE: usize = 17;
const PC_SET_PIPELINE_STATE: usize = 25;
const PC_RESOURCE_BARRIER: usize = 26;
const PC_IA_SET_INDEX_BUFFER: usize = 43;
const PC_BEGIN_QUERY: usize = 52;
const PC_END_QUERY: usize = 53;
const PC_RESOLVE_QUERY_DATA: usize = 54;
const PC_SET_PREDICATION: usize = 55;
const PC_OM_SET_DEPTH_BOUNDS: usize = 62;
const PC_WRITE_BUFFER_IMMEDIATE: usize = 66;
const PC_QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;

/// Write64BitValueBottomOfPipeX flags 1 / 2, Write64BitValueTopOfPipeX flags 4 / 8: the value
/// written is a GPU / system timestamp
const BOTTOM_TIMESTAMPS: u32 = 0x3;
const TOP_TIMESTAMPS: u32 = 0xC;
/// fills larger than this are not done (one pattern buffer per value, up to this size)
const MAX_FILL: u64 = 256 << 20;
/// Bounded wait timeout for Wait32BitValueX / Wait64BitValueX so an unsatisfied wait never hangs.
const WAIT_TIMEOUT: Duration = Duration::from_secs(2);
/// Cap on tracked per-list written UMA ranges (for same-list write-then-wait detection).
const MAX_WRITTEN_RANGES: usize = 256;

fn once(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < n
}

/// a PC method of the list (its runtime's own)
unsafe fn pc<F: Copy>(list: *mut c_void, slot: usize) -> F {
    transmute_copy(&real::orig_fn(list, slot))
}

unsafe fn list_type(list: *mut c_void) -> D3D12_COMMAND_LIST_TYPE {
    pc::<unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE>(list, PC_GET_TYPE)(list)
}

unsafe fn transition(list: *mut c_void, res: usize, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: transmute_copy(&res),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(list, PC_RESOURCE_BARRIER)(list, 1, &b);
}

/// `op` with the window buffer `res` as a copy destination (back to COMMON after it)
unsafe fn as_copy_dest(list: *mut c_void, res: usize, op: impl FnOnce()) {
    transition(list, res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    op();
    transition(list, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
}

unsafe fn copy_buffer_region(list: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(list, PC_COPY_BUFFER_REGION)(list, dst, dst_off, src, src_off, n);
}

fn warn(counter: &AtomicU64, what: &str, why: std::fmt::Arguments) {
    if once(counter, 4) {
        crate::xlog_warn!("D3D12.X: {what}: {why} – not done");
    }
}

// ---------------------------------------------------------------------------------------
// per-list state & live queue tracking (KickoffX, Wait32/64BitValueX, RSSetDepthBiasX,
// index buffer restoration, nested occlusion queries)
// ---------------------------------------------------------------------------------------

/// a query: (heap, index)
type Query = (usize, u32);

/// One list's occlusion queries: the one active on the PC (with its type) and the outer ones
/// it ended early.
#[derive(Default)]
struct Occlusion {
    active: Option<(Query, D3D12_QUERY_TYPE)>,
    ended_early: Vec<Query>,
}

impl Occlusion {
    /// Begin `q` on the PC; Some(outer): end that one on the PC first.
    fn begin(&mut self, q: Query, ty: D3D12_QUERY_TYPE) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let outer = self.active.replace((q, ty));
        if let Some((o, _)) = outer {
            self.ended_early.push(o);
        }
        outer
    }

    /// End the currently active PC query early (e.g. before a `KickoffX` split), so `EndQuery`
    /// later resolves it as visible.
    fn end_active_early(&mut self) -> Option<(Query, D3D12_QUERY_TYPE)> {
        let active = self.active.take()?;
        self.ended_early.push(active.0);
        Some(active)
    }

    /// Whether the PC ends `q` (not: an outer one already ended – it resolves as visible).
    fn end(&mut self, q: Query) -> bool {
        if self.active.is_some_and(|(a, _)| a == q) {
            self.active = None;
            return true;
        }
        match self.ended_early.iter().position(|e| *e == q) {
            Some(i) => {
                self.ended_early.swap_remove(i);
                false
            }
            None => true,
        }
    }
}

/// Recorded GPU-side memory wait from `Wait32BitValueX` (#71) or `Wait64BitValueX` (#72).
/// Evaluates `(mem_value & mask) <comparison> value` using `D3D12_COMPARISON_FUNC` (1..=8),
/// matching AMD PM4 `WAIT_REG_MEM` semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingWait {
    va: u64,
    width: u8, // 4 or 8
    comparison: u32,
    value: u64,
    mask: u64,
}

fn compare_u64(op: u32, lhs: u64, rhs: u64) -> Option<bool> {
    match op {
        1 => Some(false),        // D3D12_COMPARISON_FUNC_NEVER
        2 => Some(lhs < rhs),    // D3D12_COMPARISON_FUNC_LESS
        3 => Some(lhs == rhs),   // D3D12_COMPARISON_FUNC_EQUAL
        4 => Some(lhs <= rhs),   // D3D12_COMPARISON_FUNC_LESS_EQUAL
        5 => Some(lhs > rhs),    // D3D12_COMPARISON_FUNC_GREATER
        6 => Some(lhs != rhs),   // D3D12_COMPARISON_FUNC_NOT_EQUAL
        7 => Some(lhs >= rhs),   // D3D12_COMPARISON_FUNC_GREATER_EQUAL
        8 => Some(true),         // D3D12_COMPARISON_FUNC_ALWAYS
        _ => None,
    }
}

impl PendingWait {
    unsafe fn read_value(&self) -> Option<u64> {
        let addr = self.va as usize;
        if !crate::gpu::shader::readable(addr, self.width as usize) {
            return None;
        }
        match self.width {
            4 => Some((*(addr as *const AtomicU32)).load(Ordering::Acquire) as u64),
            8 => Some((*(addr as *const AtomicU64)).load(Ordering::Acquire)),
            _ => None,
        }
    }

    unsafe fn is_satisfied(&self) -> bool {
        let Some(cur) = self.read_value() else {
            return true;
        };
        compare_u64(self.comparison, cur & self.mask, self.value).unwrap_or(true)
    }

    unsafe fn wait_until_satisfied(&self, timeout: Duration) -> bool {
        if self.is_satisfied() {
            return true;
        }
        for _ in 0..512 {
            std::hint::spin_loop();
            if self.is_satisfied() {
                return true;
            }
        }
        let start = Instant::now();
        while start.elapsed() < timeout {
            let step = if start.elapsed() < Duration::from_millis(10) {
                Duration::from_micros(100)
            } else {
                Duration::from_millis(1)
            };
            std::thread::sleep(step);
            if self.is_satisfied() {
                return true;
            }
        }
        self.is_satisfied()
    }
}

/// Recorded `OMSetRenderTargets` state for `KickoffX` restoration and `XCL_TRACE_FRAME`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderTargets {
    /// Requested `NumRenderTargetDescriptors` (clamped to D3D12 max 8; 0 if `handles` is empty).
    num_targets: u32,
    /// Saved descriptor handles: 1 base handle when `single != 0` and `num_targets > 0`,
    /// otherwise `num_targets` handles.
    handles: Vec<usize>,
    single: i32,
    dsv: Option<usize>,
}

/// What console code set on a list that a PC Reset clears: `KickoffX` closes, submits and
/// resets the PC list mid-recording, and on the console a kickoff leaves the list's state as it
/// was – so it is set again (descriptor heaps above all: translated shaders index the bound
/// heap, and a draw without one reads garbage descriptors).
#[derive(Default, Clone)]
struct Bindings {
    heaps: Vec<usize>,
    topology: Option<u32>,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<windows::Win32::Foundation::RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    vertex_buffers: std::collections::BTreeMap<u32, D3D12_VERTEX_BUFFER_VIEW>,
    render_targets: Option<RenderTargets>,
    /// slot → the title's (address, bytes, stride) of the vertex buffer (`XCL_TRACE_FRAME`)
    vertex_titles: std::collections::BTreeMap<u32, (u64, u32, u32)>,
    /// the title's (address, bytes, format) of the index buffer (`XCL_TRACE_FRAME`)
    index_title: Option<(u64, u32, i32)>,
    /// the title set the console's rect list topology
    rect_list: bool,
}

/// Whether `list` draws rect lists now, and the title's vertex buffers (slot, (address, bytes, stride)).
pub(super) fn rect_list_state(list: *mut c_void) -> Option<Vec<(u32, (u64, u32, u32), Option<D3D12_VERTEX_BUFFER_VIEW>)>> {
    let g = LISTS.lock().unwrap();
    let b = &g.as_ref()?.get(&(list as usize))?.bindings;
    b.rect_list.then(|| b.vertex_titles.iter().map(|(&k, &v)| (k, v, b.vertex_buffers.get(&k).copied())).collect())
}

/// `pc_va` for other modules.
pub(super) fn pc_va_of(va: u64) -> u64 {
    pc_va(va)
}

/// `XCL_TRACE_FRAME`: the fixed-function state a list last set (topology, first viewport and
/// scissor, render targets) as text.
pub(super) fn state_summary(list: *mut c_void) -> String {
    let mut out = String::new();
    with_bindings(list, |b| {
        let vp = b.viewports.first().map(|v| (v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth));
        let sc = b.scissors.first().map(|r| (r.left, r.top, r.right, r.bottom));
        out = format!(
            "topology {:?}, stencil ref {:?}, viewport {vp:?} ({} set), scissor {sc:?} ({} set), render targets {:?}{}",
            b.topology,
            b.stencil,
            b.viewports.len(),
            b.scissors.len(),
            b.render_targets.as_ref().map(|rt| (rt.num_targets as usize, rt.dsv.is_some())),
            b.render_targets.as_ref().and_then(|rt| rt.dsv).map_or(String::new(), |d| format!(", DSV {d:#x}"))
        );
    });
    out
}

/// `XCL_TRACE_FRAME`: the vertex and index buffers a list binds, with their first vertices
/// and indices as the CPU sees the title's memory.
pub(super) unsafe fn geometry_summary(list: *mut c_void, draw: (bool, u32, u32, u32, i32)) -> String {
    let (vbs, ib) = {
        let mut out = (Vec::new(), None);
        with_bindings(list, |b| out = (b.vertex_titles.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(), b.index_title));
        out
    };
    let words = |a: u64, n: usize| -> Option<Vec<u32>> {
        (a != 0 && crate::gpu::shader::readable(a as usize, 4 * n)).then(|| (0..n).map(|k| ((a as usize + 4 * k) as *const u32).read_unaligned()).collect())
    };
    let mut out = Vec::new();
    for (slot, (a, size, stride)) in vbs {
        let per = (stride as usize / 4).clamp(1, 12);
        let verts: Vec<String> = (0..4u64).filter_map(|i| words(a + i * stride as u64, per).map(|w| format!("{w:x?}"))).collect();
        out.push(format!("VB{slot} {a:#x} ({size} bytes, stride {stride}): {}", verts.join(" ")));
    }
    // the vertices the draw names: through its indices (indexed) or from its first vertex
    let (indexed, count, _, first, base) = draw;
    let n = count.min(6) as usize;
    let names: Option<Vec<u64>> = if indexed {
        ib.and_then(|(a, _, fmt)| {
            let w = if fmt == 57 { 2 } else { 4 };
            let at = a as usize + w * first as usize;
            (a != 0 && crate::gpu::shader::readable(at, w * n)).then(|| {
                (0..n).map(|k| if w == 2 { ((at + 2 * k) as *const u16).read_unaligned() as u64 } else { ((at + 4 * k) as *const u32).read_unaligned() as u64 }).map(|i| (i as i64 + base as i64) as u64).collect()
            })
        })
    } else {
        Some((0..n as u64).map(|k| first as u64 + k).collect())
    };
    out.push(format!("IB {:x?}; the draw's vertices (as indexed now) {names:?}", ib.map(|(a, size, fmt)| (a, size, fmt))));
    // the title fills ring buffers after recording a draw: the vertices are read again when
    // the list is executed (`dump_pending_geometry`)
    if let Some((_, vb)) = vbs_first(list) {
        PENDING_GEOMETRY.lock().unwrap().push((list as usize, vb, ib, draw));
    }
    out.join("; ")
}

/// draws whose vertices are dumped when their list runs: (list, (VB address, bytes, stride),
/// (IB address, bytes, format), draw arguments)
#[allow(clippy::type_complexity)]
static PENDING_GEOMETRY: Mutex<Vec<(usize, (u64, u32, u32), Option<(u64, u32, i32)>, (bool, u32, u32, u32, i32))>> = Mutex::new(Vec::new());

/// `XCL_TRACE_FRAME`: the vertices of traced draws as the title's memory holds them at the
/// next present (the CPU has written the frame's vertices by then).
pub(super) unsafe fn dump_pending_geometry() {
    let mine: Vec<_> = std::mem::take(&mut *PENDING_GEOMETRY.lock().unwrap());
    for (_, (a, _, stride), ib, (indexed, count, _, first, base)) in mine {
        let n = count.min(48) as usize;
        let names: Vec<u64> = match (indexed, ib) {
            (true, Some((ia, _, fmt))) => {
                let w = if fmt == 57 { 2 } else { 4 };
                let at = ia as usize + w * first as usize;
                if !crate::gpu::shader::readable(at, w * n) {
                    continue;
                }
                (0..n).map(|k| if w == 2 { ((at + 2 * k) as *const u16).read_unaligned() as u64 } else { ((at + 4 * k) as *const u32).read_unaligned() as u64 }).map(|i| (i as i64 + base as i64) as u64).collect()
            }
            _ => (0..n as u64).map(|k| first as u64 + k).collect(),
        };
        let per = (stride as usize / 4).clamp(1, 12);
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        let verts: Vec<String> = unique
            .iter()
            .take(24)
            .map(|&i| {
                let at = (a + i * stride as u64) as usize;
                let w: Option<Vec<u32>> = crate::gpu::shader::readable(at, 4 * per).then(|| (0..per).map(|k| ((at + 4 * k) as *const u32).read_unaligned()).collect());
                format!("v{i}: {w:x?}")
            })
            .collect();
        crate::xlog_info!("frame trace: … its vertices at the present (VB {a:#x}, stride {stride}, indices {names:?}): {}", verts.join("; "));
    }
}

fn vbs_first(list: *mut c_void) -> Option<(u32, (u64, u32, u32))> {
    let mut first = None;
    with_bindings(list, |b| first = b.vertex_titles.iter().next().map(|(&k, &v)| (k, v)));
    first
}

/// The depth-stencil view handle a list last bound (`XCL_TRACE_FRAME`).
pub(super) fn bound_dsv(list: *mut c_void) -> Option<usize> {
    let mut d = None;
    with_bindings(list, |b| d = b.render_targets.as_ref().and_then(|rt| rt.dsv));
    d
}

/// depth-stencil views bound by draws that read a watched target: their clears are traced
static WATCHED_DSVS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub(super) fn watch_dsv(dsv: usize) {
    let mut g = WATCHED_DSVS.lock().unwrap();
    if !g.contains(&dsv) && g.len() < 8 {
        g.push(dsv);
    }
}

fn with_bindings(list: *mut c_void, f: impl FnOnce(&mut Bindings)) {
    let mut g = LISTS.lock().unwrap();
    f(&mut g.get_or_insert_with(HashMap::new).entry(list as usize).or_default().bindings);
}

const PC_IA_SET_PRIMITIVE_TOPOLOGY: usize = 20;
const PC_RS_SET_VIEWPORTS: usize = 21;
const PC_RS_SET_SCISSOR_RECTS: usize = 22;
const PC_OM_SET_BLEND_FACTOR: usize = 23;
const PC_OM_SET_STENCIL_REF: usize = 24;
const PC_OM_SET_RENDER_TARGETS: usize = 46;

/// console #20 IASetPrimitiveTopology (PC #20)
/// The console's rect list (AMD's DI_PT_RECTLIST, 0x11): three vertices of a rectangle, the
/// fourth implied. The PC has no such topology; until a geometry shader expands them, each rect
/// is drawn as the triangle of its three vertices (a retail game's full-screen passes).
pub(super) const RECT_LIST: u32 = 17;

pub(super) unsafe extern "system" fn ia_set_topology_x(this: *mut c_void, t: u32) {
    // a topology the PC does not have (console ones): logged once each
    if !matches!(t, 0..=5 | 10..=13 | 33..=64) {
        static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
        let new = {
            let mut g = SEEN.lock().unwrap();
            let n = !g.contains(&t);
            if n {
                g.push(t);
            }
            n
        };
        if new {
            crate::xlog_warn!("D3D12.X: IASetPrimitiveTopology({t}) – a console topology the PC does not have{}", if t == RECT_LIST { ": rect lists are drawn as triangle lists (each rect's first half only)" } else { "" });
        }
    }
    let rect = t == RECT_LIST;
    let t = if t == RECT_LIST { windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32 } else { t };
    with_bindings(this, |b| {
        b.topology = Some(t);
        b.rect_list = rect;
    });
    pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_IA_SET_PRIMITIVE_TOPOLOGY)(this, t)
}

/// console #21 RSSetViewports (PC #21)
pub(super) unsafe extern "system" fn rs_set_viewports_x(this: *mut c_void, n: u32, v: *const D3D12_VIEWPORT) {
    if !v.is_null() {
        let vs = std::slice::from_raw_parts(v, n as usize).to_vec();
        with_bindings(this, |b| b.viewports = vs);
    }
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(this, PC_RS_SET_VIEWPORTS)(this, n, v)
}

/// console #22 RSSetScissorRects (PC #22)
pub(super) unsafe extern "system" fn rs_set_scissors_x(this: *mut c_void, n: u32, r: *const windows::Win32::Foundation::RECT) {
    if !r.is_null() {
        let rs = std::slice::from_raw_parts(r, n as usize).to_vec();
        with_bindings(this, |b| b.scissors = rs);
    }
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, n, r)
}

/// console #23 OMSetBlendFactor (PC #23)
pub(super) unsafe extern "system" fn om_set_blend_factor_x(this: *mut c_void, f: *const [f32; 4]) {
    let v = f.as_ref().copied();
    with_bindings(this, |b| b.blend = v);
    pc::<unsafe extern "system" fn(*mut c_void, *const [f32; 4])>(this, PC_OM_SET_BLEND_FACTOR)(this, f)
}

/// console #24 OMSetStencilRef (PC #24)
pub(super) unsafe extern "system" fn om_set_stencil_ref_x(this: *mut c_void, r: u32) {
    with_bindings(this, |b| b.stencil = Some(r));
    pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_OM_SET_STENCIL_REF)(this, r)
}

/// OMSetRenderTargets as the PC list got it (`real::cl_om_set_render_targets`)
pub(super) unsafe fn note_render_targets(this: *mut c_void, n: u32, rtvs: *const usize, single: i32, dsv: *const usize) {
    let num_targets = if rtvs.is_null() { 0 } else { n.min(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT) };
    let read_count = if single != 0 { num_targets.min(1) } else { num_targets } as usize;
    // SAFETY: `rtvs` is non-null whenever `read_count > 0`, and D3D12 requires at least 1
    // handle when `RTsSingleHandleToDescriptorRange != 0` or `num_targets` handles otherwise.
    let handles = if read_count == 0 { Vec::new() } else { std::slice::from_raw_parts(rtvs, read_count).to_vec() };
    let d = dsv.as_ref().copied();
    with_bindings(this, |b| b.render_targets = Some(RenderTargets { num_targets, handles, single, dsv: d }));
}

/// Set again what `b` holds (after KickoffX's Reset).
unsafe fn restore_bindings(this: *mut c_void, b: &Bindings) {
    if !b.heaps.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, 28)(this, b.heaps.len() as u32, b.heaps.as_ptr());
    }
    if let Some(t) = b.topology {
        pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_IA_SET_PRIMITIVE_TOPOLOGY)(this, t);
    }
    if !b.viewports.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(this, PC_RS_SET_VIEWPORTS)(this, b.viewports.len() as u32, b.viewports.as_ptr());
    }
    if !b.scissors.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(this, PC_RS_SET_SCISSOR_RECTS)(this, b.scissors.len() as u32, b.scissors.as_ptr());
    }
    if let Some(f) = b.blend {
        pc::<unsafe extern "system" fn(*mut c_void, *const [f32; 4])>(this, PC_OM_SET_BLEND_FACTOR)(this, &f);
    }
    if let Some(r) = b.stencil {
        pc::<unsafe extern "system" fn(*mut c_void, u32)>(this, PC_OM_SET_STENCIL_REF)(this, r);
    }
    for (&slot, v) in &b.vertex_buffers {
        pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS)(this, slot, 1, v);
    }
    if let Some(rt) = &b.render_targets {
        let d = rt.dsv.unwrap_or(0);
        let (n, rts_ptr) = if rt.handles.is_empty() { (0, std::ptr::null()) } else { (rt.num_targets, rt.handles.as_ptr()) };
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(this, PC_OM_SET_RENDER_TARGETS)(
            this,
            n,
            rts_ptr,
            rt.single,
            if rt.dsv.is_some() { &d } else { std::ptr::null() },
        );
    }
}

#[derive(Default)]
struct ListState {
    allocator: usize,
    list_type: Option<D3D12_COMMAND_LIST_TYPE>,
    last_queue: Option<usize>,
    /// Title's currently bound PSO (before dynamic depth-bias override)
    bound_pso: usize,
    /// Active `RSSetDepthBiasX` override (None = zero / disabled)
    depth_bias: Option<DepthBias>,
    /// Last depth bias sent to `ID3D12GraphicsCommandList9::RSSetDepthBias` on this PC list
    /// since CreateCommandList / Reset / ClearState / KickoffX (resets to ZERO on each).
    native_bias: DepthBias,
    /// True when the PC list's bound PSO needs patching via the fallback cache before the next draw
    pso_dirty: bool,
    /// True when a fallback depth-bias derived PSO is currently bound on the PC list
    using_fallback_pso: bool,
    /// Bound index buffer (PC address), restored after `DrawIndexedX` and `KickoffX`
    index_buffer: Option<D3D12_INDEX_BUFFER_VIEW>,
    /// Last `OMSetDepthBounds` values, restored after `KickoffX`
    depth_bounds: Option<(f32, f32)>,
    occlusion: Occlusion,
    /// UMA address ranges written by this list since last Reset/KickoffX
    written_ranges: Vec<(u64, u64)>,
    /// Sparse-window `(va, bytes)` ranges written by memory ops (`Write32/64`, `Fill*`,
    /// `CopyMemoryToMemoryX`, timestamps) to copy back to the CPU view after execution.
    sparse_writes: Vec<(usize, usize)>,
    /// External memory waits recorded on this list
    pending_waits: Vec<PendingWait>,
    /// bottom-of-pipe writes into memory the GPU does not see (`cpu_writes`)
    cpu_writes: Vec<super::cpu_writes::Write>,
    /// state a PC Reset clears (restored after KickoffX)
    bindings: Bindings,
}

impl ListState {
    fn note_write(&mut self, va: u64, size: u64) {
        if size == 0 {
            return;
        }
        let end = va.saturating_add(size);
        if self.written_ranges.len() < MAX_WRITTEN_RANGES {
            self.written_ranges.push((va, end));
        } else if let Some(last) = self.written_ranges.last_mut() {
            last.0 = last.0.min(va);
            last.1 = last.1.max(end);
        }
        if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(va as usize)) {
            if self.sparse_writes.is_empty() {
                SPARSE_WRITE_LISTS.fetch_add(1, Ordering::AcqRel);
            }
            self.sparse_writes.push((va as usize, size as usize));
        }
    }

    fn clear_sparse_writes(&mut self) -> Vec<(usize, usize)> {
        if self.sparse_writes.is_empty() {
            return Vec::new();
        }
        SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
        std::mem::take(&mut self.sparse_writes)
    }

    fn has_prior_write(&self, va: u64, size: u64) -> bool {
        let end = va.saturating_add(size);
        self.written_ranges.iter().any(|&(s, e)| va < e && s < end)
    }

    fn clear_waits(&mut self) {
        if !self.pending_waits.is_empty() {
            self.pending_waits.clear();
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn push_wait(&mut self, w: PendingWait) {
        if self.pending_waits.is_empty() {
            PENDING_WAIT_LISTS.fetch_add(1, Ordering::AcqRel);
        }
        self.pending_waits.push(w);
    }
}

#[derive(Clone, Copy)]
struct QueueInfo {
    queue: usize,
    device: usize,
    ty: D3D12_COMMAND_LIST_TYPE,
}

static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);
static LIVE_QUEUES: Mutex<Vec<QueueInfo>> = Mutex::new(Vec::new());
/// Number of lists with at least one pending `Wait32/64BitValueX` (fast-path guard for ExecuteCommandLists).
static PENDING_WAIT_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Number of lists with recorded sparse-window memory writes (fast-path guard for post-execute readback).
static SPARSE_WRITE_LISTS: AtomicUsize = AtomicUsize::new(0);
/// Set once any list uses `RSSetDepthBiasX` (fast-path guard for Draw* hooks).
static DEPTH_BIAS_USED: AtomicBool = AtomicBool::new(false);

/// Record a newly created title command queue so `KickoffX` can find a compatible queue.
pub(super) fn on_queue_create(queue: usize, device: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if queue == 0 {
        return;
    }
    let mut g = LIVE_QUEUES.lock().unwrap();
    g.retain(|q| q.queue != queue);
    g.push(QueueInfo { queue, device, ty });
}

/// Remove a released command queue so `KickoffX` never submits to a dead queue.
pub(super) fn on_queue_release(queue: usize) {
    LIVE_QUEUES.lock().unwrap().retain(|q| q.queue != queue);
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            if s.last_queue == Some(queue) {
                s.last_queue = None;
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Optional test interceptor for `try_native_depth_bias` so unit tests deterministically
    /// verify the native `ID3D12GraphicsCommandList9::RSSetDepthBias` path and its fallback
    /// even on Windows builds whose inbox D3D12Core.dll stops at `ID3D12GraphicsCommandList7`.
    static TEST_NATIVE_BIAS: std::cell::RefCell<Option<Box<dyn FnMut(usize, DepthBias) -> bool>>> = const { std::cell::RefCell::new(None) };
}

/// Apply depth bias natively via `ID3D12GraphicsCommandList9::RSSetDepthBias` (exact f32,
/// without rounding to i32 or deriving a patched PSO). Returns `false` if unavailable.
unsafe fn try_native_depth_bias(list: *mut c_void, bias: DepthBias) -> bool {
    if list.is_null() {
        return false;
    }
    #[cfg(test)]
    if let Some(res) = TEST_NATIVE_BIAS.with(|h| h.borrow_mut().as_mut().map(|f| f(list as usize, bias))) {
        return res;
    }
    let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else {
        return false;
    };
    let Ok(cl9) = cl.cast::<ID3D12GraphicsCommandList9>() else {
        return false;
    };
    cl9.RSSetDepthBias(bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias());
    true
}

/// Record a newly created command list's allocator, initial pipeline, and type.
pub(super) unsafe fn on_create(list: *mut c_void, alloc: usize, pso: usize, ty: D3D12_COMMAND_LIST_TYPE) {
    if list.is_null() {
        return;
    }
    let key = list as usize;
    {
        let mut g = LISTS.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        if let Some(old) = m.remove(&key) {
            if !old.pending_waits.is_empty() {
                PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
            if !old.sparse_writes.is_empty() {
                SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
            }
        }
        m.insert(key, ListState { allocator: alloc, list_type: Some(ty), bound_pso: pso, ..Default::default() });
    }
    // Dynamic-depth-bias PSOs ignore baked RasterizerState depth bias; seed the list with the
    // initial PSO's static bias if non-zero (CreateCommandList resets RSSetDepthBias to 0).
    if ty != D3D12_COMMAND_LIST_TYPE_BUNDLE && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Clean up all tracked state when a command list is released for good.
pub(super) fn on_release(list: usize) {
    if let Some(old) = LISTS.lock().unwrap().as_mut().and_then(|m| m.remove(&list)) {
        if !old.pending_waits.is_empty() {
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
        if !old.sparse_writes.is_empty() {
            SPARSE_WRITE_LISTS.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
pub(super) fn has_list_state(list: usize) -> bool {
    LISTS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&list))
}

/// Forget a released command allocator so `KickoffX` will not reset a list with a dead allocator.
pub(super) fn on_allocator_release(alloc: usize) {
    if let Some(m) = LISTS.lock().unwrap().as_mut() {
        for s in m.values_mut() {
            if s.allocator == alloc {
                s.allocator = 0;
            }
        }
    }
}

/// Track `SetPipelineState` on a command list (for `RSSetDepthBiasX` and `KickoffX`),
/// syncing `ID3D12GraphicsCommandList9::RSSetDepthBias` when the bound PSO uses native dynamic
/// depth bias.
pub(super) unsafe fn on_set_pipeline_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let sync_target = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        // Keep the title-visible base PSO key when `bind_abi` re-binds its resolved background PSO.
        if s.bound_pso == 0 || pso == s.bound_pso || super::pso_async::resolve(s.bound_pso) != pso {
            s.bound_pso = pso;
        }
        let effective = s.bound_pso;
        s.using_fallback_pso = false;
        let is_bundle = s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE);
        let info = if effective != 0 { device_x::base_depth_bias_info(effective) } else { None };
        if !is_bundle {
            if let Some((true, static_bias)) = info {
                let target = s.depth_bias.unwrap_or(static_bias);
                s.pso_dirty = false;
                if target != s.native_bias {
                    Some((target, s.depth_bias.is_some()))
                } else {
                    None
                }
            } else {
                s.pso_dirty = s.depth_bias.is_some();
                if s.pso_dirty {
                    DEPTH_BIAS_USED.store(true, Ordering::Release);
                }
                None
            }
        } else {
            s.pso_dirty = s.depth_bias.is_some();
            if s.pso_dirty {
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some((target, has_override)) = sync_target {
        if try_native_depth_bias(list, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
        } else if has_override {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
    }
}

/// Track `ClearState` on a command list: resets dynamic state and binds `pso`.
pub(super) unsafe fn on_clear_state(list: *mut c_void, pso: usize) {
    let key = list as usize;
    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&key) {
            s.bound_pso = pso;
            s.depth_bias = None;
            s.native_bias = DepthBias::ZERO;
            s.pso_dirty = false;
            s.using_fallback_pso = false;
            s.index_buffer = None;
            s.depth_bounds = None;
            s.bindings = Bindings::default();
            s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
        } else {
            false
        }
    };
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(list, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
}

/// Associate executed lists with `queue` and keep `queue` at the head of `LIVE_QUEUES`.
pub(super) unsafe fn note_queue_for_lists(queue: usize, lists: &[*mut c_void]) {
    if queue == 0 {
        return;
    }
    {
        let mut g = LIVE_QUEUES.lock().unwrap();
        if let Some(pos) = g.iter().position(|q| q.queue == queue) {
            let info = g.remove(pos);
            g.push(info);
        } else {
            let raw = queue as *mut c_void;
            if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
                let ty = q.GetDesc().Type;
                let mut dev: Option<ID3D12Device> = None;
                let dev_raw = if q.GetDevice(&mut dev).is_ok() {
                    dev.as_ref().map_or(0, |d| d.as_raw() as usize)
                } else {
                    0
                };
                g.push(QueueInfo { queue, device: dev_raw, ty });
            }
        }
    }
    let mut g = LISTS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    for &l in lists {
        if !l.is_null() {
            m.entry(l as usize).or_default().last_queue = Some(queue);
        }
    }
}

fn pick_queue(last_queue: Option<usize>, device: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<usize> {
    let g = LIVE_QUEUES.lock().unwrap();
    if let Some(q) = last_queue {
        if g.iter().any(|info| info.queue == q) {
            return Some(q);
        }
    }
    g.iter().rev().find(|info| info.ty == ty && (device == 0 || info.device == 0 || info.device == device)).map(|info| info.queue)
}

fn note_write(list: usize, va: u64, size: u64) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(list).or_default().note_write(va, size);
}

unsafe fn wait_all(waits: &[PendingWait]) {
    static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
    for w in waits {
        if !w.wait_until_satisfied(WAIT_TIMEOUT) && once(&TIMEOUTS, 4) {
            crate::xlog_warn!(
                "D3D12.X: Wait{}BitValueX timed out after {} ms at {:#x} (op {}, want {:#x}, mask {:#x})",
                w.width * 8,
                WAIT_TIMEOUT.as_millis(),
                w.va,
                w.comparison,
                w.value,
                w.mask
            );
        }
    }
}

/// Copy any GPU memory writes into sparse VRAM windows recorded by `lists` back into their
/// CPU view after `queue.ExecuteCommandLists` so CPU reads and `Wait32/64BitValueX` see them.
pub(super) unsafe fn readback_sparse_writes(queue: *mut c_void, lists: &[*mut c_void]) {
    if SPARSE_WRITE_LISTS.load(Ordering::Acquire) == 0 || queue.is_null() || lists.is_empty() {
        return;
    }
    let ranges: Vec<(usize, usize)> = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let mut out = Vec::new();
        for &l in lists {
            if let Some(s) = m.get(&(l as usize)) {
                out.extend_from_slice(&s.sparse_writes);
            }
        }
        out
    };
    if ranges.is_empty() {
        return;
    }
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        super::sparse::readback_after_execute(q, &ranges);
    }
}

/// After `lists` went to `queue`: their writes into memory the GPU does not see (`cpu_writes`).
pub(super) unsafe fn cpu_writes_after(queue: *mut c_void, lists: &[*mut c_void]) {
    let writes: Vec<super::cpu_writes::Write> = {
        let g = LISTS.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        lists.iter().filter_map(|&l| m.get(&(l as usize))).flat_map(|s| s.cpu_writes.iter().cloned()).collect()
    };
    super::cpu_writes::after(queue, writes);
}

/// Submit `lists` through `submit`, splitting the batch before any list whose `Wait32/64BitValueX`
/// condition is not yet satisfied so earlier lists in the same batch can run on the GPU first.
pub(super) unsafe fn execute_with_waits(lists: &[*mut c_void], mut submit: impl FnMut(&[*mut c_void])) {
    if lists.is_empty() || PENDING_WAIT_LISTS.load(Ordering::Acquire) == 0 {
        submit(lists);
        return;
    }
    let per_list: Vec<Vec<PendingWait>> = {
        let g = LISTS.lock().unwrap();
        lists
            .iter()
            .map(|&l| {
                g.as_ref()
                    .and_then(|m| m.get(&(l as usize)))
                    .map(|s| s.pending_waits.clone())
                    .unwrap_or_default()
            })
            .collect()
    };
    let mut start = 0usize;
    for (i, waits) in per_list.iter().enumerate() {
        if waits.is_empty() || waits.iter().all(|w| w.is_satisfied()) {
            continue;
        }
        if i > start {
            submit(&lists[start..i]);
        }
        wait_all(waits);
        start = i;
    }
    if start < lists.len() {
        submit(&lists[start..]);
    }
}

// ---------------------------------------------------------------------------------------
// memory copy / fill / immediate write / timestamp
// ---------------------------------------------------------------------------------------

/// command list #60 CopyMemoryToMemoryX(destination, source, size, flags)
pub(super) unsafe extern "system" fn copy_memory_x(this: *mut c_void, dst: u64, src: u64, size: u64, _flags: u32) {
    note_gpu_write("CopyMemoryToMemoryX", dst as usize, size);
    static N: AtomicU64 = AtomicU64::new(0);
    const W: &str = "CopyMemoryToMemoryX";
    if size == 0 || !size.is_multiple_of(4) {
        return warn(&N, W, format_args!("size {size:#x} is not a non-zero multiple of 4"));
    }
    let (Some((d, doff)), Some((s, soff))) = (real::uma_window_buffer(dst as usize, size as usize), real::uma_window_buffer(src as usize, size as usize)) else {
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) is not in unified memory"));
    };
    if d == s && dst < src + size && src < dst + size {
        return warn(&N, W, format_args!("{src:#x} → {dst:#x} ({size:#x} bytes) overlap"));
    }
    note_write(this as usize, dst, size);
    if d == s {
        // one window: the buffer is the source and the destination (the ranges do not overlap)
        as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
        return;
    }
    transition(this, s, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    as_copy_dest(this, d, || copy_buffer_region(this, d, doff, s, soff, size));
    transition(this, s, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

unsafe fn device_of(list: *mut c_void) -> Option<ID3D12Device> {
    let mut dev: Option<ID3D12Device> = None;
    ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
    dev
}

/// upload buffers holding a repeated 16-byte pattern: (pattern, buffer, bytes)
static PATTERNS: Mutex<Vec<([u8; 16], ID3D12Resource, u64)>> = Mutex::new(Vec::new());

/// A buffer of at least `n` bytes repeating `pat` (the device's upload heap).
unsafe fn pattern_buffer(list: *mut c_void, pat: [u8; 16], n: u64) -> Option<usize> {
    let mut cache = PATTERNS.lock().unwrap();
    if let Some((_, b, _)) = cache.iter().find(|(p, _, len)| *p == pat && *len >= n) {
        return Some(b.as_raw() as usize);
    }
    // a larger one replaces an older pattern buffer only when nothing may still read it: kept
    if cache.len() >= 32 {
        return None;
    }
    let dev = device_of(list)?;
    let len = n.next_power_of_two().max(0x1_0000);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        &mut b,
    )
    .ok()?;
    let b = b?;
    let mut p: *mut c_void = std::ptr::null_mut();
    b.Map(0, None, Some(&mut p)).ok()?;
    for chunk in std::slice::from_raw_parts_mut(p as *mut u8, len as usize).chunks_exact_mut(16) {
        chunk.copy_from_slice(&pat);
    }
    b.Unmap(0, None);
    let raw = b.as_raw() as usize;
    cache.push((pat, b, len));
    Some(raw)
}

unsafe fn fill(list: *mut c_void, va: u64, size: u64, pat: [u8; 16], unit: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    if size == 0 || !size.is_multiple_of(unit) || size > MAX_FILL {
        return warn(&N, what, format_args!("size {size:#x} (a non-zero multiple of {unit} up to {MAX_FILL:#x} is filled)"));
    }
    let Some((d, off)) = real::uma_window_buffer(va as usize, size as usize) else {
        return warn(&N, what, format_args!("{va:#x} ({size:#x} bytes) is not in unified memory"));
    };
    let Some(src) = pattern_buffer(list, pat, size) else {
        return warn(&N, what, format_args!("no pattern buffer for {pat:x?}"));
    };
    note_write(list as usize, va, size);
    as_copy_dest(list, d, || copy_buffer_region(list, d, off, src, 0, size));
}

fn pattern(bytes: &[u8]) -> [u8; 16] {
    let mut p = [0u8; 16];
    for (i, b) in p.iter_mut().enumerate() {
        *b = bytes[i % bytes.len()];
    }
    p
}

/// command list #61 FillMemoryWith32BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill32_x(this: *mut c_void, va: u64, size: u64, value: u32) {
    fill(this, va, size, pattern(&value.to_le_bytes()), 4, "FillMemoryWith32BitValueX");
}

/// command list #62 FillMemoryWith64BitValueX(address, size, value)
pub(super) unsafe extern "system" fn fill64_x(this: *mut c_void, va: u64, size: u64, value: u64) {
    fill(this, va, size, pattern(&value.to_le_bytes()), 8, "FillMemoryWith64BitValueX");
}

/// command list #63 FillMemoryWith128BitValueX(address, size, value*)
pub(super) unsafe extern "system" fn fill128_x(this: *mut c_void, va: u64, size: u64, value: *const [u8; 16]) {
    if !value.is_null() {
        fill(this, va, size, value.read_unaligned(), 16, "FillMemoryWith128BitValueX");
    }
}

/// Values written by the GPU at `va` (WriteBufferImmediate: MARKER_IN = when the command is
/// reached, MARKER_OUT = when the work before it is done).
unsafe fn write(list: *mut c_void, va: u64, words: &[u32], mode: D3D12_WRITEBUFFERIMMEDIATE_MODE, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = 4 * words.len() as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    let (Some((d, _)), Some(at)) = (real::uma_window_buffer(va as usize, n as usize), real::uma_va_of(va as usize)) else {
        // the title's own memory: written by the CPU once the GPU is there
        if super::cpu_writes::writable(va as usize, n as usize) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
                s.cpu_writes.push((va as usize, words.to_vec()));
                return;
            }
        }
        return warn(&N, what, format_args!("{va:#x} is not in unified memory"));
    };
    note_write(list as usize, va, n);
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = words.iter().enumerate().map(|(i, &w)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: at + 4 * i as u64, Value: w }).collect();
    let modes = vec![mode; words.len()];
    type F = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    as_copy_dest(list, d, || pc::<F>(list, PC_WRITE_BUFFER_IMMEDIATE)(list, params.len() as u32, params.as_ptr(), modes.as_ptr()));
}

/// timestamp queries (one heap per device, used round-robin)
static TIMESTAMPS: Mutex<Option<HashMap<usize, ID3D12QueryHeap>>> = Mutex::new(None);
const TIMESTAMP_SLOTS: u32 = 4096;
static NEXT_TIMESTAMP: AtomicU32 = AtomicU32::new(0);

/// The GPU timestamp (the PC queue's ticks) written at `va` when the work before it is done.
unsafe fn timestamp(list: *mut c_void, va: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(list);
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
        return warn(&N, what, format_args!("a timestamp on a type-{} list", ty.0));
    }
    let Some((d, off)) = real::uma_window_buffer(va as usize, 8).filter(|_| va.is_multiple_of(8)) else {
        return warn(&N, what, format_args!("{va:#x} is not 8-byte aligned in unified memory"));
    };
    let Some(dev) = device_of(list) else { return };
    let heap = {
        let mut m = TIMESTAMPS.lock().unwrap();
        let m = m.get_or_insert_with(HashMap::new);
        if let std::collections::hash_map::Entry::Vacant(slot) = m.entry(dev.as_raw() as usize) {
            let mut h: Option<ID3D12QueryHeap> = None;
            let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: TIMESTAMP_SLOTS, NodeMask: 0 };
            match dev.CreateQueryHeap(&desc, &mut h) {
                Ok(()) if h.is_some() => slot.insert(h.unwrap()),
                _ => return warn(&N, what, format_args!("no timestamp query heap")),
            };
        }
        m[&(dev.as_raw() as usize)].as_raw() as usize
    };
    note_write(list as usize, va, 8);
    let i = NEXT_TIMESTAMP.fetch_add(1, Ordering::Relaxed) % TIMESTAMP_SLOTS;
    pc::<unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32)>(list, PC_END_QUERY)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
    type Resolve = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    as_copy_dest(list, d, || pc::<Resolve>(list, PC_RESOLVE_QUERY_DATA)(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, i, 1, d, off));
}

/// command list #69 Write32BitValueBottomOfPipeX(address, value, flush, flags)
pub(super) unsafe extern "system" fn write32_bottom_x(this: *mut c_void, va: u64, value: u32, _flush: u32, _flags: u32) {
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfPipeX");
}

/// command list #70 Write64BitValueBottomOfPipeX(address, value, flush, flags)
pub(super) unsafe extern "system" fn write64_bottom_x(this: *mut c_void, va: u64, value: u64, _flush: u32, flags: u32) {
    const W: &str = "Write64BitValueBottomOfPipeX";
    if flags & BOTTOM_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, W);
}

/// Record a `Wait32BitValueX` or `Wait64BitValueX` memory wait on `list`.
///
/// If an earlier write in the same open list already targeted an overlapping UMA range, the
/// `COMMON -> COPY_DEST -> COMMON` barrier around that write already orders subsequent work
/// on the GPU, so we do not stall before submission (which would deadlock waiting on our own
/// unsubmitted list). Otherwise the wait is evaluated on the CPU before `ExecuteCommandLists`
/// (or `KickoffX`) submits the list.
unsafe fn wait_memory(list: *mut c_void, va: u64, width: u8, comparison: u32, value: u64, mask: u64, what: &str) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = width as u64;
    if !va.is_multiple_of(n) {
        return warn(&N, what, format_args!("{va:#x} is not {n}-byte aligned"));
    }
    // (the wait reads the memory on the CPU: the title's own memory too, which `cpu_writes` fills)
    if real::uma_window_buffer(va as usize, width as usize).is_none() && !crate::gpu::shader::readable(va as usize, width as usize) {
        return warn(&N, what, format_args!("{va:#x} is not readable memory"));
    }
    if compare_u64(comparison, 0, 0).is_none() {
        return warn(&N, what, format_args!("unknown comparison op {comparison}"));
    }
    if comparison == D3D12_COMPARISON_FUNC_ALWAYS.0 as u32 {
        return;
    }
    if comparison == D3D12_COMPARISON_FUNC_NEVER.0 as u32 {
        return warn(&N, what, format_args!("NEVER comparison at {va:#x} (skipped to avoid hang)"));
    }
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
    if s.has_prior_write(va, n) {
        return;
    }
    s.push_wait(PendingWait { va, width, comparison, value, mask });
}

/// command list #71 Wait32BitValueX(address, comparison, value, mask, flags)
pub(super) unsafe extern "system" fn wait32_x(this: *mut c_void, va: u64, comparison: u32, value: u32, mask: u32, _flags: u32) {
    wait_memory(this, va, 4, comparison, value as u64, mask as u64, "Wait32BitValueX");
}

/// command list #72 Wait64BitValueX(address, comparison, value, mask, flags)
pub(super) unsafe extern "system" fn wait64_x(this: *mut c_void, va: u64, comparison: u32, value: u64, mask: u64, _flags: u32) {
    wait_memory(this, va, 8, comparison, value, mask, "Wait64BitValueX");
}

/// command list #88 Write32BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write32_top_x(this: *mut c_void, va: u64, value: u32, _flags: u32) {
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, "Write32BitValueTopOfPipeX");
}

/// command list #89 Write64BitValueTopOfPipeX(address, value, flags)
pub(super) unsafe extern "system" fn write64_top_x(this: *mut c_void, va: u64, value: u64, flags: u32) {
    const W: &str = "Write64BitValueTopOfPipeX";
    if flags & TOP_TIMESTAMPS != 0 {
        return timestamp(this, va, W);
    }
    write(this, va, &[value as u32, (value >> 32) as u32], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN, W);
}

/// command list #90 Write32BitValueBottomOfShaderX(address, value, stage)
pub(super) unsafe extern "system" fn write32_bottom_of_shader_x(this: *mut c_void, va: u64, value: u32, _stage: u32) {
    write(this, va, &[value], D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, "Write32BitValueBottomOfShaderX");
}

/// command list #68 OMSetDepthBounds(min, max) (PC #62)
pub(super) unsafe extern "system" fn set_depth_bounds_x(this: *mut c_void, min: f32, max: f32) {
    LISTS.lock().unwrap().get_or_insert_with(HashMap::new).entry(this as usize).or_default().depth_bounds = Some((min, max));
    pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(this, PC_OM_SET_DEPTH_BOUNDS)(this, min, max);
}

/// command list #78 DispatchX(x, y, z, flags)
pub(super) unsafe extern "system" fn dispatch_x(this: *mut c_void, x: u32, y: u32, z: u32, _flags: u32) {
    super::bind::dispatch(this, x, y, z);
}

/// command list #146 console compute/UAV barrier (used by `XboxBvhBuilder.dll` between BVH
/// build passes): emits a global UAV barrier so UMA writes become visible to the next pass
pub(super) unsafe extern "system" fn uav_barrier_x(this: *mut c_void, _n: u32, _barriers: *const c_void) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }),
        },
    };
    pc::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(this, PC_RESOURCE_BARRIER)(this, 1, &b);
}

/// command list #84 CopyResourceX(destination, source, flags); in place (the console's
/// compression operations) there is nothing to do on PC
pub(super) unsafe extern "system" fn copy_resource_flags_x(this: *mut c_void, dst: usize, src: usize, _flags: u32) {
    if dst != src {
        copy_resource_x(this, dst, src);
    }
}

/// command list #85 SetPredicationBufferX({ u64 address; u32 operation }*)
pub(super) unsafe extern "system" fn set_predication_buffer_x(this: *mut c_void, desc: *const u64) {
    static N: AtomicU64 = AtomicU64::new(0);
    type F = unsafe extern "system" fn(*mut c_void, usize, u64, D3D12_PREDICATION_OP);
    let f = pc::<F>(this, PC_SET_PREDICATION);
    if desc.is_null() {
        return f(this, 0, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }
    let (va, op) = (desc.read_unaligned(), (desc.add(1) as *const u32).read_unaligned());
    let Some((b, off)) = real::uma_window_buffer(va as usize, 8).filter(|_| va.is_multiple_of(8)) else {
        return warn(&N, "SetPredicationBufferX", format_args!("{va:#x} is not 8-byte aligned in unified memory"));
    };
    transition(this, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION);
    f(this, b, off, D3D12_PREDICATION_OP(op as i32));
    transition(this, b, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON);
}

/// command list #93 KickoffX(): flush commands recorded so far by closing the list, submitting
/// it to a compatible live command queue, and resetting it on the same command allocator so
/// recording can continue seamlessly until the title's normal `Close` + `ExecuteCommandLists`.
///
/// If no compatible queue or allocator is known yet (e.g. a bundle, or a list created before
/// any queue of its type exists), `KickoffX` is a safe logged no-op and the recorded work runs
/// at `Close` + `ExecuteCommandLists`.
pub(super) unsafe extern "system" fn kickoff_x(this: *mut c_void) {
    static NO_QUEUE: AtomicU64 = AtomicU64::new(0);
    static ERR: AtomicU64 = AtomicU64::new(0);
    let ty = list_type(this);
    if ty == D3D12_COMMAND_LIST_TYPE_BUNDLE {
        return;
    }
    let dev_raw = device_of(this).map_or(0, |d| d.as_raw() as usize);
    let (alloc, queue, bound_pso, depth_bias, index_buffer, depth_bounds, active_query, waits, sparse_writes) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(this as usize))) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        };
        if s.allocator == 0 {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – list allocator unknown, deferring to Close + ExecuteCommandLists");
            }
            return;
        }
        let Some(q) = pick_queue(s.last_queue, dev_raw, s.list_type.unwrap_or(ty)) else {
            if once(&NO_QUEUE, 4) {
                crate::xlog_info!("D3D12.X: KickoffX – no live queue of type {} yet, deferring to Close + ExecuteCommandLists", ty.0);
            }
            return;
        };
        let active_query = s.occlusion.end_active_early();
        let waits = if s.pending_waits.is_empty() {
            Vec::new()
        } else {
            PENDING_WAIT_LISTS.fetch_sub(1, Ordering::AcqRel);
            std::mem::take(&mut s.pending_waits)
        };
        let sparse_writes = s.clear_sparse_writes();
        s.written_ranges.clear();
        let cpu_writes = std::mem::take(&mut s.cpu_writes);
        (s.allocator, q, s.bound_pso, s.depth_bias, s.index_buffer, s.depth_bounds, active_query, waits, (sparse_writes, cpu_writes))
    };
    let (sparse_writes, cpu_writes) = sparse_writes;

    // End any active occlusion query before Close (PC forbids closing with an open query).
    if let Some(((h, i), qty)) = active_query {
        pc::<QueryFn>(this, PC_END_QUERY)(this, h, qty, i);
    }

    let ext = *(this as *const usize);
    let close_hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_CLOSE)(this);
    real::preserve_vtable(this, ext);
    if close_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Close failed ({close_hr:#x})"));
    }

    let q_raw = queue as *mut c_void;
    super::title_textures::refresh(q_raw);
    if !waits.is_empty() {
        wait_all(&waits);
    }

    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let exec: ExecFn = transmute_copy(&real::orig_fn(q_raw, PC_QUEUE_EXECUTE_COMMAND_LISTS));
    let one = [this];
    exec(q_raw, 1, one.as_ptr());
    if !sparse_writes.is_empty() {
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&q_raw) {
            super::sparse::readback_after_execute(q, &sparse_writes);
        }
    }
    super::cpu_writes::after(q_raw, cpu_writes);

    let ext_closed = *(this as *const usize);
    let init_pso = super::pso_async::resolve_for_bind(bound_pso);
    let reset_hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(this, PC_RESET)(this, alloc, init_pso);
    real::preserve_vtable(this, ext_closed);
    if reset_hr < 0 {
        return warn(&ERR, "KickoffX", format_args!("Reset after kickoff failed ({reset_hr:#x})"));
    }
    let bindings = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize))).map(|s| s.bindings.clone());
    if let Some(b) = bindings {
        restore_bindings(this, &b);
    }

    // Reset() cleared the PC list's native RSSetDepthBias state to (0, 0, 0) and bound init_pso.
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
        s.native_bias = DepthBias::ZERO;
        s.using_fallback_pso = false;
        s.pso_dirty = false;
    }
    let info = if bound_pso != 0 { device_x::base_depth_bias_info(bound_pso) } else { None };
    let mut applied_native = false;
    if let Some((true, static_bias)) = info {
        let target = depth_bias.unwrap_or(static_bias);
        if target.is_zero() {
            applied_native = true;
        } else if try_native_depth_bias(this, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                s.native_bias = target;
            }
            applied_native = true;
        }
    } else if bound_pso == 0 {
        if let Some(b) = depth_bias {
            if device_x::dynamic_depth_bias_supported(dev_raw as *mut c_void) && try_native_depth_bias(this, b) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                    s.native_bias = b;
                }
                applied_native = true;
            }
        }
    }
    if depth_bias.is_some() && !applied_native {
        if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
            s.pso_dirty = true;
        }
        DEPTH_BIAS_USED.store(true, Ordering::Release);
    }
    if let Some(ref view) = index_buffer {
        set_index_buffer(this, Some(view));
    }
    if let Some((min, max)) = depth_bounds {
        pc::<unsafe extern "system" fn(*mut c_void, f32, f32)>(this, PC_OM_SET_DEPTH_BOUNDS)(this, min, max);
    }
}

/// command list #94 RSSetDepthBiasX(depthBias, depthBiasClamp, slopeScaledDepthBias):
/// dynamic depth bias override. Uses `ID3D12GraphicsCommandList9::RSSetDepthBias` natively
/// when `D3D12_FEATURE_D3D12_OPTIONS16::DynamicDepthBiasSupported` and the bound PSO support
/// it (preserving exact `f32` bias with zero PSO derivation), falling back to the patched-PSO
/// cache (`device_x::depth_bias_pso`) otherwise.
pub(super) unsafe extern "system" fn set_depth_bias_x(this: *mut c_void, depth_bias: f32, depth_bias_clamp: f32, slope_scaled_depth_bias: f32) {
    static NON_FINITE: AtomicU64 = AtomicU64::new(0);
    let clean = |v: f32| {
        if v.is_finite() {
            v
        } else {
            if once(&NON_FINITE, 4) {
                crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – non-finite value {v} clamped to 0.0");
            }
            0.0
        }
    };
    let bias = DepthBias::new(clean(depth_bias), clean(depth_bias_clamp), clean(slope_scaled_depth_bias));
    let key = this as usize;

    let (bound_pso, is_bundle) = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        (s.bound_pso, s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE))
    };

    let can_native = if is_bundle {
        false
    } else if bound_pso != 0 {
        matches!(device_x::base_depth_bias_info(bound_pso), Some((true, _)))
    } else {
        device_of(this).is_some_and(|dev| device_x::dynamic_depth_bias_supported(dev.as_raw()))
    };

    if can_native {
        let target = if bias.is_zero() { DepthBias::ZERO } else { bias };
        let (need_restore_base, need_native_call) = {
            let mut g = LISTS.lock().unwrap();
            let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
            s.depth_bias = if bias.is_zero() { None } else { Some(bias) };
            s.pso_dirty = false;
            let restore = s.using_fallback_pso && s.bound_pso != 0;
            s.using_fallback_pso = false;
            (if restore { Some(s.bound_pso) } else { None }, target != s.native_bias)
        };
        if let Some(base) = need_restore_base {
            let resolved = super::pso_async::resolve_for_bind(base);
            pc::<unsafe extern "system" fn(*mut c_void, usize)>(this, PC_SET_PIPELINE_STATE)(this, resolved);
        }
        if !need_native_call {
            return;
        }
        if try_native_depth_bias(this, target) {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.native_bias = target;
            }
            return;
        }
        // ID3D12GraphicsCommandList9 unavailable: fall through to patched-PSO fallback.
        if !bias.is_zero() {
            if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&key)) {
                s.pso_dirty = true;
            }
            DEPTH_BIAS_USED.store(true, Ordering::Release);
        }
        return;
    }

    let restore = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(key).or_default();
        if bias.is_zero() {
            let was_active = s.depth_bias.take().is_some();
            let was_fallback = std::mem::take(&mut s.using_fallback_pso);
            s.pso_dirty = false;
            if (was_active || was_fallback) && s.bound_pso != 0 {
                Some(s.bound_pso)
            } else {
                None
            }
        } else {
            if s.depth_bias != Some(bias) || !s.using_fallback_pso {
                s.depth_bias = Some(bias);
                s.pso_dirty = true;
                DEPTH_BIAS_USED.store(true, Ordering::Release);
            }
            None
        }
    };
    if let Some(base) = restore {
        let resolved = super::pso_async::resolve_for_bind(base);
        pc::<unsafe extern "system" fn(*mut c_void, usize)>(this, PC_SET_PIPELINE_STATE)(this, resolved);
    }
}

/// Apply a pending fallback `RSSetDepthBiasX` override before a draw call on `list`.
pub(super) unsafe fn prepare_draw(list: *mut c_void) {
    if !DEPTH_BIAS_USED.load(Ordering::Acquire) {
        return;
    }
    let (bound_pso, bias) = {
        let mut g = LISTS.lock().unwrap();
        let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) else {
            return;
        };
        if !s.pso_dirty {
            return;
        }
        s.pso_dirty = false;
        (s.bound_pso, s.depth_bias)
    };
    if bound_pso == 0 {
        return;
    }
    let (target, is_fallback) = match bias {
        Some(b) => {
            static MISS: AtomicU64 = AtomicU64::new(0);
            let Some(dev) = device_of(list) else { return };
            match device_x::depth_bias_pso(dev.as_raw(), bound_pso, b) {
                Some(derived) => (derived, true),
                None => {
                    if once(&MISS, 4) {
                        crate::xlog_warn!("D3D12.X: RSSetDepthBiasX – could not derive pipeline for {bound_pso:#x}, using base pipeline");
                    }
                    return;
                }
            }
        }
        None => (bound_pso, false),
    };
    let resolved = super::pso_async::resolve_for_bind(target);
    pc::<unsafe extern "system" fn(*mut c_void, usize)>(list, PC_SET_PIPELINE_STATE)(list, resolved);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(list as usize))) {
        s.using_fallback_pso = is_fallback;
    }
}

// ---------------------------------------------------------------------------------------
// index buffers & DrawIndexedX
// ---------------------------------------------------------------------------------------

/// a console GPU address as the PC's (unified memory: the window buffer's)
/// A GPU address the title gave: its memory's PC address when it is the title's memory, else
/// as it is (a PC resource's GPU address – it may lie in a window's range by number).
fn pc_va(va: u64) -> u64 {
    if !crate::gpu::uma::get().is_some_and(|u| u.is_title_address(va as usize)) {
        return va;
    }
    real::uma_va_of(va as usize).unwrap_or(va)
}

/// `pc_va` for an index or vertex buffer of `bytes` bytes bound on `list`. A retail title also
/// binds buffers in its ordinary memory (a private allocation outside the unified-memory
/// windows – on the console the GPU reads all of the title's memory). The PC's GPU has no
/// address for that (IASetIndexBuffer of it: debug layer message 725, then the device hangs,
/// 0x887a0006), so the bytes are copied into the list's upload memory as they are at the bind;
/// what the title writes there afterwards is not seen.
///
/// A buffer that stays bound (`mirror`) gets one block of unified memory for good – its
/// *mirror*, the same for every bind of that address and size – and each draw copies the part
/// it reads there first (`sync_title_buffers`), as the title wrote it by then. (A copy of the
/// whole buffer at every bind filled unified memory within two minutes of a retail title's
/// menus, and missed what the title wrote after the bind.)
unsafe fn gpu_va(list: *mut c_void, va: u64, bytes: usize, what: &str, mirror: bool) -> u64 {
    let t = pc_va(va);
    if t != va || va == 0 || bytes == 0 {
        return t;
    }
    // in a window by number: a PC resource's GPU address
    if crate::gpu::uma::get().is_none_or(|u| u.contains(va as usize)) || !crate::gpu::shader::readable(va as usize, bytes) {
        return t;
    }
    if mirror {
        if let Some(p) = mirror_of(va, bytes, what) {
            return pc_va(p as u64);
        }
    }
    let p = super::bind::allocate_upload_memory(list, bytes, 16);
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if p.is_null() {
        if n < 4 {
            crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory and there is no room for a copy");
        }
        return t;
    }
    std::ptr::copy_nonoverlapping(va as *const u8, p, bytes);
    if n < 4 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory – bound as a copy made now (#{n})");
    }
    pc_va(p as u64)
}

/// Ranges of the title's memory with a mirror in unified memory: (start, bytes, mirror). A
/// title binds one buffer at many offsets (a ring it fills through the frame: a retail title's
/// views all end at the buffer's end), so a mirror covers a 64 KiB-aligned range and a view
/// inside a mirrored range uses that mirror at its offset.
static MIRRORS: Mutex<Vec<(u64, usize, usize)>> = Mutex::new(Vec::new());
/// bytes of mirrors made (never freed)
static MIRROR_BYTES: AtomicU64 = AtomicU64::new(0);

/// Where `[va, va + bytes)` is in a mirror, if it is in one.
fn mirrored(va: u64, bytes: usize) -> Option<usize> {
    MIRRORS.lock().unwrap().iter().find(|&&(s, l, _)| va >= s && va + bytes as u64 <= s + l as u64).map(|&(s, _, m)| m + (va - s) as usize)
}

/// The mirror address of the title's buffer at `va`; its range's mirror is made (and filled
/// once) the first time.
unsafe fn mirror_of(va: u64, bytes: usize, what: &str) -> Option<usize> {
    if let Some(p) = mirrored(va, bytes) {
        return Some(p);
    }
    let mut g = MIRRORS.lock().unwrap();
    let start = va & !0xffff;
    let len = ((va + bytes as u64 + 0xffff) & !0xffff) - start;
    // only as much as is the title's readable memory (the view itself is)
    let (start, len) = if crate::gpu::shader::readable(start as usize, len as usize) { (start, len as usize) } else { (va, bytes) };
    if MIRROR_BYTES.load(Ordering::Relaxed) + len as u64 > 2 << 30 {
        return None;
    }
    let u = crate::gpu::uma::get()?;
    let p = match u.has_sparse().then(|| super::sparse::alloc(len, 0x1_0000)).flatten() {
        Some((p, _)) => p,
        None => u.alloc(len, 0x1_0000)? as usize,
    };
    std::ptr::copy_nonoverlapping(start as *const u8, p as *mut u8, len);
    g.push((start, len, p));
    let total = MIRROR_BYTES.fetch_add(len as u64, Ordering::Relaxed) + len as u64;
    if g.len() <= 8 || g.len().is_power_of_two() {
        crate::xlog_warn!("D3D12.X: {what} at {va:#x} ({bytes} bytes) is outside unified memory – {start:#x}..{:#x} mirrored at {p:#x}, each draw copies what it reads ({} mirrors, {} MB)", start + len as u64, g.len(), total >> 20);
    }
    Some(p + (va - start) as usize)
}

/// Before a draw on `list`: the parts of mirrored buffers it reads are copied from the title's
/// memory. `indexed`: (first index, index count, base vertex); else `first_vertex`, `vertices`.
pub(super) unsafe fn sync_title_buffers(list: *mut c_void, indexed: Option<(u32, u32, i32)>, first_vertex: u32, vertices: u32, instances: u32) {
    if MIRROR_BYTES.load(Ordering::Relaxed) == 0 {
        return;
    }
    let (vbs, ib) = {
        let g = LISTS.lock().unwrap();
        let Some(b) = g.as_ref().and_then(|m| m.get(&(list as usize))).map(|s| &s.bindings) else { return };
        (b.vertex_titles.values().copied().collect::<Vec<_>>(), b.index_title)
    };
    let mirrors: Vec<Option<usize>> = {
        let mut v: Vec<Option<usize>> = vbs.iter().map(|&(va, bytes, _)| mirrored(va, bytes as usize)).collect();
        v.push(ib.and_then(|(va, bytes, _)| mirrored(va, bytes as usize)));
        if v.iter().all(Option::is_none) {
            return;
        }
        v
    };
    let copy = |va: u64, mirror: usize, bytes: u32, from: u64, len: u64| {
        let from = from.min(bytes as u64);
        let len = len.min(bytes as u64 - from) as usize;
        if len > 0 && crate::gpu::shader::readable(va as usize + from as usize, len) {
            std::ptr::copy_nonoverlapping((va + from) as *const u8, (mirror + from as usize) as *mut u8, len);
        }
    };
    // the vertices read: [lo, hi)
    let (lo, hi) = match (indexed, ib) {
        (Some((first, count, base)), Some((va, bytes, format))) => {
            let size: u64 = if format == DXGI_FORMAT_R32_UINT.0 { 4 } else { 2 };
            let (from, len) = (first as u64 * size, count as u64 * size);
            if let Some(m) = mirrors[vbs.len()] {
                copy(va, m, bytes, from, len);
            }
            let n = (len.min((bytes as u64).saturating_sub(from)) / size) as usize;
            if n == 0 || !crate::gpu::shader::readable((va + from) as usize, n * size as usize) {
                (0, u64::MAX)
            } else {
                let at = (va + from) as *const u8;
                let (mut lo, mut hi) = (u32::MAX, 0u32);
                for k in 0..n {
                    let i = if size == 4 { at.add(4 * k).cast::<u32>().read_unaligned() } else { at.add(2 * k).cast::<u16>().read_unaligned() as u32 };
                    // strip cut values are not vertices
                    if i == u32::MAX || (size == 2 && i == 0xffff) {
                        continue;
                    }
                    lo = lo.min(i);
                    hi = hi.max(i);
                }
                if lo > hi {
                    return;
                }
                ((lo as i64 + base as i64).max(0) as u64, (hi as i64 + base as i64 + 1).max(0) as u64)
            }
        }
        (Some(_), None) => (0, u64::MAX),
        (None, _) => (first_vertex as u64, first_vertex as u64 + vertices as u64),
    };
    for (k, &(va, bytes, stride)) in vbs.iter().enumerate() {
        let Some(m) = mirrors[k] else { continue };
        // a buffer stepped by instance is not told apart: with instances, all of it
        if instances > 1 || hi == u64::MAX || stride == 0 {
            copy(va, m, bytes, 0, bytes as u64);
        } else {
            copy(va, m, bytes, lo * stride as u64, (hi - lo) * stride as u64);
        }
    }
}

unsafe fn set_index_buffer(list: *mut c_void, view: Option<&D3D12_INDEX_BUFFER_VIEW>) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_INDEX_BUFFER_VIEW)>(list, PC_IA_SET_INDEX_BUFFER);
    f(list, view.map_or(std::ptr::null(), |v| v as *const _));
}

/// The heaps a list binds for console code: a view heap in the shared shader-visible heap
/// is that one (`console_heap`), which is bound whatever the title names (a console
/// descriptor table may be in any heap); a view heap that kept its own shader-visible PC heap
/// (no shared heap) is bound itself, for its local indices; sampler heaps as they are. A view
/// heap whose views were meant for the shared heap but found no room there is CPU-only
/// (`dev_create_descriptor_heap_x` dropped the flag) and cannot be bound.
pub(super) unsafe fn pc_heaps(n: u32, heaps: *const usize) -> Vec<Option<ID3D12DescriptorHeap>> {
    let mut view_heap = super::console_heap::global_heap();
    let mut out = Vec::new();
    for k in 0..n as usize {
        let p = *heaps.add(k) as *mut c_void;
        if p.is_null() || super::console_heap::in_global(p as usize) {
            continue;
        }
        let Some(h) = ID3D12DescriptorHeap::from_raw_borrowed(&p) else { continue };
        let desc = h.GetDesc();
        if desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV {
            if desc.Flags.contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
                view_heap = Some(h.clone());
            }
        } else {
            out.push(Some(h.clone()));
        }
    }
    out.extend(view_heap.map(Some));
    out
}

/// console #32 SetDescriptorHeaps (PC #28)
pub(super) unsafe extern "system" fn set_descriptor_heaps_x(this: *mut c_void, n: u32, heaps: *const usize) {
    const PC_SET_DESCRIPTOR_HEAPS: usize = 28;
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize)>(this, PC_SET_DESCRIPTOR_HEAPS);
    if heaps.is_null() || super::console_heap::global_heap().is_none() {
        if !heaps.is_null() {
            let given = std::slice::from_raw_parts(heaps, n as usize).to_vec();
            with_bindings(this, |b| b.heaps = given);
        }
        return f(this, n, heaps);
    }
    let h = pc_heaps(n, heaps);
    let raw: Vec<usize> = h.iter().map(|x| x.as_ref().map_or(0, |x| x.as_raw() as usize)).collect();
    let kept = raw.clone();
    with_bindings(this, |b| b.heaps = kept);
    f(this, raw.len() as u32, raw.as_ptr())
}

const PC_IA_SET_VERTEX_BUFFERS: usize = 44;

/// command list #48 IASetVertexBuffers(start slot, count, views) (PC #44): the addresses are
/// the title's (its memory: the GPU address of that memory for the PC)
pub(super) unsafe extern "system" fn ia_set_vertex_buffers_x(this: *mut c_void, start: u32, n: u32, views: *const D3D12_VERTEX_BUFFER_VIEW) {
    let f = pc::<unsafe extern "system" fn(*mut c_void, u32, u32, *const D3D12_VERTEX_BUFFER_VIEW)>(this, PC_IA_SET_VERTEX_BUFFERS);
    if views.is_null() || n == 0 {
        return f(this, start, n, views);
    }
    let v: Vec<D3D12_VERTEX_BUFFER_VIEW> = (0..n as usize)
        .map(|k| {
            let mut v = views.add(k).read_unaligned();
            let title = v.BufferLocation;
            v.BufferLocation = gpu_va(this, v.BufferLocation, v.SizeInBytes as usize, "vertex buffer", true);
            if super::present::trace_frame() {
                static N: AtomicU64 = AtomicU64::new(0);
                let i = N.fetch_add(1, Ordering::Relaxed);
                if i < 12 || (i.is_power_of_two() && i < 1 << 16) {
                    let n = (v.SizeInBytes as usize).min(256);
                    let data = (title != 0 && crate::gpu::shader::readable(title as usize, n)).then(|| std::slice::from_raw_parts(title as *const u8, n).iter().filter(|&&b| b != 0).count());
                    let win = crate::gpu::uma::get().and_then(|u| u.window_of(title as usize).map(|w| (w, u.window(w), u.is_sparse(title as usize))));
                    let mut mbi: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    let q = windows_sys::Win32::System::Memory::VirtualQuery(title as *const c_void, &mut mbi, std::mem::size_of_val(&mbi));
                    crate::xlog_info!(
                        "frame trace: vertex buffer #{i} slot {}: title {title:#x} → PC {:#x}, {} bytes, stride {}; non-zero in the first {n} bytes (CPU view): {data:?}; window (index, (base, size), sparse) {win:x?}; memory state {:#x} protect {:#x} (query {q})",
                        start + k as u32,
                        v.BufferLocation,
                        v.SizeInBytes,
                        v.StrideInBytes,
                        mbi.State,
                        mbi.Protect
                    );
                }
            }
            v
        })
        .collect();
    let kept = v.clone();
    let titles: Vec<(u64, u32, u32)> = (0..n as usize).map(|k| { let t = views.add(k).read_unaligned(); (t.BufferLocation, t.SizeInBytes, t.StrideInBytes) }).collect();
    with_bindings(this, |b| {
        for (k, view) in kept.into_iter().enumerate() {
            b.vertex_buffers.insert(start + k as u32, view);
            b.vertex_titles.insert(start + k as u32, titles[k]);
        }
    });
    f(this, start, n, v.as_ptr())
}

/// command list #47 IASetIndexBuffer(view) (PC #43): the address is the title's
pub(super) unsafe extern "system" fn ia_set_index_buffer_x(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
    let mut g = LISTS.lock().unwrap();
    let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
    if view.is_null() {
        s.index_buffer = None;
        drop(g);
        return set_index_buffer(this, None);
    }
    let mut v = view.read_unaligned();
    s.bindings.index_title = Some((v.BufferLocation, v.SizeInBytes, v.Format.0));
    drop(g);
    v.BufferLocation = gpu_va(this, v.BufferLocation, v.SizeInBytes as usize, "index buffer", true);
    if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
        s.index_buffer = Some(v);
    }
    set_index_buffer(this, Some(&v));
}

/// command list #81 DrawIndexedX(index address, index format, index count, flags): one
/// instance of `count` indices read from that address; the bound index buffer stays
pub(super) unsafe extern "system" fn draw_indexed_x(this: *mut c_void, va: u64, format: DXGI_FORMAT, count: u32, _flags: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    let stride = match format {
        DXGI_FORMAT_R16_UINT => 2,
        DXGI_FORMAT_R32_UINT => 4,
        f => return warn(&N, "DrawIndexedX", format_args!("index format {}", f.0)),
    };
    if count == 0 {
        return;
    }
    let Some(list) = super::bind::prepare_indexed_draw(this, count, 1, 0, 0) else {
        return;
    };
    let bound = LISTS.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).and_then(|s| s.index_buffer));
    sync_title_buffers(this, None, 0, 0, 2);
    let view = D3D12_INDEX_BUFFER_VIEW { BufferLocation: gpu_va(this, va, count.saturating_mul(stride) as usize, "DrawIndexedX indices", false), SizeInBytes: count.saturating_mul(stride), Format: format };
    set_index_buffer(this, Some(&view));
    super::bind::emit_prepared_indexed_draw(&list, count, 1, 0, 0, 0);
    set_index_buffer(this, bound.as_ref());
}

// ---------------------------------------------------------------------------------------
// nested occlusion queries & Reset
// ---------------------------------------------------------------------------------------

/// outer queries ended early, and the result they resolve to
static VISIBLE: Mutex<Option<HashMap<Query, u64>>> = Mutex::new(None);
/// what such an OCCLUSION query reports (BINARY_OCCLUSION: 1)
const VISIBLE_SAMPLES: u64 = 0x1_0000;

fn is_occlusion(ty: D3D12_QUERY_TYPE) -> bool {
    ty == D3D12_QUERY_TYPE_OCCLUSION || ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION
}

type QueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);

/// command list #10 Reset(allocator, initial pipeline): updates allocator/pipeline and clears
/// per-recording state (queries, index buffer, depth bias, pending waits, written ranges)
pub(super) unsafe extern "system" fn reset_x(this: *mut c_void, alloc: usize, pso: usize) -> i32 {
    // A failed native Reset leaves the old recording alive. Do not discard its tracked state or
    // transfer its allocations until the driver accepts the new recording.
    let ext = *(this as *const usize);
    let resolved = super::pso_async::resolve_for_bind(pso);
    let hr = pc::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(this, PC_RESET)(this, alloc, resolved);
    real::preserve_vtable(this, ext);
    if hr < 0 {
        return hr;
    }

    let is_bundle = {
        let mut g = LISTS.lock().unwrap();
        let s = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
        s.allocator = alloc;
        s.bound_pso = pso;
        s.depth_bias = None;
        s.native_bias = DepthBias::ZERO;
        s.pso_dirty = false;
        s.using_fallback_pso = false;
        s.index_buffer = None;
        s.depth_bounds = None;
        s.occlusion = Occlusion::default();
        s.written_ranges.clear();
        s.clear_sparse_writes();
        s.clear_waits();
        s.cpu_writes.clear();
        s.bindings = Bindings::default();
        s.list_type == Some(D3D12_COMMAND_LIST_TYPE_BUNDLE)
    };
    super::bind::reset(this, alloc, pso);
    if !is_bundle && pso != 0 {
        if let Some((true, static_bias)) = device_x::base_depth_bias_info(pso) {
            if !static_bias.is_zero() && try_native_depth_bias(this, static_bias) {
                if let Some(s) = LISTS.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
                    s.native_bias = static_bias;
                }
            }
        }
    }
    hr
}

/// the D3D12_RESOURCE_STATES bits the PC knows
const PC_STATES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0xF_0000 | 0x20_0000 | 0x40_0000 | 0x80_0000 | 0x100_0000;

/// #26 ResourceBarrier(count, barriers): console barriers (D3D12_RESOURCE_BARRIER, 32 bytes) as
/// PC ones – state bits the PC does not know are dropped (logged: the console's own), a
/// transition left with nothing to do goes, aliasing barriers name placed textures
pub(super) unsafe extern "system" fn resource_barrier_x(this: *mut c_void, n: u32, barriers: *const [u32; 8]) {
    static N: AtomicU64 = AtomicU64::new(0);
    if barriers.is_null() || n == 0 {
        return;
    }
    let mut out: Vec<[u32; 8]> = Vec::with_capacity(n as usize + 1);
    // a UAV barrier for all resources is in `out` / a transition from or to UNORDERED_ACCESS
    let (mut any_uav, mut uav_transition) = (false, false);
    for k in 0..n as usize {
        let mut b = barriers.add(k).read_unaligned();
        let (ty, flags) = (b[0], b[1]);
        if super::present::trace_frame() && ty == 0 {
            let r = (b[2] as u64 | (b[3] as u64) << 32) as usize;
            if super::present::is_watched(r) {
                static W: AtomicU64 = AtomicU64::new(0);
                if W.fetch_add(1, Ordering::Relaxed) < 24 {
                    crate::xlog_info!("frame trace: ResourceBarrier on a watched target {r:#x}: subresource {:#x}, {:#x} → {:#x}, flags {flags:#x}", b[4], b[5], b[6]);
                }
            }
        }
        if ty > 2 || flags > 3 {
            if N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console barrier type {ty} flags {flags:#x} dropped ({:x?})", b);
            }
            continue;
        }
        if ty == 1 {
            // Keep the identity of textures that really share placed GPU memory.
            // Committed resources still need the wildcard form on the PC.
            for words in b[2..6].chunks_exact_mut(2) {
                let resource = (words[0] as u64 | (words[1] as u64) << 32) as usize;
                if resource != 0 && !super::vram::is_placed_texture(resource) {
                    words.fill(0);
                }
            }
        }
        if ty == 2 {
            // UAV (resource +8): translated shaders reach title memory through XCL's memory
            // window buffers – other resources over the same memory – which a UAV barrier for
            // the title's resource alone does not order (AMD drivers wait and flush for any UAV
            // barrier, NVIDIA's need not): a UAV barrier for all resources, once per batch
            b[2..4].fill(0);
            if any_uav {
                continue;
            }
            any_uav = true;
        }
        if ty == 0 {
            // transition: resource (+8), subresource (+16), before (+20), after (+24)
            let (before, after) = (b[5], b[6]);
            if (before | after) & !PC_STATES != 0 && N.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!("D3D12.X: ResourceBarrier: console state bits {:#x} → {:#x} dropped", before & !PC_STATES, after & !PC_STATES);
            }
            b[5] &= PC_STATES;
            b[6] &= PC_STATES;
            // VERTEX_AND_CONSTANT_BUFFER / INDEX_BUFFER are buffer states on the PC; the console
            // takes them for textures too (Minecraft moves a texture to a generic-read
            // combination) – the PC list then fails to close
            if (b[5] | b[6]) & BUFFER_ONLY_STATES != 0 && !is_buffer((b[2] as u64 | (b[3] as u64) << 32) as usize) {
                b[5] &= !BUFFER_ONLY_STATES;
                b[6] &= !BUFFER_ONLY_STATES;
            }
            if b[5] == b[6] {
                continue;
            }
            // UAV writes before (or after) the transition: the same holds for the window buffers
            uav_transition |= (b[5] | b[6]) & UNORDERED_ACCESS != 0;
        }
        out.push(b);
    }
    if uav_transition && !any_uav {
        out.push([2, 0, 0, 0, 0, 0, 0, 0]);
    }
    if !out.is_empty() {
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8])>(this, PC_RESOURCE_BARRIER)(this, out.len() as u32, out.as_ptr());
    }
}

/// D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | INDEX_BUFFER: valid for buffers only
const BUFFER_ONLY_STATES: u32 = 0x1 | 0x2;

/// D3D12_RESOURCE_STATE_UNORDERED_ACCESS
const UNORDERED_ACCESS: u32 = 0x8;

/// Whether `res` is a buffer (null / unknown: taken as one – its states are left alone).
unsafe fn is_buffer(res: usize) -> bool {
    let r = res as *mut c_void;
    ID3D12Resource::from_raw_borrowed(&r).is_none_or(|x| x.GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
}

/// DMA list #9 Close: as on the PC; a failure is logged (the title stops on it)
pub(super) unsafe extern "system" fn dma_close_x(this: *mut c_void) -> i32 {
    let ext = *(this as *const usize);
    let hr = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, 9)(this);
    real::preserve_vtable(this, ext);
    if hr < 0 {
        let ty = pc::<unsafe extern "system" fn(*mut c_void) -> i32>(this, PC_GET_TYPE)(this);
        crate::xlog_warn!("D3D12.X: DMA list Close → {:#x} (PC list type {ty})", hr as u32);
    }
    hr
}

/// a copy location with the offset of a buffer placed below its console address added
unsafe fn pc_location(l: *const D3D12_TEXTURE_COPY_LOCATION) -> Option<D3D12_TEXTURE_COPY_LOCATION> {
    let mut l = std::ptr::read(l.as_ref()?);
    if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        l.Anonymous.PlacedFootprint.Offset += real::buffer_delta(transmute_copy::<_, usize>(&l.pResource));
    }
    Some(l)
}

/// CopyBufferRegion(dst, dst offset, src, src offset, bytes) (PC #15; DMA list #11)
/// `XCL_TRACE_FRAME`: a GPU copy into title memory that a texture kept in step lies in (the
/// texture does not see GPU writes)
fn note_gpu_write(what: &str, va: usize, n: u64) {
    if !super::present::trace_frame() {
        return;
    }
    let hit = super::title_textures::overlapping(va, n as usize);
    if hit.is_empty() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k < 16 || k.is_power_of_two() {
        crate::xlog_info!("frame trace: {what} #{k} writes {n:#x} bytes at {va:#x} – into title textures {hit:x?}");
    }
}

pub(super) unsafe extern "system" fn copy_buffer_region_x(this: *mut c_void, dst: usize, dst_off: u64, src: usize, src_off: u64, n: u64) {
    if super::present::trace_frame() {
        if let Some(a) = real::title_address(dst) {
            note_gpu_write("CopyBufferRegion", a + dst_off as usize, n);
        }
    }
    pc::<unsafe extern "system" fn(*mut c_void, usize, u64, usize, u64, u64)>(this, PC_COPY_BUFFER_REGION)(this, dst, dst_off + real::buffer_delta(dst), src, src_off + real::buffer_delta(src), n)
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): copies out of it – that
/// object, or another at its title address (the first 16)
unsafe fn trace_watched_copy(what: &str, dst: usize, src: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let addr = |r: usize| real::title_address(r).or_else(|| super::title_textures::address(r));
    if !super::present::is_watched(src) && (addr(src).is_none() || addr(src) != addr(w)) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 16 {
        return;
    }
    if !super::present::is_presented(dst) {
        super::present::follow(dst);
    }
    let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0) });
    crate::xlog_info!(
        "frame trace: {what} out of a watched target {src:#x} {:?} into {dst:#x} {:?}{}",
        d(src),
        d(dst),
        if super::present::is_presented(dst) { " (presented)" } else { "" }
    );
}

/// CopyTextureRegion(dst, x, y, z, src, box) (PC #16; DMA list #12)
pub(super) unsafe extern "system" fn copy_texture_region_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX) {
    if let Some(d) = dst.as_ref() {
        let r: usize = transmute_copy(&d.pResource);
        super::present::note_use(r, super::present::USE_COPY);
        super::title_textures::note_gpu_write(r);
        if super::present::trace_frame() {
            if let Some(va) = super::title_textures::address(r) {
                static N: AtomicU64 = AtomicU64::new(0);
                let k = N.fetch_add(1, Ordering::Relaxed);
                if k < 8 || k.is_power_of_two() {
                    // the source: a buffer footprint – where its bytes are for the CPU, and whether
                    // there is anything there
                    let src_info = src.as_ref().map(|sl| {
                        let sr: usize = transmute_copy(&sl.pResource);
                        let rr = sr as *mut c_void;
                        let heap = ID3D12Resource::from_raw_borrowed(&rr).and_then(|x| {
                            let mut hp = D3D12_HEAP_PROPERTIES::default();
                            let mut hf = D3D12_HEAP_FLAGS::default();
                            x.GetHeapProperties(Some(&mut hp), Some(&mut hf)).ok().map(|_| hp.Type.0)
                        });
                        let (ty, fp) = (sl.Type.0, if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT { Some(sl.Anonymous.PlacedFootprint) } else { None });
                        let cpu = real::title_address(sr);
                        let nonzero = match (cpu, fp) {
                            (Some(a), Some(f)) => {
                                let at = a + f.Offset as usize;
                                let n = (f.Footprint.RowPitch as usize * 4).min(4096);
                                if crate::gpu::shader::readable(at, n) { Some(std::slice::from_raw_parts(at as *const u8, n).iter().filter(|&&b| b != 0).count()) } else { None }
                            }
                            _ => None,
                        };
                        format!("src {sr:#x} type {ty} heap {heap:?} title address {:?} delta {:#x} footprint {:?} nonzero in the first rows {nonzero:?}", cpu.map(|a| format!("{a:#x}")), real::buffer_delta(sr), fp.map(|f| (f.Offset, f.Footprint.Format.0, f.Footprint.Width, f.Footprint.Height, f.Footprint.RowPitch)))
                    });
                    crate::xlog_info!("frame trace: CopyTextureRegion #{k} into title texture {r:#x} (at {va:#x}), subresource {}; {src_info:?}", d.Anonymous.SubresourceIndex);
                }
            }
        }
    }
    if super::present::trace_frame() {
        if let (Some(d), Some(sl)) = (dst.as_ref(), src.as_ref()) {
            trace_watched_copy("CopyTextureRegion", transmute_copy(&d.pResource), transmute_copy(&sl.pResource));
            // footprints the PC's copy rules do not allow (row pitch: 256 bytes, offset: 512)
            if sl.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
                let f = sl.Anonymous.PlacedFootprint;
                if f.Footprint.RowPitch % 256 != 0 || f.Offset % 512 != 0 {
                    // each (format, pitch alignment, offset alignment) once
                    static SEEN: Mutex<Vec<(i32, u32, u64)>> = Mutex::new(Vec::new());
                    let key = (f.Footprint.Format.0, f.Footprint.RowPitch % 256, f.Offset % 512);
                    let new = {
                        let mut g = SEEN.lock().unwrap();
                        let n = !g.contains(&key) && g.len() < 24;
                        if n {
                            g.push(key);
                        }
                        n
                    };
                    if new {
                        crate::xlog_info!(
                            "frame trace: CopyTextureRegion from a footprint the PC does not take: offset {:#x}, format {}, {}x{}x{}, row pitch {} (box {:?}, to ({x}, {y}, {z}))",
                            f.Offset,
                            f.Footprint.Format.0,
                            f.Footprint.Width,
                            f.Footprint.Height,
                            f.Footprint.Depth,
                            f.Footprint.RowPitch,
                            bx.as_ref().map(|b| (b.left, b.top, b.right, b.bottom))
                        );
                    }
                }
            }
        }
    }
    let (Some(d), Some(s)) = (pc_location(dst), pc_location(src)) else { return };
    pc::<unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX)>(this, PC_COPY_TEXTURE_REGION)(this, &d, x, y, z, &s, bx);
    std::mem::forget((d, s)); // borrowed resource pointers
}

/// command list #83 CopyTextureRegionX(dst, x, y, z, src, box, flags): CopyTextureRegion with
/// console copy flags (the title's texture streaming calls it where the DMA list gets #12)
pub(super) unsafe extern "system" fn copy_texture_region_flags_x(this: *mut c_void, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32, src: *const D3D12_TEXTURE_COPY_LOCATION, bx: *const D3D12_BOX, _flags: u32) {
    copy_texture_region_x(this, dst, x, y, z, src, bx)
}

const PC_CLEAR_DEPTH_STENCIL_VIEW: usize = 47;

/// command list #51 ClearDepthStencilView(dsv, flags, depth, stencil, rects, rect list) (PC #47)
pub(super) unsafe extern "system" fn clear_depth_stencil_x(this: *mut c_void, dsv: usize, flags: u32, depth: f32, stencil: u8, n: u32, rects: *const windows::Win32::Foundation::RECT) {
    if super::present::trace_frame() {
        static N: AtomicU64 = AtomicU64::new(0);
        static W: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed);
        let watched = WATCHED_DSVS.lock().unwrap().contains(&dsv);
        if k < 8 || k.is_power_of_two() || (watched && W.fetch_add(1, Ordering::Relaxed) < 12) {
            crate::xlog_info!(
                "frame trace: ClearDepthStencilView #{k} {dsv:#x}{}: flags {flags:#x} (1 depth, 2 stencil), depth {depth}, stencil {stencil}, {n} rects",
                if watched { " (bound by a draw reading a watched target)" } else { "" }
            );
        }
    }
    pc::<unsafe extern "system" fn(*mut c_void, usize, u32, f32, u8, u32, *const windows::Win32::Foundation::RECT)>(this, PC_CLEAR_DEPTH_STENCIL_VIEW)(this, dsv, flags, depth, stencil, n, rects)
}

const PC_RESOLVE_SUBRESOURCE: usize = 19;
const PC_RESOLVE_SUBRESOURCE_REGION: usize = 64;

/// `XCL_TRACE_FRAME`: a resolve (first 12 and powers of two) – what goes where
unsafe fn trace_resolve(what: &str, dst: usize, src: usize, format: i32, extra: &str) {
    if !super::present::trace_frame() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if k >= 12 && !k.is_power_of_two() {
        return;
    }
    let desc = |r: usize| {
        let p = r as *mut c_void;
        ID3D12Resource::from_raw_borrowed(&p).map(|x| {
            let d = x.GetDesc();
            (d.Width, d.Height, d.Format.0, d.SampleDesc.Count)
        })
    };
    crate::xlog_info!(
        "frame trace: {what} #{k} {src:#x} {:?} → {dst:#x} {:?}{} format {format}{extra}",
        desc(src),
        desc(dst),
        if super::present::is_presented(dst) { " (presented)" } else { "" }
    );
}

/// command list #19 ResolveSubresource(dst, dst sub, src, src sub, format) (PC #19)
pub(super) unsafe extern "system" fn resolve_subresource_x(this: *mut c_void, dst: usize, dst_sub: u32, src: usize, src_sub: u32, format: i32) {
    super::present::note_use(dst, super::present::USE_COPY);
    trace_resolve("ResolveSubresource", dst, src, format, "");
    pc::<unsafe extern "system" fn(*mut c_void, usize, u32, usize, u32, i32)>(this, PC_RESOLVE_SUBRESOURCE)(this, dst, dst_sub, src, src_sub, format)
}

/// command list #108 ResolveSubresourceRegion(dst, dst sub, x, y, src, src sub, rect, format,
/// mode) (PC #64)
#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "system" fn resolve_subresource_region_x(this: *mut c_void, dst: usize, dst_sub: u32, x: u32, y: u32, src: usize, src_sub: u32, rect: *const windows::Win32::Foundation::RECT, format: i32, mode: i32) {
    super::present::note_use(dst, super::present::USE_COPY);
    if super::present::trace_frame() {
        trace_resolve("ResolveSubresourceRegion", dst, src, format, &format!(" mode {mode} at ({x}, {y}) rect {:?}", rect.as_ref().map(|r| (r.left, r.top, r.right, r.bottom))));
    }
    pc::<unsafe extern "system" fn(*mut c_void, usize, u32, u32, u32, usize, u32, *const windows::Win32::Foundation::RECT, i32, i32)>(this, PC_RESOLVE_SUBRESOURCE_REGION)(this, dst, dst_sub, x, y, src, src_sub, rect, format, mode)
}

/// CopyResource(dst, src) (PC #17; DMA list #13): buffers placed below their console address
/// are copied as their own bytes (the PC resources also hold what lies before them)
pub(super) unsafe extern "system" fn copy_resource_x(this: *mut c_void, dst: usize, src: usize) {
    super::present::note_use(dst, super::present::USE_COPY);
    super::title_textures::note_gpu_write(dst);
    if super::present::trace_frame() {
        trace_watched_copy("CopyResource", dst, src);
    }
    let (dd, sd) = (real::buffer_delta(dst), real::buffer_delta(src));
    if dd == 0 && sd == 0 {
        return pc::<unsafe extern "system" fn(*mut c_void, usize, usize)>(this, PC_COPY_RESOURCE)(this, dst, src);
    }
    let n = real::console_buffer(dst).or(real::console_buffer(src)).map_or(0, |b| b.1 as u64);
    copy_buffer_region_x(this, dst, 0, src, 0, n);
}

/// command list #28 BeginQuery(heap, type, index) (PC #52)
pub(super) unsafe extern "system" fn begin_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    static N: AtomicU64 = AtomicU64::new(0);
    if is_occlusion(ty) {
        let outer = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .begin((heap, index), ty);
        if let Some(((h, i), oty)) = outer {
            if once(&N, 1) {
                crate::xlog_info!("D3D12.X: nested occlusion query – the outer one ends there and reports visible, the inner one counts");
            }
            pc::<QueryFn>(this, PC_END_QUERY)(this, h, oty, i);
        }
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, index));
        }
    }
    pc::<QueryFn>(this, PC_BEGIN_QUERY)(this, heap, ty, index);
}

/// command list #29 EndQuery(heap, type, index) (PC #53)
pub(super) unsafe extern "system" fn end_query_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, index: u32) {
    if is_occlusion(ty) {
        let forward = LISTS
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .entry(this as usize)
            .or_default()
            .occlusion
            .end((heap, index));
        if !forward {
            let v = if ty == D3D12_QUERY_TYPE_BINARY_OCCLUSION { 1 } else { VISIBLE_SAMPLES };
            VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, index), v);
            return;
        }
    }
    pc::<QueryFn>(this, PC_END_QUERY)(this, heap, ty, index);
}

/// command list #30 ResolveQueryData(heap, type, first, count, destination, offset) (PC #54):
/// outer occlusion queries ended early are written over the resolved data (after it, in COPY_DEST as
/// ResolveQueryData wants the destination). Buffers placed below their console address add
/// `buffer_delta(dest)` to the destination offset.
pub(super) unsafe extern "system" fn resolve_query_data_x(this: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, first: u32, count: u32, dest: usize, offset: u64) {
    let pc_offset = offset.saturating_add(real::buffer_delta(dest));
    type F = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);
    pc::<F>(this, PC_RESOLVE_QUERY_DATA)(this, heap, ty, first, count, dest, pc_offset);
    if !is_occlusion(ty) {
        return;
    }
    let visible: Vec<(u32, u64)> = match VISIBLE.lock().unwrap().as_ref() {
        Some(m) if !m.is_empty() => (first..first.saturating_add(count)).filter_map(|i| m.get(&(heap, i)).map(|&v| (i, v))).collect(),
        _ => return,
    };
    let dest = dest as *mut c_void;
    let Some(res) = ID3D12Resource::from_raw_borrowed(&dest).filter(|_| !visible.is_empty()) else { return };
    let at = res.GetGPUVirtualAddress().saturating_add(pc_offset);
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = visible
        .iter()
        .flat_map(|&(i, v)| {
            let a = at + 8 * (i - first) as u64;
            [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a, Value: v as u32 }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a + 4, Value: (v >> 32) as u32 }]
        })
        .collect();
    let modes = vec![D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; params.len()];
    type W = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
    pc::<W>(this, PC_WRITE_BUFFER_IMMEDIATE)(this, params.len() as u32, params.as_ptr(), modes.as_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    static RESET_TEST_HR: AtomicU32 = AtomicU32::new(0);
    static RESET_TEST_REPLACEMENT_VT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn reset_test_native(this: *mut c_void, _alloc: usize, _pso: usize) -> i32 {
        *(this as *mut usize) = RESET_TEST_REPLACEMENT_VT.load(Ordering::SeqCst) as usize;
        RESET_TEST_HR.load(Ordering::SeqCst) as i32
    }

    #[test]
    fn nested_occlusion_queries_end_the_outer_one_early() {
        let ty = D3D12_QUERY_TYPE_OCCLUSION;
        let mut o = Occlusion::default();
        assert_eq!(o.begin((1, 0), ty), None);
        assert_eq!(o.begin((1, 1), ty), Some(((1, 0), ty)), "the outer one ends on the PC");
        assert!(o.end((1, 1)), "inner: a real query");
        // more per-object queries inside the same outer one
        assert_eq!(o.begin((1, 2), ty), None);
        assert!(o.end((1, 2)));
        assert!(!o.end((1, 0)), "outer: already ended");
        // overlapping, not nested: A, B, end A, end B
        assert_eq!(o.begin((1, 3), ty), None);
        assert_eq!(o.begin((1, 4), ty), Some(((1, 3), ty)));
        assert!(!o.end((1, 3)) && o.end((1, 4)));
        assert!(o.active.is_none() && o.ended_early.is_empty());
        assert!(o.end((1, 9)), "a query not begun here: the PC's call");
        // kickoff ends active query early so EndQuery later resolves it as visible
        assert_eq!(o.begin((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION), None);
        assert_eq!(o.end_active_early(), Some(((1, 5), D3D12_QUERY_TYPE_BINARY_OCCLUSION)));
        assert!(!o.end((1, 5)));
    }

    #[test]
    fn reset_commits_state_only_after_native_success_and_always_restores_vtable() {
        let mut native = vec![0usize; PC_RESET + 1];
        native[PC_RESET] = reset_test_native as usize;
        let extended = [native.as_ptr() as usize, 0usize];
        let ext = unsafe { extended.as_ptr().add(1) as usize };
        let replacement = 0x1234_5678usize;
        let mut object = [ext];
        let list = object.as_mut_ptr() as *mut c_void;
        let key = list as usize;

        unsafe { on_create(list, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT) };
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            s.allocator = 0x111;
            s.bound_pso = 0x222;
            s.depth_bias = Some(DepthBias::new(-3.5, 0.25, 2.0));
            s.native_bias = DepthBias::new(-4.5, 0.5, 3.0);
            s.pso_dirty = true;
            s.using_fallback_pso = true;
            s.index_buffer = Some(D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: 0x5000,
                SizeInBytes: 12,
                Format: DXGI_FORMAT_R32_UINT,
            });
            s.depth_bounds = Some((0.25, 0.75));
            s.written_ranges.push((0x6000, 0x6010));
        }
        super::super::bind::seed_reset_state_for_test(list, 0x111, 0x222, vec![0x4444]);
        RESET_TEST_REPLACEMENT_VT.store(replacement as u64, Ordering::SeqCst);

        RESET_TEST_HR.store(0x8000_4005, Ordering::SeqCst);
        assert!(unsafe { reset_x(list, 0x333, 0x555) } < 0);
        assert_eq!(object[0], ext);
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!((s.allocator, s.bound_pso), (0x111, 0x222));
            assert_eq!(s.depth_bias, Some(DepthBias::new(-3.5, 0.25, 2.0)));
            assert_eq!(s.native_bias, DepthBias::new(-4.5, 0.5, 3.0));
            assert!(s.pso_dirty && s.using_fallback_pso);
            assert_eq!(s.index_buffer.as_ref().unwrap().BufferLocation, 0x5000);
            assert_eq!(s.depth_bounds, Some((0.25, 0.75)));
            assert_eq!(s.written_ranges, vec![(0x6000, 0x6010)]);
        }
        assert_eq!(
            super::super::bind::reset_state_for_test(list),
            (0x111, 0x222, vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![0x4444])
        );
        assert!(super::super::console_list::take_pending_for_test(0x111).is_empty());

        RESET_TEST_HR.store(0, Ordering::SeqCst);
        assert_eq!(unsafe { reset_x(list, 0x333, 0x555) }, 0);
        assert_eq!(object[0], ext);
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!((s.allocator, s.bound_pso), (0x333, 0x555));
            assert!(s.depth_bias.is_none() && s.index_buffer.is_none() && s.depth_bounds.is_none());
            assert!(!s.pso_dirty && !s.using_fallback_pso && s.written_ranges.is_empty());
        }
        assert_eq!(
            super::super::bind::reset_state_for_test(list),
            (0x333, 0x555, Vec::new(), Vec::new(), Vec::new())
        );
        assert_eq!(super::super::console_list::take_pending_for_test(0x111), vec![0x4444]);

        LISTS.lock().unwrap().as_mut().unwrap().remove(&key);
        super::super::bind::remove_state_for_test(list);
    }

    #[test]
    fn fill_patterns_repeat_the_value() {
        assert_eq!(pattern(&0x1122_3344u32.to_le_bytes())[..8], [0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11]);
        assert_eq!(pattern(&[7u8; 16]), [7u8; 16]);
    }

    #[test]
    fn wait_comparison_ops_cover_all_d3d12_funcs() {
        assert_eq!(compare_u64(0, 1, 1), None);
        assert_eq!(compare_u64(9, 1, 1), None);
        assert_eq!(compare_u64(1, 5, 5), Some(false)); // NEVER
        assert_eq!(compare_u64(2, 4, 5), Some(true));  // LESS
        assert_eq!(compare_u64(2, 5, 5), Some(false));
        assert_eq!(compare_u64(3, 5, 5), Some(true));  // EQUAL
        assert_eq!(compare_u64(3, 4, 5), Some(false));
        assert_eq!(compare_u64(4, 5, 5), Some(true));  // LESS_EQUAL
        assert_eq!(compare_u64(4, 6, 5), Some(false));
        assert_eq!(compare_u64(5, 6, 5), Some(true));  // GREATER
        assert_eq!(compare_u64(5, 5, 5), Some(false));
        assert_eq!(compare_u64(6, 4, 5), Some(true));  // NOT_EQUAL
        assert_eq!(compare_u64(6, 5, 5), Some(false));
        assert_eq!(compare_u64(7, 5, 5), Some(true));  // GREATER_EQUAL
        assert_eq!(compare_u64(7, 4, 5), Some(false));
        assert_eq!(compare_u64(8, 0, 1), Some(true));  // ALWAYS
    }

    #[test]
    fn pending_wait_32_and_64_bit_with_mask_and_cross_thread_signal() {
        let cell32 = Arc::new(AtomicU32::new(0xDEAD_0000));
        let w32 = PendingWait {
            va: Arc::as_ptr(&cell32) as u64,
            width: 4,
            comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
            value: 0x0000_005A,
            mask: 0x0000_00FF,
        };
        unsafe {
            assert!(!w32.is_satisfied());
            let writer = cell32.clone();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                writer.store(0xBEEF_005A, Ordering::Release);
            });
            assert!(w32.wait_until_satisfied(Duration::from_secs(1)));
            t.join().unwrap();
        }

        let cell64 = Arc::new(AtomicU64::new(0x0000_0001_0000_0000));
        let w64 = PendingWait {
            va: Arc::as_ptr(&cell64) as u64,
            width: 8,
            comparison: D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            value: 0x0000_0002_0000_0000,
            mask: u64::MAX,
        };
        unsafe {
            assert!(!w64.is_satisfied());
            // bounded timeout returns false without hanging
            assert!(!w64.wait_until_satisfied(Duration::from_millis(5)));
            cell64.store(0x0000_0002_0000_0001, Ordering::Release);
            assert!(w64.is_satisfied());
        }
    }

    #[test]
    fn written_ranges_detect_same_list_overlap() {
        let mut s = ListState::default();
        s.note_write(0x1000, 8);
        assert!(s.has_prior_write(0x1000, 4));
        assert!(s.has_prior_write(0x1004, 4));
        assert!(s.has_prior_write(0x0FFC, 8));
        assert!(!s.has_prior_write(0x1008, 4));
        assert!(!s.has_prior_write(0x0FF8, 8));
    }

    #[test]
    fn execute_with_waits_splits_batches_before_unsatisfied_list() {
        // Use synthetic non-zero list pointers distinct from real COM objects.
        let l0 = 0x7000_0001usize as *mut c_void;
        let l1 = 0x7000_0002usize as *mut c_void;
        let l2 = 0x7000_0003usize as *mut c_void;
        let cell = Box::new(AtomicU32::new(0));
        let va = &*cell as *const AtomicU32 as u64;

        unsafe {
            on_create(l0, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l1, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_create(l2, 0x100, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&(l1 as usize)).unwrap();
            s.push_wait(PendingWait {
                va,
                width: 4,
                comparison: D3D12_COMPARISON_FUNC_EQUAL.0 as u32,
                value: 42,
                mask: u32::MAX as u64,
            });
        }

        let mut batches: Vec<Vec<usize>> = Vec::new();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
                // Simulate l0's GPU execution satisfying l1's wait!
                if chunk.contains(&l0) {
                    cell.store(42, Ordering::Release);
                }
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize], vec![l1 as usize, l2 as usize]]);

        // Now that cell == 42 is already satisfied, the whole batch submits in one call.
        batches.clear();
        unsafe {
            execute_with_waits(&[l0, l1, l2], |chunk| {
                batches.push(chunk.iter().map(|&p| p as usize).collect());
            });
        }
        assert_eq!(batches, vec![vec![l0 as usize, l1 as usize, l2 as usize]]);

        on_release(l0 as usize);
        on_release(l1 as usize);
        on_release(l2 as usize);
    }

    #[test]
    fn queue_picking_prefers_last_queue_and_respects_release() {
        let dev = 0x8000_0001usize;
        let q1 = 0x8000_0010usize;
        let q2 = 0x8000_0020usize;
        let q_copy = 0x8000_0030usize;
        on_queue_create(q1, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q2, dev, D3D12_COMMAND_LIST_TYPE_DIRECT);
        on_queue_create(q_copy, dev, D3D12_COMMAND_LIST_TYPE_COPY);

        // Most recently created matching queue wins when no last_queue
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_COPY), Some(q_copy));
        // Explicit last_queue wins when still live
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q1));
        // Releasing q1 falls back to q2
        on_queue_release(q1);
        assert_eq!(pick_queue(Some(q1), dev, D3D12_COMMAND_LIST_TYPE_DIRECT), Some(q2));
        on_queue_release(q2);
        on_queue_release(q_copy);
        assert_eq!(pick_queue(None, dev, D3D12_COMMAND_LIST_TYPE_DIRECT), None);
    }

    #[test]
    fn depth_bias_state_tracks_fallback_and_clear() {
        let l = 0x9000_0001usize as *mut c_void;
        let key = l as usize;
        unsafe {
            on_create(l, 0x100, 0xAAA, D3D12_COMMAND_LIST_TYPE_DIRECT);
            on_set_pipeline_state(l, 0xBBB);
        }
        {
            let mut g = LISTS.lock().unwrap();
            let s = g.as_mut().unwrap().get_mut(&key).unwrap();
            assert_eq!(s.bound_pso, 0xBBB);
            assert!(!s.pso_dirty);
            s.depth_bias = Some(DepthBias::new(-10.0, 0.0, 1.0));
            s.pso_dirty = true;
        }
        // Switching to a non-dynamic PSO while depth bias is active keeps pso_dirty for fallback
        unsafe { on_set_pipeline_state(l, 0xCCC) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xCCC);
            assert!(s.pso_dirty);
        }
        // ClearState resets depth bias, native_bias, and dirty flag
        unsafe { on_clear_state(l, 0xDDD) };
        {
            let g = LISTS.lock().unwrap();
            let s = g.as_ref().unwrap().get(&key).unwrap();
            assert_eq!(s.bound_pso, 0xDDD);
            assert!(s.depth_bias.is_none());
            assert_eq!(s.native_bias, DepthBias::ZERO);
            assert!(!s.pso_dirty);
        }
        on_release(key);
    }

    #[test]
    fn depth_bias_native_list9_and_bundle_fallback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let supported = device_x::probe_dynamic_depth_bias(&dev);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            let has_list9 = list.cast::<ID3D12GraphicsCommandList9>().is_ok();
            let raw = list.as_raw();
            let key = raw as usize;

            let pso_dyn_zero = 0x9100_0001usize;
            let pso_dyn_static = 0x9100_0002usize;
            let pso_static_only = 0x9100_0003usize;
            let static_bias = DepthBias::new(-500.0, 0.0, 0.0);
            device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
            device_x::register_test_base(pso_dyn_static, true, static_bias);
            device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

            let cache_before = device_x::depth_bias_cache_len();
            on_create(raw, alloc.as_raw() as usize, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_DIRECT);

            if supported && has_list9 {
                // Fractional f32 depth bias takes the native ID3D12GraphicsCommandList9 path
                // without rounding to i32 or dirtying the patched-PSO fallback cache.
                set_depth_bias_x(raw, -125.75, 0.05, 1.25);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.depth_bias, Some(DepthBias::new(-125.75, 0.05, 1.25)));
                    assert_eq!(s.native_bias, DepthBias::new(-125.75, 0.05, 1.25));
                    assert_eq!(s.native_bias.depth_bias(), -125.75);
                    assert!(!s.pso_dirty);
                }
                assert_eq!(device_x::depth_bias_cache_len(), cache_before);

                // Resetting depth bias to 0 updates native_bias to ZERO without touching fallback cache.
                set_depth_bias_x(raw, 0.0, 0.0, 0.0);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert!(s.depth_bias.is_none());
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }

                // Binding a dynamic PSO with non-zero static bias syncs native_bias to its static bias.
                on_set_pipeline_state(raw, pso_dyn_static);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, static_bias);
                    assert!(!s.pso_dirty);
                }

                // Switching back to a zero-static-bias dynamic PSO restores native_bias to ZERO.
                on_set_pipeline_state(raw, pso_dyn_zero);
                {
                    let g = LISTS.lock().unwrap();
                    let s = g.as_ref().unwrap().get(&key).unwrap();
                    assert_eq!(s.native_bias, DepthBias::ZERO);
                    assert!(!s.pso_dirty);
                }
            }

            // Non-dynamic PSO must fall back to the patched-PSO cache (pso_dirty = true).
            on_set_pipeline_state(raw, pso_static_only);
            set_depth_bias_x(raw, -50.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert_eq!(s.depth_bias, Some(DepthBias::new(-50.0, 0.0, 0.0)));
                assert!(s.pso_dirty);
            }
            on_release(key);

            // Bundle lists cannot call ID3D12GraphicsCommandList9::RSSetDepthBias and must fall back.
            let b_raw = 0x9100_00F0usize as *mut c_void;
            on_create(b_raw, 0x100, pso_dyn_zero, D3D12_COMMAND_LIST_TYPE_BUNDLE);
            set_depth_bias_x(b_raw, -75.0, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&(b_raw as usize)).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::ZERO);
            }
            on_release(b_raw as usize);

            device_x::forget_base(pso_dyn_zero);
            device_x::forget_base(pso_dyn_static);
            device_x::forget_base(pso_static_only);
        }
    }

    #[test]
    fn depth_bias_native_path_exact_f32_and_list9_fallback() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Verify PC vtable slot of ID3D12GraphicsCommandList9::RSSetDepthBias is in layout table.
        let cl9 = super::super::layout::find("ID3D12GraphicsCommandList9").expect("ID3D12GraphicsCommandList9 in layout");
        assert_eq!(super::super::layout::slot(cl9, "RSSetDepthBias"), Some(82));

        let calls: Rc<RefCell<Vec<(usize, f32, f32, f32)>>> = Rc::new(RefCell::new(Vec::new()));
        let allow = Rc::new(std::cell::Cell::new(true));
        let calls_cb = calls.clone();
        let allow_cb = allow.clone();
        TEST_NATIVE_BIAS.with(|h| {
            *h.borrow_mut() = Some(Box::new(move |list, bias| {
                if !allow_cb.get() {
                    return false;
                }
                calls_cb.borrow_mut().push((list, bias.depth_bias(), bias.depth_bias_clamp(), bias.slope_scaled_depth_bias()));
                true
            }));
        });

        let l = 0x9200_0001usize as *mut c_void;
        let key = l as usize;
        let pso_dyn_zero = 0x9200_0010usize;
        let pso_dyn_static = 0x9200_0011usize;
        let pso_static_only = 0x9200_0012usize;
        device_x::register_test_base(pso_dyn_zero, true, DepthBias::ZERO);
        device_x::register_test_base(pso_dyn_static, true, DepthBias::new(-500.0, 0.25, 2.0));
        device_x::register_test_base(pso_static_only, false, DepthBias::ZERO);

        let cache_before = device_x::depth_bias_cache_len();
        unsafe {
            // 1. Initial PSO with baked static bias seeds native RSSetDepthBias on CreateCommandList.
            on_create(l, 0x100, pso_dyn_static, D3D12_COMMAND_LIST_TYPE_DIRECT);
            assert_eq!(*calls.borrow(), vec![(key, -500.0, 0.25, 2.0)]);

            // 2. Switching to zero-static-bias dynamic PSO resets native RSSetDepthBias to (0, 0, 0).
            on_set_pipeline_state(l, pso_dyn_zero);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));

            // 3. Fractional f32 override is forwarded verbatim (no i32 rounding, no fallback PSO cache).
            set_depth_bias_x(l, -1000.5, 0.05, 1.5);
            assert_eq!(calls.borrow().last(), Some(&(key, -1000.5, 0.05, 1.5)));
            assert_eq!(device_x::depth_bias_cache_len(), cache_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(!s.pso_dirty);
                assert_eq!(s.native_bias, DepthBias::new(-1000.5, 0.05, 1.5));
            }

            // 4. Switching between dynamic PSOs while override is active keeps the override without redundant calls.
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().len(), n_before);

            // 5. Resetting to (0, 0, 0) sets native bias to (0, 0, 0); ClearState with static-bias PSO re-seeds it.
            set_depth_bias_x(l, 0.0, 0.0, 0.0);
            assert_eq!(calls.borrow().last(), Some(&(key, 0.0, 0.0, 0.0)));
            on_clear_state(l, pso_dyn_static);
            assert_eq!(calls.borrow().last(), Some(&(key, -500.0, 0.25, 2.0)));

            // 6. If ID3D12GraphicsCommandList9 is unavailable, set_depth_bias_x falls back cleanly to pso_dirty.
            on_set_pipeline_state(l, pso_dyn_zero);
            allow.set(false);
            set_depth_bias_x(l, -42.5, 0.0, 0.0);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
                assert_eq!(s.depth_bias, Some(DepthBias::new(-42.5, 0.0, 0.0)));
            }

            // 7. Non-dynamic PSO always uses the patched-PSO fallback and never invokes RSSetDepthBias.
            allow.set(true);
            let n_before = calls.borrow().len();
            on_set_pipeline_state(l, pso_static_only);
            set_depth_bias_x(l, -10.0, 0.0, 0.0);
            assert_eq!(calls.borrow().len(), n_before);
            {
                let g = LISTS.lock().unwrap();
                let s = g.as_ref().unwrap().get(&key).unwrap();
                assert!(s.pso_dirty);
            }
        }

        TEST_NATIVE_BIAS.with(|h| *h.borrow_mut() = None);
        on_release(key);
        device_x::forget_base(pso_dyn_zero);
        device_x::forget_base(pso_dyn_static);
        device_x::forget_base(pso_static_only);
    }

    #[test]
    fn resolve_query_data_applies_buffer_delta_to_pc_resolve_and_occlusion_writeback() {
        use std::cell::RefCell;

        thread_local! {
            static RESOLVED: RefCell<Vec<(usize, u32, u32, usize, u64)>> = const { RefCell::new(Vec::new()) };
            static WRITES: RefCell<Vec<(u64, u32)>> = const { RefCell::new(Vec::new()) };
        }

        unsafe extern "system" fn fake_resolve(
            _this: *mut c_void,
            heap: usize,
            _ty: D3D12_QUERY_TYPE,
            first: u32,
            count: u32,
            dest: usize,
            offset: u64,
        ) {
            RESOLVED.with(|r| r.borrow_mut().push((heap, first, count, dest, offset)));
        }

        unsafe extern "system" fn fake_write_immediate(
            _this: *mut c_void,
            count: u32,
            params: *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER,
            _modes: *const D3D12_WRITEBUFFERIMMEDIATE_MODE,
        ) {
            let slice = std::slice::from_raw_parts(params, count as usize);
            WRITES.with(|w| {
                let mut w = w.borrow_mut();
                for p in slice {
                    w.push((p.Dest, p.Value));
                }
            });
        }

        const BASE_GPU_VA: u64 = 0x1_0000_0000;
        unsafe extern "system" fn fake_gpu_va(_this: *mut c_void) -> u64 {
            BASE_GPU_VA
        }

        #[repr(C)]
        struct FakeCom {
            vt: *const usize,
        }

        let mut orig_list_vt = [0usize; 70];
        orig_list_vt[PC_RESOLVE_QUERY_DATA] = fake_resolve as usize;
        orig_list_vt[PC_WRITE_BUFFER_IMMEDIATE] = fake_write_immediate as usize;
        // Adopted objects store `orig` at `ext[-1]`.
        let ext_list_vt = [orig_list_vt.as_ptr() as usize, 0usize];
        let mut fake_list = FakeCom {
            // SAFETY: `ext_list_vt` has 2 elements; `.add(1)` points at `ext[0]` with `ext[-1] = orig`.
            vt: unsafe { ext_list_vt.as_ptr().add(1) },
        };

        let mut res_vt = [0usize; 16];
        res_vt[11] = fake_gpu_va as usize;
        let mut fake_res = FakeCom { vt: res_vt.as_ptr() };

        let heap = 0x5555_0001usize;
        let dest = &mut fake_res as *mut FakeCom as usize;
        let delta = 0x3400u64;
        let title_offset = 0x100u64;

        real::set_test_buffer_delta(dest, Some(delta));
        VISIBLE.lock().unwrap().get_or_insert_with(HashMap::new).insert((heap, 2), VISIBLE_SAMPLES);
        RESOLVED.with(|r| r.borrow_mut().clear());
        WRITES.with(|w| w.borrow_mut().clear());

        unsafe {
            resolve_query_data_x(
                &mut fake_list as *mut FakeCom as *mut c_void,
                heap,
                D3D12_QUERY_TYPE_OCCLUSION,
                1,
                3,
                dest,
                title_offset,
            );
        }

        real::set_test_buffer_delta(dest, None);
        if let Some(m) = VISIBLE.lock().unwrap().as_mut() {
            m.remove(&(heap, 2));
        }

        let expected_off = title_offset + delta;
        RESOLVED.with(|r| {
            assert_eq!(*r.borrow(), vec![(heap, 1, 3, dest, expected_off)]);
        });
        WRITES.with(|w| {
            let expected_va = BASE_GPU_VA + expected_off + 8; // query index 2 = slot (2 - 1) = +8 bytes
            assert_eq!(
                *w.borrow(),
                vec![
                    (expected_va, VISIBLE_SAMPLES as u32),
                    (expected_va + 4, (VISIBLE_SAMPLES >> 32) as u32),
                ]
            );
        });
    }

    #[test]
    fn draw_indexed_x_routes_through_bind_and_restores_index_buffer() {
        #[repr(C)]
        struct FakeDrawList {
            vt: *const usize,
            refs: AtomicUsize,
            draws: Mutex<Vec<(u32, u32, u32, i32, u32)>>,
            index_buffers: Mutex<Vec<Option<D3D12_INDEX_BUFFER_VIEW>>>,
        }

        unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeDrawList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
            (*(this as *const FakeDrawList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn fake_draw_indexed(this: *mut c_void, ic: u32, inst: u32, si: u32, bv: i32, sinst: u32) {
            (*(this as *const FakeDrawList)).draws.lock().unwrap().push((ic, inst, si, bv, sinst));
        }
        unsafe extern "system" fn fake_set_ib(this: *mut c_void, view: *const D3D12_INDEX_BUFFER_VIEW) {
            let v = view.as_ref().copied();
            (*(this as *const FakeDrawList)).index_buffers.lock().unwrap().push(v);
        }
        unsafe extern "system" fn fake_noop_ptr(_this: *mut c_void, _p: *mut c_void) {}
        unsafe extern "system" fn fake_noop_uav(_this: *mut c_void, _slot: u32, _va: u64) {}

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[13] = fake_draw_indexed as usize;
        orig_vt[25] = fake_noop_ptr as usize; // SetPipelineState
        orig_vt[30] = fake_noop_ptr as usize; // SetGraphicsRootSignature
        orig_vt[42] = fake_noop_uav as usize; // SetGraphicsRootUnorderedAccessView
        orig_vt[43] = fake_set_ib as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeDrawList {
            // SAFETY: `ext_vt` has 65 entries so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            draws: Mutex::new(Vec::new()),
            index_buffers: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakeDrawList as *mut c_void;

        let orig_ib = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: 0x0000_0001_0000_1000,
            SizeInBytes: 64,
            Format: DXGI_FORMAT_R16_UINT,
        };

        unsafe {
            ia_set_index_buffer_x(list_ptr, &orig_ib);
            fake.index_buffers.lock().unwrap().clear();

            // 1. Untranslated (PC) PSO: routes through bind::draw_indexed_instanced -> DrawIndexedInstanced,
            // records DRAW_ARGS, binds temp IB, then restores orig_ib.
            super::super::bind::reset(list_ptr, 0x7400_0100, 0);
            let before = super::super::bind::counts_snapshot();
            draw_indexed_x(list_ptr, 0x0000_0001_0000_2000, DXGI_FORMAT_R16_UINT, 6, 0);
            let after = super::super::bind::counts_snapshot();
            assert_eq!(super::super::bind::last_draw_args(), (true, 6, 1, 0, 0));
            assert!(after[2] > before[2], "DRAWS_PC must increment via bind::draw_indexed_instanced");
            assert_eq!(*fake.draws.lock().unwrap(), vec![(6, 1, 0, 0, 0)]);
            let ibs = std::mem::take(&mut *fake.index_buffers.lock().unwrap());
            assert_eq!(ibs.len(), 2);
            assert_eq!(ibs[0].unwrap().BufferLocation, 0x0000_0001_0000_2000);
            assert_eq!(ibs[0].unwrap().SizeInBytes, 12);
            assert_eq!(ibs[0].unwrap().Format, DXGI_FORMAT_R16_UINT);
            assert_eq!(ibs[1].unwrap().BufferLocation, orig_ib.BufferLocation);
            assert_eq!(ibs[1].unwrap().SizeInBytes, orig_ib.SizeInBytes);

            // 2. Translated graphics PSO: enters bind::graphics_bound translated path (DRAWS or
            // DRAWS_SKIPPED, NOT DRAWS_PC) instead of bypassing bind.rs directly to PC slot 13.
            let translated_pso = 0x7400_0055usize;
            real::register_translated_graphics_for_test(translated_pso);
            super::super::bind::reset(list_ptr, 0x7400_0100, translated_pso);
            let before = super::super::bind::counts_snapshot();
            draw_indexed_x(list_ptr, 0x0000_0001_0000_3000, DXGI_FORMAT_R32_UINT, 12, 0);
            let after = super::super::bind::counts_snapshot();
            assert_eq!(super::super::bind::last_draw_args(), (true, 12, 1, 0, 0));
            assert_eq!(
                (after[0] + after[1]) - (before[0] + before[1]),
                1,
                "translated graphics PSO must go through bind::graphics_bound (DRAWS or DRAWS_SKIPPED)"
            );
            assert_eq!(after[2], before[2], "translated graphics PSO must not take the PC draw path");
            let ibs = std::mem::take(&mut *fake.index_buffers.lock().unwrap());
            if after[0] > before[0] {
                assert_eq!(ibs.len(), 2);
                assert_eq!(ibs[0].unwrap().SizeInBytes, 48);
                assert_eq!(ibs[0].unwrap().Format, DXGI_FORMAT_R32_UINT);
                assert_eq!(ibs[1].unwrap().BufferLocation, orig_ib.BufferLocation);
            } else {
                assert!(ibs.is_empty(), "a rejected preparation must not mutate the native index buffer");
            }

            // 3. Zero count or invalid format returns early without touching IB or draws.
            let draws_before = fake.draws.lock().unwrap().len();
            draw_indexed_x(list_ptr, 0x0000_0001_0000_4000, DXGI_FORMAT_R16_UINT, 0, 0);
            draw_indexed_x(list_ptr, 0x0000_0001_0000_4000, DXGI_FORMAT_R8_UINT, 3, 0);
            assert_eq!(fake.draws.lock().unwrap().len(), draws_before);
            assert!(fake.index_buffers.lock().unwrap().is_empty());

            real::forget_pso_for_test(translated_pso);
        }

        super::super::bind::forget(list_ptr);
        on_release(list_ptr as usize);
    }

    #[test]
    fn kickoff_x_render_targets_preserve_mrt_count_for_single_descriptor_range() {
        #[derive(Debug, PartialEq, Eq)]
        struct RtCall {
            n: u32,
            handles: Vec<usize>,
            rtvs_was_null: bool,
            single: i32,
            dsv: Option<usize>,
        }

        #[repr(C)]
        struct FakeRtList {
            vt: *const usize,
            calls: Mutex<Vec<RtCall>>,
        }

        unsafe extern "system" fn fake_om_set_render_targets(
            this: *mut c_void,
            n: u32,
            rtvs: *const usize,
            single: i32,
            dsv: *const usize,
        ) {
            let read_n = if rtvs.is_null() {
                0
            } else if single != 0 {
                n.min(1) as usize
            } else {
                n as usize
            };
            let handles = if read_n == 0 { Vec::new() } else { std::slice::from_raw_parts(rtvs, read_n).to_vec() };
            (*(this as *const FakeRtList)).calls.lock().unwrap().push(RtCall {
                n,
                handles,
                rtvs_was_null: rtvs.is_null(),
                single,
                dsv: dsv.as_ref().copied(),
            });
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[PC_OM_SET_RENDER_TARGETS] = fake_om_set_render_targets as usize;
        let mut ext_vt = [0usize; 2];
        ext_vt[0] = orig_vt.as_ptr() as usize;

        let fake = Box::new(FakeRtList {
            // SAFETY: `ext_vt` has 2 entries so `.add(1)` has `orig_vt` at `-1` as `real::orig_fn` expects.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            calls: Mutex::new(Vec::new()),
        });
        let list_ptr = &*fake as *const FakeRtList as *mut c_void;

        let restore = || {
            let b = LISTS.lock().unwrap().as_ref().unwrap().get(&(list_ptr as usize)).unwrap().bindings.clone();
            unsafe { restore_bindings(list_ptr, &b) };
        };

        unsafe {
            // 1. Single contiguous descriptor range for 4 MRTs + DSV: only 1 handle exists at `rtvs`,
            //    yet restore_bindings must pass `n = 4` (not 1!) to OMSetRenderTargets.
            let single_base = [0x1000usize];
            let dsv_val = 0x9000usize;
            note_render_targets(list_ptr, 4, single_base.as_ptr(), 1, &dsv_val);
            assert!(
                state_summary(list_ptr).contains("render targets Some((4, true))"),
                "state_summary should report full MRT count: {}",
                state_summary(list_ptr)
            );
            restore();

            // 2. Oversized count with single range is clamped to D3D12 max (8).
            note_render_targets(list_ptr, 12, single_base.as_ptr(), 1, std::ptr::null());
            restore();

            // 3. Separate handles (single == 0) for 3 MRTs.
            let multi = [0x2000usize, 0x2020, 0x2040];
            note_render_targets(list_ptr, 3, multi.as_ptr(), 0, std::ptr::null());
            restore();

            // 4. Depth-only (n == 0, rtvs == null, dsv != null).
            note_render_targets(list_ptr, 0, std::ptr::null(), 0, &dsv_val);
            restore();

            // 5. Null rtvs pointer with non-zero n must safely restore n == 0, null pointer.
            note_render_targets(list_ptr, 4, std::ptr::null(), 1, std::ptr::null());
            restore();
        }

        let calls = fake.calls.lock().unwrap();
        assert_eq!(
            *calls,
            vec![
                RtCall { n: 4, handles: vec![0x1000], rtvs_was_null: false, single: 1, dsv: Some(0x9000) },
                RtCall { n: 8, handles: vec![0x1000], rtvs_was_null: false, single: 1, dsv: None },
                RtCall { n: 3, handles: vec![0x2000, 0x2020, 0x2040], rtvs_was_null: false, single: 0, dsv: None },
                RtCall { n: 0, handles: vec![], rtvs_was_null: true, single: 0, dsv: Some(0x9000) },
                RtCall { n: 0, handles: vec![], rtvs_was_null: true, single: 1, dsv: None },
            ]
        );

        on_release(list_ptr as usize);
    }
}
