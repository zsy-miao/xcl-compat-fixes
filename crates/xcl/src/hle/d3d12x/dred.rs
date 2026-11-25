//! `XCL_DRED=1`: which translated pipeline a command list had bound where the GPU faulted.
//!
//! DRED's breadcrumbs give the operations a list got through, not which pipeline was bound, so
//! a page fault in a translated shader names no console shader. With `XCL_DRED=1` every
//! `SetPipelineState` XCL records for the title is followed by a `SetMarker`, and XCL keeps the
//! pipeline's label (its console shaders' hashes) in the list's order. The runtime kept no
//! marker text here (breadcrumb contexts came back empty: Windows 26300, NVIDIA 621.40), but it
//! records each marker as an operation, so the markers before the operation DRED stopped at
//! give the label bound there. DRED's count can lag behind the GPU (NVIDIA: a fault in the
//! second dispatch left the first one not done), so the labels bound after it are given too.
//! The title's own markers are counted, unlabelled, to keep the count in step; DRED gave the
//! list's name and not its pointer, so XCL names each list `xcl list 0x…` it marks.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{ID3D12GraphicsCommandList, D3D12_AUTO_BREADCRUMB_OP_SETMARKER};

/// translated pipeline → its label
static LABELS: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
/// PC command list → its last `RECORDINGS` recordings, oldest first: the label of each marker
/// (`None`: the title's own)
static MARKS: Mutex<Option<HashMap<usize, Vec<Vec<Option<String>>>>>> = Mutex::new(None);

/// Recordings kept per list: a list can be Reset and recorded again while an earlier
/// submission of it still runs, and that one can be the one that faults.
const RECORDINGS: usize = 4;

thread_local! {
    /// XCL's own marker is being recorded: the list's SetMarker hook must not count it again
    static OWN_MARKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_DRED").is_some())
}

/// Name a translated pipeline by its console shaders (`("ms", bytes)`, …).
pub fn label(pso: usize, shaders: &[(&str, Option<&[u8]>)]) {
    if !on() || pso == 0 {
        return;
    }
    let text = shaders.iter().filter_map(|(k, b)| b.map(|b| format!("{k} {:016x}", crate::gpu::shader::fnv64(b)))).collect::<Vec<_>>().join(" ");
    LABELS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, text);
}

fn label_of(pso: usize) -> Option<String> {
    LABELS.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// After `SetPipelineState(pso)` on `list`: a marker, and the label of what is bound now.
/// `pso` is the title's object, `bound` what reached the PC list (a stand-in while it builds).
pub unsafe fn mark_pipeline(list: *mut c_void, pso: usize, bound: usize) {
    if !on() || list.is_null() {
        return;
    }
    // nothing labelled: a stand-in still building, or a pipeline of PC shaders
    let text = label_of(bound).unwrap_or_else(|| format!("pipeline {pso:#x} (a stand-in, or not translated)"));
    let first = !MARKS.lock().unwrap().as_ref().and_then(|m| m.get(&(list as usize))).and_then(|r| r.last()).is_some_and(|v| v.iter().any(Option::is_some));
    push(list as usize, Some(text));
    if let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) {
        // DRED's node may give the list's name and not its pointer (NVIDIA, Windows 26300)
        if first {
            let _ = l.SetName(&windows::core::HSTRING::from(format!("xcl list {:#x}", list as usize)));
        }
        let w: Vec<u16> = "xcl pipeline".encode_utf16().chain([0]).collect();
        OWN_MARKER.with(|f| f.set(true));
        l.SetMarker(0, Some(w.as_ptr() as *const c_void), (w.len() * 2) as u32);
        OWN_MARKER.with(|f| f.set(false));
    }
}

/// The title's own `SetMarker` on `list`: counted, so the labels after it stay in step.
pub fn title_marker(list: *mut c_void) {
    if on() && !OWN_MARKER.with(|f| f.get()) {
        push(list as usize, None);
    }
}

fn push(list: usize, mark: Option<String>) {
    let mut g = MARKS.lock().unwrap();
    let recordings = g.get_or_insert_with(HashMap::new).entry(list).or_default();
    if recordings.is_empty() {
        recordings.push(Vec::new());
    }
    recordings.last_mut().unwrap().push(mark);
}

/// A list was Reset: a new recording, as DRED's history of it starts again too.
pub fn reset(list: *mut c_void) {
    if on() {
        let mut g = MARKS.lock().unwrap();
        let recordings = g.get_or_insert_with(HashMap::new).entry(list as usize).or_default();
        recordings.push(Vec::new());
        if recordings.len() > RECORDINGS {
            recordings.remove(0);
        }
    }
}

/// A list was released: its address may be another list's next.
pub fn forget_list(list: *mut c_void) {
    if on() {
        if let Some(m) = MARKS.lock().unwrap().as_mut() {
            m.remove(&(list as usize));
        }
    }
}

/// A pipeline was released: its address may be another pipeline's next.
pub fn forget_pso(pso: usize) {
    if on() {
        if let Some(m) = LABELS.lock().unwrap().as_mut() {
            m.remove(&pso);
        }
    }
}

/// The recording a DRED history is of: the newest one with as many markers as the history
/// (else the newest).
fn recording_of<'a>(recordings: &'a [Vec<Option<String>>], history: &[i32]) -> Option<&'a Vec<Option<String>>> {
    let total = markers_before(history, history.len());
    recordings.iter().rev().find(|r| r.len() == total).or(recordings.last())
}

/// How many markers come before operation `done` of a list's history.
fn markers_before(history: &[i32], done: usize) -> usize {
    history.iter().take(done).filter(|&&op| op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER.0).count()
}

/// The list a DRED node is about: its pointer, or the one in the name XCL gave it.
pub fn node_list(ptr: usize, name: Option<&str>) -> usize {
    if ptr != 0 {
        return ptr;
    }
    name.and_then(|n| n.strip_prefix("xcl list 0x")).and_then(|h| usize::from_str_radix(h, 16).ok()).unwrap_or(0)
}

/// The labels of the pipelines bound after operation `done` (DRED's count can lag behind the
/// GPU: the faulting operation may be one of the later ones).
pub fn bound_after(list: usize, history: &[i32], done: usize) -> Vec<String> {
    let (n, total) = (markers_before(history, done), markers_before(history, history.len()));
    let g = MARKS.lock().unwrap();
    let Some(marks) = g.as_ref().and_then(|m| m.get(&list)).and_then(|r| recording_of(r, history)) else { return Vec::new() };
    marks.iter().take(total).skip(n).flatten().cloned().collect()
}

/// The label of the pipeline `list` had bound at operation `done` of its DRED history.
pub fn bound_at(list: usize, history: &[i32], done: usize) -> Option<String> {
    let n = markers_before(history, done);
    let g = MARKS.lock().unwrap();
    let marks = recording_of(g.as_ref()?.get(&list)?, history)?;
    marks.get(..n.min(marks.len()))?.iter().rev().find_map(|m| m.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fault_in_an_earlier_recording_finds_that_recording() {
        // the list was Reset and recorded again while its first submission (2 markers) ran
        let recordings = vec![vec![Some("cs old".to_string()), Some("ms old".to_string())], vec![Some("cs new".to_string())]];
        let old_history = [46, 0, 6, 0, 6];
        assert_eq!(recording_of(&recordings, &old_history).unwrap()[1].as_deref(), Some("ms old"));
        let new_history = [46, 0, 6];
        assert_eq!(recording_of(&recordings, &new_history).unwrap()[0].as_deref(), Some("cs new"));
        // no recording with that many markers: the newest
        assert_eq!(recording_of(&recordings, &[46, 0, 0, 0, 6]).unwrap()[0].as_deref(), Some("cs new"));
    }

    #[test]
    fn node_list_takes_the_pointer_or_the_name_xcl_gave() {
        assert_eq!(node_list(0x1000, Some("xcl list 0x2000")), 0x1000);
        assert_eq!(node_list(0, Some("xcl list 0x2c234208c30")), 0x2c2_3420_8c30);
        assert_eq!(node_list(0, Some("the title's list")), 0);
        assert_eq!(node_list(0, None), 0);
    }

    #[test]
    fn markers_before_counts_only_markers_ahead_of_the_failing_op() {
        // BEGIN_COMMAND_LIST, SETPIPELINESTATE-like ops, markers, a dispatch
        let h = [46, 0, 6, 1, 0, 2, 6, 0, 6];
        assert_eq!(markers_before(&h, 0), 0);
        assert_eq!(markers_before(&h, 2), 1);
        assert_eq!(markers_before(&h, 6), 2);
        assert_eq!(markers_before(&h, 8), 3);
        assert_eq!(markers_before(&h, 100), 3);
    }

    #[test]
    fn bound_at_skips_the_titles_markers() {
        let list = 0x1234_5670usize;
        {
            let mut g = MARKS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            m.insert(list, vec![vec![Some("cs a".into()), None, Some("ms b ps c".into())]]);
        }
        // ops: begin, our marker (cs a), dispatch, the title's marker, dispatch, our marker (ms b), dispatch
        let h = [46, 0, 6, 0, 6, 0, 6];
        assert_eq!(bound_at(list, &h, 2).as_deref(), Some("cs a"));
        // after the title's marker the last label is still the compute one
        assert_eq!(bound_at(list, &h, 4).as_deref(), Some("cs a"));
        assert_eq!(bound_at(list, &h, 6).as_deref(), Some("ms b ps c"));
        assert_eq!(bound_at(list, &h, 1), None);
        assert_eq!(bound_at(0xdead, &h, 6), None);
        assert_eq!(bound_after(list, &h, 2), vec!["ms b ps c".to_string()]);
        assert_eq!(bound_after(list, &h, 6), Vec::<String>::new());
        MARKS.lock().unwrap().as_mut().unwrap().remove(&list);
    }
}
