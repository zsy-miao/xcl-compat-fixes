//! Shader capture and classification.
//!
//! Every shader blob a title hands to D3D12.X (pipeline state creation, root
//! signatures) is classified and – once per unique hash – written to
//! `<data>\shaders\`. Unknown formats are scanned for RDNA2 programs and a
//! disassembly is written next to the blob. This is how we learn, per title, whether
//! shaders still carry DXIL/DXBC or only console GPU microcode.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::{rdna2, xbox};

/// DXBC container part FourCCs that are well understood on PC.
const KNOWN_PARTS: &[&str] = &[
    "DXIL", "SHEX", "SHDR", "ISGN", "OSGN", "ISG1", "OSG1", "PSG1", "PCSG", "PSV0", "RTS0", "STAT", "ILDB", "ILDN",
    "HASH", "SFI0", "RDEF", "SPDB", "PRIV", "VERS", "SRCI", "PDBI", "SPDB", "OSG5", "IFCE", "RDAT", "MESH",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Part {
    pub fourcc: String,
    pub offset: usize,
    pub size: usize,
}

/// Parse a DXBC-style container ("DXBC", hash, version, size, part table).
pub fn container_parts(blob: &[u8]) -> Option<Vec<Part>> {
    if blob.len() < 32 || &blob[0..4] != b"DXBC" {
        return None;
    }
    let rd = |o: usize| blob.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let count = rd(28)?;
    if count > 64 {
        return None;
    }
    let mut parts = Vec::with_capacity(count);
    for i in 0..count {
        let off = rd(32 + i * 4)?;
        let fourcc = blob.get(off..off + 4)?;
        let size = rd(off + 4)?;
        if off + 8 + size > blob.len() {
            return None;
        }
        parts.push(Part {
            fourcc: String::from_utf8_lossy(fourcc).chars().map(|c| if c.is_ascii_graphic() { c } else { '?' }).collect(),
            offset: off + 8,
            size,
        });
    }
    Some(parts)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// DXBC container with a DXIL part – runnable on PC D3D12 (after re-signing if needed)
    Dxil,
    /// DXBC container with SM5 byte code – runnable on PC D3D12
    Dxbc,
    /// console container: DXIL part with XDAT → XBCH → FSSM (native RDNA2 code), no bitcode
    XboxNative,
    /// DXBC container with only parts we do not know (console microcode?)
    ContainerUnknown,
    /// ELF object (AMD code objects are ELF)
    Elf,
    /// no container, but RDNA2 programs were found
    Rdna2Raw,
    Unknown,
}

impl Kind {
    pub fn label(&self) -> &'static str {
        match self {
            Kind::Dxil => "dxil",
            Kind::Dxbc => "dxbc",
            Kind::XboxNative => "xbox-native",
            Kind::ContainerUnknown => "container-unknown",
            Kind::Elf => "elf",
            Kind::Rdna2Raw => "rdna2-raw",
            Kind::Unknown => "unknown",
        }
    }
}

pub fn classify(blob: &[u8]) -> (Kind, Vec<Part>) {
    if let Some(parts) = container_parts(blob) {
        let has = |n: &str| parts.iter().any(|p| p.fourcc == n);
        let kind = if has("DXIL") && xbox::is_xbox(blob) {
            Kind::XboxNative
        } else if has("DXIL") {
            Kind::Dxil
        } else if has("SHEX") || has("SHDR") {
            Kind::Dxbc
        } else if has("RTS0") && parts.iter().all(|p| KNOWN_PARTS.contains(&p.fourcc.as_str())) {
            Kind::Dxbc // root signature container
        } else {
            Kind::ContainerUnknown
        };
        return (kind, parts);
    }
    if blob.len() >= 4 && &blob[0..4] == b"\x7fELF" {
        return (Kind::Elf, Vec::new());
    }
    if !rdna2::find_programs(blob).is_empty() {
        return (Kind::Rdna2Raw, Vec::new());
    }
    (Kind::Unknown, Vec::new())
}

/// Disassembly of every plausible RDNA2 program in the blob's unknown regions.
pub fn rdna2_listing(blob: &[u8]) -> String {
    if let Some(sh) = xbox::parse(blob) {
        return xbox::listing(&sh);
    }
    let mut regions: Vec<(String, usize, usize)> = Vec::new();
    match container_parts(blob) {
        Some(parts) => {
            for p in parts.iter().filter(|p| !KNOWN_PARTS.contains(&p.fourcc.as_str())) {
                regions.push((format!("part {}", p.fourcc), p.offset, p.size));
            }
        }
        None => regions.push(("blob".into(), 0, blob.len())),
    }
    let mut out = String::new();
    for (label, off, size) in regions {
        let data = &blob[off..off + size];
        for (start, len) in rdna2::find_programs(data) {
            out.push_str(&format!("; {label}: program at +{:#x} ({} bytes)\n", off + start, len));
            for l in rdna2::disassemble(&data[start..start + len], (off + start) as u64, false) {
                let hex: Vec<String> = l.bytes.iter().map(|b| format!("{b:02x}")).collect();
                out.push_str(&format!("  {:08x}: {:<48} {}\n", l.pc, l.text, hex.join(" ")));
            }
        }
    }
    out
}

pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ShaderStats {
    pub unique: usize,
    pub submitted: usize,
    pub by_kind: BTreeMap<String, usize>,
    pub unknown_parts: BTreeMap<String, usize>,
    pub by_stage: BTreeMap<String, usize>,
    pub rejected_pointers: usize,
    pub dump_dir: String,
    /// console containers: hardware stages of their FSSM variants (VS, PS, NGG, ES, GS+VS, CS …)
    #[serde(default)]
    pub xbox_hw_stages: BTreeMap<String, usize>,
    /// console containers whose FSSM could not be fully parsed
    #[serde(default)]
    pub xbox_errors: usize,
    /// RDNA2 instructions decoded from unique console shaders (recompiler priorities)
    #[serde(default)]
    pub rdna2_instructions: u64,
    #[serde(default)]
    pub rdna2_undecoded_dwords: u64,
    /// (mnemonic, count), most frequent first
    #[serde(default)]
    pub rdna2_mnemonics: Vec<(String, u64)>,
    /// recompiler at capture: programs translated (HLSL next to the dumps), complete ones,
    /// by mode, mnemonics still missing
    #[serde(default)]
    pub recomp_programs: usize,
    #[serde(default)]
    pub recomp_complete: usize,
    #[serde(default)]
    pub recomp_modes: BTreeMap<String, usize>,
    #[serde(default)]
    pub recomp_unsupported: BTreeMap<String, usize>,
    /// DXC on the translations (`XCL_RECOMP_DXC=1`; cached in `recomp::dxc::cache_dir()`)
    #[serde(default)]
    pub recomp_dxc_ok: usize,
    #[serde(default)]
    pub recomp_dxc_cached: usize,
    #[serde(default)]
    pub recomp_dxc_failed: usize,
}

struct State {
    seen: HashSet<u64>,
    stats: ShaderStats,
    insts: xbox::InstStats,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn dump_dir() -> PathBuf {
    crate::state::data_dir().join("shaders")
}

/// Is `[p, p+len)` committed, readable memory? (Game-provided pointers may be garbage
/// when our guess of a console struct layout is wrong.)
pub fn readable(p: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS};
    if p == 0 || len == 0 || len > 256 << 20 {
        return false;
    }
    let end = match p.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(p, len)) {
        return true;
    }
    if resident_and_readable(p, end) {
        return true;
    }
    let mut cur = p;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(cur as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0 || mbi.State != MEM_COMMIT || mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || mbi.Protect == 0 {
            return false;
        }
        cur = mbi.BaseAddress as usize + mbi.RegionSize;
    }
    true
}

/// Is the whole range currently committed and writable? Instruction emulation must
/// validate every region before writing a batch, including guard and read-only pages.
/// Query the OS afresh so decommit/free/protection changes cannot leave a stale proof.
pub(crate) fn writable(p: usize, len: usize) -> bool {
    if p == 0 || len == 0 || len > 256 << 20 {
        return false;
    }
    writable_range(p, len)
}

/// DirectStorage accepts an empty destination and has no shader-sized range cap.
/// A resident page's current protection proves access without VirtualQuery's region walk.
/// Never infer the protection of a nonresident page, even in XCL's coherent window.
pub(crate) fn writable_range(p: usize, len: usize) -> bool {
    let Some(end) = p.checked_add(len) else { return false };
    if len == 0 { return true; }
    if p == 0 { return false; }
    match resident_access(p, end, true) {
        Some(writable) => writable,
        None => writable_os(p, end),
    }
}

fn writable_os(p: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE,
        PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, PAGE_WRITECOPY,
    };
    let mut cur = p;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(cur as *const _, &mut mbi, std::mem::size_of_val(&mbi)) };
        if n == 0 || mbi.State != MEM_COMMIT || mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0
            || !matches!(mbi.Protect & 0xff, PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)
        {
            return false;
        }
        let Some(next) = (mbi.BaseAddress as usize).checked_add(mbi.RegionSize) else { return false };
        if next <= cur { return false; }
        cur = next;
    }
    true
}

/// Preserve the readable check's existing small-query policy.
fn resident_and_readable(p: usize, end: usize) -> bool {
    let pages = ((end - 1) >> 12) - (p >> 12) + 1;
    pages <= 16 && resident_access(p, end, false) == Some(true)
}

/// Fresh working-set proof, adapted from mrp1no's PR #501 (c73b0c8) and the existing
/// readable fast path. Unlike #501, EVERY page must be resident: an invalid entry has
/// no usable Win32Protection. It is unknown, not writable, and falls back to VirtualQuery.
/// No proof is cached across calls; protect/decommit/recommit cannot retain a stale yes.
fn resident_access(p: usize, end: usize, write: bool) -> Option<bool> {
    use windows_sys::Win32::System::Memory::{
        PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS,
        PAGE_READWRITE, PAGE_WRITECOPY,
    };
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    const BATCH: usize = 64; // an unaligned 64 KiB destination occupies 17 pages
    let (mut page, last) = (p >> 12, (end - 1) >> 12);
    let mut ws: [PSAPI_WORKING_SET_EX_INFORMATION; BATCH] = unsafe { std::mem::zeroed() };
    while page <= last {
        let count = (last - page + 1).min(BATCH);
        for (i, e) in ws[..count].iter_mut().enumerate() {
            *e = unsafe { std::mem::zeroed() };
            e.VirtualAddress = ((page + i) << 12) as *mut std::ffi::c_void;
        }
        let size = (count * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe { K32QueryWorkingSetEx(windows_sys::Win32::System::Threading::GetCurrentProcess(), ws.as_mut_ptr() as *mut std::ffi::c_void, size) } == 0 {
            return None;
        }
        for e in &ws[..count] {
            let flags = unsafe { e.VirtualAttributes.Flags };
            if flags & 1 == 0 { return None; }
            let protect = ((flags >> 4) & 0x7FF) as u32;
            let allowed = protect != 0 && protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
                && (!write || matches!(protect & 0xff, PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY));
            if !allowed { return Some(false); }
        }
        page += count;
    }
    Some(true)
}

/// Record a shader blob submitted by the title. Safe on garbage pointers.
pub fn capture(stage: &str, ptr: usize, len: usize) {
    if ptr == 0 && len == 0 {
        return; // unused stage
    }
    let mut g = STATE.lock().unwrap();
    let st = g.get_or_insert_with(|| State { seen: HashSet::new(), stats: ShaderStats::default(), insts: xbox::InstStats::default() });
    st.stats.submitted += 1;
    if !readable(ptr, len) {
        st.stats.rejected_pointers += 1;
        if st.stats.rejected_pointers <= 5 {
            crate::xlog_warn!("[SHADER] {stage}: invalid pointer/size ({ptr:#x}, {len}) – different struct layout on the console?");
        }
        return;
    }
    let blob = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec();
    let h = fnv64(&blob);
    *st.stats.by_stage.entry(stage.to_string()).or_default() += 1;
    if !st.seen.insert(h) {
        return;
    }
    st.stats.unique += 1;
    let (kind, parts) = classify(&blob);
    *st.stats.by_kind.entry(kind.label().to_string()).or_default() += 1;
    for p in parts.iter().filter(|p| !KNOWN_PARTS.contains(&p.fourcc.as_str())) {
        *st.stats.unknown_parts.entry(p.fourcc.clone()).or_default() += 1;
    }
    drop(g);
    if !dumps_enabled() {
        let names: Vec<&str> = parts.iter().map(|p| p.fourcc.as_str()).collect();
        crate::xlog_info!("[SHADER] {stage} {h:016x}: {} B, format {}, parts [{}]", blob.len(), kind.label(), names.join(","));
        if kind == Kind::XboxNative {
            account_xbox(&blob, h);
        }
        return;
    }
    let dir = dump_dir();
    let _ = std::fs::create_dir_all(&dir);
    let base = dir.join(format!("{h:016x}_{stage}"));
    let _ = std::fs::write(base.with_extension("bin"), &blob);
    let names: Vec<&str> = parts.iter().map(|p| p.fourcc.as_str()).collect();
    crate::xlog_info!("[SHADER] {stage} {h:016x}: {} B, format {}, parts [{}]", blob.len(), kind.label(), names.join(","));
    if kind == Kind::XboxNative {
        capture_xbox(&blob, &base, h);
    } else if matches!(kind, Kind::ContainerUnknown | Kind::Rdna2Raw | Kind::Unknown) {
        let listing = rdna2_listing(&blob);
        if !listing.is_empty() {
            let _ = std::fs::write(base.with_extension("rdna2.txt"), listing);
            crate::xlog_info!("[SHADER] {h:016x}: RDNA2 code found – disassembly in {}", base.with_extension("rdna2.txt").display());
        }
    }
}

/// `XCL_DUMP_SHADERS=1`: every shader the title submits is written to the title's `shaders`
/// folder – the blob, a console program's FSSMs, its disassembly and its HLSL (translated once
/// more for that: the report's `recomp_*` counts come from it). Off by default: a retail game's
/// ~17 000 dumps took 4.8 GB, and translating each again took its loading threads ~0.1 s a
/// program at every start.
fn dumps_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DUMP_SHADERS").is_ok_and(|v| v == "1"))
}

/// Console container without dumps: its FSSM variants and instructions accounted.
fn account_xbox(blob: &[u8], h: u64) {
    let Some(sh) = xbox::parse(blob) else { return };
    let mut insts = xbox::InstStats::default();
    insts.add_shader(&sh);
    let failed = !sh.notes.is_empty() || sh.fssm.iter().any(|f| f.error.is_some() || f.programs.is_empty());
    if let Some(st) = STATE.lock().unwrap().as_mut() {
        for f in &sh.fssm {
            *st.stats.xbox_hw_stages.entry(f.stage_label()).or_default() += 1;
        }
        st.stats.xbox_errors += failed as usize;
        st.insts.merge(&insts);
    }
    if failed {
        crate::xlog_info!("[SHADER] {h:016x}: console container {} – WARNING: incomplete parse (xcl disasm)", sh.stage);
    }
}

/// Console container: account its FSSM variants and instructions, write the
/// decompressed FSSMs and the disassembly next to the blob.
fn capture_xbox(blob: &[u8], base: &std::path::Path, h: u64) {
    let Some(sh) = xbox::parse(blob) else { return };
    let mut insts = xbox::InstStats::default();
    insts.add_shader(&sh);
    let failed = !sh.notes.is_empty() || sh.fssm.iter().any(|f| f.error.is_some() || f.programs.is_empty());
    if let Some(st) = STATE.lock().unwrap().as_mut() {
        for f in &sh.fssm {
            *st.stats.xbox_hw_stages.entry(f.stage_label()).or_default() += 1;
        }
        st.stats.xbox_errors += failed as usize;
        st.insts.merge(&insts);
    }
    for (i, f) in sh.fssm.iter().enumerate() {
        let _ = std::fs::write(base.with_extension(format!("fssm{i}.bin")), &f.data);
    }
    let listing = xbox::listing(&sh);
    let _ = std::fs::write(base.with_extension("rdna2.txt"), &listing);
    recompile(blob, &sh, base);
    let stages: Vec<String> = sh.fssm.iter().map(|f| f.stage_label()).collect();
    crate::xlog_info!(
        "[SHADER] {h:016x}: console container {} SM {}.{}, FSSM [{}], {} RDNA2 instructions{}",
        sh.stage,
        sh.shader_model.0,
        sh.shader_model.1,
        stages.join(", "),
        insts.instructions,
        if failed { " – WARNING: incomplete parse (xcl disasm)" } else { "" }
    );
}

/// Translate every program of a captured console shader (mode from its interface) and
/// write the HLSL next to the dumps; with `XCL_RECOMP_DXC=1` also compile it (cached).
fn recompile(blob: &[u8], sh: &xbox::XboxShader, base: &std::path::Path) {
    use crate::gpu::recomp;
    let dxc = std::env::var("XCL_RECOMP_DXC").is_ok_and(|v| v == "1");
    for (i, f) in sh.fssm.iter().enumerate() {
        let mode = recomp::interface(blob, sh, f).unwrap_or(recomp::Mode::Kernel);
        let (label, profile) = match &mode {
            recomp::Mode::Vertex(_) => ("VS", "vs_6_6"),
            recomp::Mode::Pixel(_) => ("PS", "ps_6_6"),
            recomp::Mode::Mesh(_) => ("MS", "cs_6_6"),
            recomp::Mode::Geometry(_) => ("GS", "cs_6_6"),
            recomp::Mode::Compute(_) => ("CS", "cs_6_6"),
            recomp::Mode::Kernel => ("kernel", "cs_6_6"),
        };
        for (j, p) in f.programs.iter().enumerate() {
            let code = &f.data[p.offset..p.offset + p.len];
            let opt = recomp::Options { mode: mode.clone(), wave64: recomp::detect_wave64(code), ..Default::default() };
            let Ok(t) = recomp::translate(code, &opt) else { continue };
            let _ = std::fs::write(base.with_extension(format!("fssm{i}.p{j}.hlsl")), &t.hlsl);
            let built = if dxc && t.complete() { Some(recomp::dxc::compile_cached(&t.hlsl, "main", profile)) } else { None };
            if let Some(Err(e)) = &built {
                crate::xlog_warn!("[SHADER] DXC rejected the translation {}: {}", base.display(), e.lines().next().unwrap_or(""));
            }
            if let Some(st) = STATE.lock().unwrap().as_mut() {
                let s = &mut st.stats;
                s.recomp_programs += 1;
                s.recomp_complete += t.complete() as usize;
                *s.recomp_modes.entry(label.into()).or_default() += 1;
                for (m, n) in &t.unsupported {
                    *s.recomp_unsupported.entry(m.clone()).or_default() += n;
                }
                match &built {
                    Some(Ok((_, true))) => s.recomp_dxc_cached += 1,
                    Some(Ok(_)) => s.recomp_dxc_ok += 1,
                    Some(Err(_)) => s.recomp_dxc_failed += 1,
                    None => {}
                }
            }
        }
    }
}

pub fn stats() -> ShaderStats {
    let g = STATE.lock().ok();
    let mut s = g.as_ref().and_then(|g| g.as_ref().map(|st| st.stats.clone())).unwrap_or_default();
    if let Some(st) = g.as_ref().and_then(|g| g.as_ref()) {
        s.rdna2_instructions = st.insts.instructions;
        s.rdna2_undecoded_dwords = st.insts.undecoded;
        s.rdna2_mnemonics = st.insts.sorted();
    }
    s.dump_dir = dump_dir().to_string_lossy().into_owned();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let header = 32 + parts.len() * 4;
        let mut v = vec![0u8; header];
        v[0..4].copy_from_slice(b"DXBC");
        v[24..28].copy_from_slice(&1u32.to_le_bytes());
        v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
        for (i, (cc, data)) in parts.iter().enumerate() {
            let off = v.len() as u32;
            v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
            v.extend_from_slice(*cc);
            v.extend_from_slice(&(data.len() as u32).to_le_bytes());
            v.extend_from_slice(data);
        }
        let total = v.len() as u32;
        v[24..28].copy_from_slice(&total.to_le_bytes());
        v
    }

    #[test]
    fn classifies_containers() {
        assert_eq!(classify(&container(&[(b"ISG1", b"x"), (b"DXIL", b"bitcode")])).0, Kind::Dxil);
        assert_eq!(classify(&container(&[(b"SHEX", b"code")])).0, Kind::Dxbc);
        // a made-up console part with RDNA2 code inside: s_mov_b32 s0, s1 ; s_endpgm
        let code = [0x01u8, 0x03, 0x80, 0xbe, 0x00, 0x00, 0x81, 0xbf];
        let blob = container(&[(b"XBOX", &code)]);
        let (k, parts) = classify(&blob);
        assert_eq!(k, Kind::ContainerUnknown);
        assert_eq!(parts[0].fourcc, "XBOX");
        let listing = rdna2_listing(&blob);
        assert!(listing.contains("s_mov_b32 s0, s1"), "{listing}");
        assert!(listing.contains("s_endpgm"));
        assert_eq!(classify(&code).0, Kind::Rdna2Raw);
        assert_eq!(classify(b"\x7fELF....").0, Kind::Elf);
        assert_eq!(classify(b"random junk!").0, Kind::Unknown);
    }

    #[test]
    fn classifies_and_lists_console_containers() {
        use crate::gpu::xbox::tests::{fssm, sample, CODE};
        let blob = sample(0, &[fssm(&CODE, &[(5, 1)])]);
        let (k, parts) = classify(&blob);
        assert_eq!(k, Kind::XboxNative);
        assert!(parts.iter().any(|p| p.fourcc == "DXIL"));
        let l = rdna2_listing(&blob);
        assert!(l.contains("FSSM #0 [PS]") && l.contains("v_add_f32"), "{l}");
    }

    #[test]
    fn readable_rejects_garbage() {
        assert!(!readable(0, 10));
        assert!(!readable(0x10, 10));
        let v = vec![1u8; 100];
        assert!(readable(v.as_ptr() as usize, 100));
    }

    #[test]
    fn writable_resident_proof_covers_unaligned_destinations_and_batch_edges() {
        use windows_sys::Win32::System::Memory::*;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 80 * 4096, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            for i in 0..80 { std::ptr::write_volatile((p + i * 4096) as *mut u8, 1); }
            assert_eq!(resident_access(p + 17, p + 17 + 65536, true), Some(true), "unaligned64KiB spans17pages");
            assert_eq!(resident_access(p + 17, p + 70 * 4096 + 17, true), Some(true), "cross64-page query boundary");
            assert!(writable_range(p + 17, 65536));
            let mut old = 0;
            assert_ne!(VirtualProtect((p + 64 * 4096) as *const _, 4096, PAGE_READONLY, &mut old), 0);
            assert!(!writable_range(p + 17, 70 * 4096));
            assert_ne!(VirtualFree(p as *mut _, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn writable_proof_is_fresh_and_does_not_touch_cold_or_guard_pages() {
        use windows_sys::Win32::System::Memory::*;
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 3 * 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            let cold = p + 4096;
            assert_eq!(resident_access(cold, cold + 4096, true), None);
            assert!(writable_range(cold, 4096), "nonresident RW uses the OS fallback");
            assert_eq!(resident_access(cold, cold + 4096, true), None, "asking must not fault the page in");
            for _ in 0..32 {
                let mut old = 0;
                std::ptr::write_volatile(p as *mut u8, 1);
                assert!(writable_range(p, 4096));
                for protect in [PAGE_READONLY, PAGE_NOACCESS, PAGE_READWRITE | PAGE_GUARD] {
                    assert_ne!(VirtualProtect(p as *const _, 4096, protect, &mut old), 0);
                    assert_ne!(resident_access(p, p + 4096, true), Some(true));
                    assert!(!writable_range(p, 4096));
                    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    assert_ne!(VirtualQuery(p as *const _, &mut m, std::mem::size_of_val(&m)), 0);
                    assert_eq!(m.Protect, protect, "a guard must not be consumed");
                    assert_ne!(VirtualProtect(p as *const _, 4096, PAGE_READWRITE, &mut old), 0);
                }
                assert_ne!(VirtualFree(p as *mut _, 4096, MEM_DECOMMIT), 0);
                assert!(!writable_range(p, 4096), "a prior yes cannot survive decommit");
                assert!(!VirtualAlloc(p as *const _, 4096, MEM_COMMIT, PAGE_READONLY).is_null());
                assert_eq!(resident_access(p, p + 4096, true), None);
                assert!(!writable_range(p, 4096), "a cold read-only page is not assumed RW");
                assert_ne!(VirtualProtect(p as *const _, 4096, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
                assert!(!writable_range(p, 4096));
                assert_ne!(VirtualProtect(p as *const _, 4096, PAGE_READWRITE, &mut old), 0);
                assert!(writable_range(p, 4096));
            }
            assert_ne!(VirtualFree(p as *mut _, 0, MEM_RELEASE), 0);
        }
        assert!(writable_range(0, 0), "retain DirectStorage's empty-range contract");
        assert!(!writable_range(0, 1));
        assert!(!writable_range(usize::MAX, 2));
    }

    #[test]
    #[ignore = "explicit CPU benchmark: commits2GiB coherent+2GiB sparse, noGPU"]
    fn benchmark_writable_runtime_windows() {
        use std::hint::black_box;
        use std::time::Instant;
        use windows_sys::Win32::System::Memory::*;
        const GIB: usize = 1 << 30;
        let u = crate::gpu::uma::Uma::for_runtime_with(2 * GIB, 2 * GIB).expect("2GiB coherent runtime window");
        let coherent = u.window(3).0;
        let sparse = u.window(0).0;
        unsafe { assert!(!VirtualAlloc(sparse as *const _, 2 * GIB, MEM_COMMIT, PAGE_READWRITE).is_null()); }
        let measure = |name: &str, p: usize, len: usize, expected: bool| {
            assert_eq!(writable_os(p, p + len), expected);
            assert_eq!(writable_range(p, len), expected);
            let mut old = Vec::new();
            let mut fast = Vec::new();
            for sample in 0..7 {
                for which in if sample % 2 == 0 { [false, true] } else { [true, false] } {
                    let t = Instant::now();
                    for _ in 0..8 {
                        let answer = if which { writable_range(black_box(p), black_box(len)) } else { writable_os(black_box(p), black_box(p + len)) };
                        assert_eq!(black_box(answer), expected);
                    }
                    let ns = t.elapsed().as_nanos() / 8;
                    if which { fast.push(ns); } else { old.push(ns); }
                }
            }
            old.sort_unstable(); fast.sort_unstable();
            println!("writable-benchmark {}", serde_json::json!({"case":name,"address":format!("{p:#x}"),"bytes":len,"pages":((p+len-1)>>12)-(p>>12)+1,"expected":expected,"old_median_ns":old[3],"fast_median_ns":fast[3],"old_samples_ns":old,"fast_samples_ns":fast}));
        };
        for (name, base) in [("coherent", coherent), ("sparse", sparse)] {
            let p = base + 65536 + 17;
            unsafe { for page in (p & !4095..p + 65536).step_by(4096) { std::ptr::write_volatile(page as *mut u8, 1); } }
            assert_eq!(resident_access(p, p + 65536, true), Some(true));
            measure(&format!("{name}-resident-unaligned64KiB"), p, 65536, true);
            let cold = base + GIB + 17;
            assert_eq!(resident_access(cold, cold + 65536, true), None);
            measure(&format!("{name}-cold64KiB"), cold, 65536, true);
            unsafe {
                let mut old = 0;
                assert_ne!(VirtualProtect((p & !4095) as *const _, 4096, PAGE_READONLY, &mut old), 0);
                measure(&format!("{name}-readonly-hole"), p, 65536, false);
                assert_ne!(VirtualProtect((p & !4095) as *const _, 4096, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
                measure(&format!("{name}-guard-hole"), p, 65536, false);
                let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(p as *const _, &mut m, std::mem::size_of_val(&m)), 0);
                assert_ne!(m.Protect & PAGE_GUARD, 0);
                assert_ne!(VirtualProtect((p & !4095) as *const _, 4096, PAGE_READWRITE, &mut old), 0);
                assert_ne!(VirtualFree((p & !4095) as *mut _, 4096, MEM_DECOMMIT), 0);
                measure(&format!("{name}-decommitted-hole"), p, 65536, false);
            }
        }
    }

    mod clzero_batch_tests {
        use super::*;
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE,
            MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        };

        const PAGE: usize = 0x1000;

        struct Reservation(usize);
        impl Drop for Reservation {
            fn drop(&mut self) { unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE); } }
        }

        #[test]
        fn writable_checks_every_native_region_and_current_protection() {
            unsafe {
                let r = Reservation(VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize);
                assert_ne!(r.0, 0);
                assert!(!writable(r.0, PAGE), "reserved pages are not writable");
                assert!(!VirtualAlloc(r.0 as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                assert!(writable(r.0 + 17, 2 * PAGE - 17));
                assert!(!writable(r.0, 2 * PAGE + 1), "reject a reserved tail");
                let mut old = 0;
                assert_ne!(VirtualProtect((r.0 + PAGE) as *const _, PAGE, PAGE_READONLY, &mut old), 0);
                assert!(readable(r.0 + PAGE, PAGE));
                assert!(!writable(r.0, 2 * PAGE), "reject a read-only tail");
                assert!(writable(r.0, PAGE));
                for protect in [PAGE_NOACCESS, PAGE_READWRITE | PAGE_GUARD] {
                    assert_ne!(VirtualProtect(r.0 as *const _, PAGE, protect, &mut old), 0);
                    assert!(!writable(r.0, 64), "reject protection {protect:#x}");
                }
                assert_ne!(VirtualProtect(r.0 as *const _, PAGE, PAGE_READWRITE, &mut old), 0);
                assert!(writable(r.0, 64), "a new query observes restored protection");
                assert_ne!(VirtualFree(r.0 as *mut _, PAGE, MEM_DECOMMIT), 0);
                assert!(!writable(r.0, 64), "a new query observes decommit");
            }
        }

        #[test]
        fn writable_rejects_invalid_ranges_and_released_pages() {
            assert!(!writable(0, 64));
            assert!(!writable(1, 0));
            assert!(!writable(usize::MAX - 31, 64));
            unsafe {
                let r = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
                assert_ne!(r, 0);
                assert!(writable(r, 64));
                assert_ne!(VirtualFree(r as *mut _, 0, MEM_RELEASE), 0);
                assert!(!writable(r, 64), "a new query observes release");
            }
        }
    }

    #[test]
    fn readable_answers_from_the_working_set_without_touching_pages() {
        use windows_sys::Win32::System::Memory::*;
        unsafe {
            // 0, 1 written; 2 no access; 3 guard; 4 reserved; 5, 6 committed and never touched
            let base = VirtualAlloc(std::ptr::null(), 7 << 12, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert_ne!(base, 0);
            let page = |k: usize| base + (k << 12);
            for (k, protect) in [(0, PAGE_READWRITE), (1, PAGE_READWRITE), (2, PAGE_NOACCESS), (3, PAGE_READWRITE | PAGE_GUARD), (5, PAGE_READWRITE), (6, PAGE_READONLY)] {
                assert!(!VirtualAlloc(page(k) as *const _, 1 << 12, MEM_COMMIT, protect).is_null());
            }
            std::ptr::write_volatile(page(0) as *mut u8, 1);
            std::ptr::write_volatile(page(1) as *mut u8, 1);
            // written pages: answered from the working set
            assert!(resident_and_readable(page(0), page(2)));
            assert!(readable(page(0), 2 << 12));
            assert!(readable(page(0) + 100, 8));
            // untouched pages: the region walk answers, and they stay untouched
            assert!(!resident_and_readable(page(5), page(5) + 8));
            assert!(readable(page(5), 8));
            assert!(readable(page(6), 1 << 12));
            assert!(!resident_and_readable(page(5), page(7)));
            assert!(!readable(page(2), 8));
            assert!(!readable(page(3), 8));
            assert!(!readable(page(4), 8));
            // asking about a guard page leaves it a guard page
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(page(3) as *const _, &mut mbi, std::mem::size_of_val(&mbi)), 0);
            assert_ne!(mbi.Protect & PAGE_GUARD, 0);
            // a range is readable only when each of its pages is
            assert!(!readable(page(1) + 0xFF0, 0x20));
            assert!(!readable(page(4) + 0xFF0, 0x20));
            assert!(!readable(page(0), 20 << 12));
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
        }
        // more pages than one working-set query takes: the region walk
        let v = vec![1u8; 32 << 12];
        assert!(!resident_and_readable(v.as_ptr() as usize, v.as_ptr() as usize + v.len()));
        assert!(readable(v.as_ptr() as usize, v.len()));
    }
}
