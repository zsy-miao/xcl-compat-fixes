//! Xbox Series shader containers: precompiled console GPU code (RDNA2), no DXIL.
//!
//! Layout reverse-engineered from a retail title's shader pack (26 503 shaders, all
//! stages, 2026-11-20). The outer DXBC container looks normal (SFI0, ISG1, OSG1, PSV0,
//! HASH, DXIL, PRIV), but the DXIL part carries no LLVM bitcode:
//!
//! ```text
//! DXIL part  DxilProgramHeader (24 B: kind/SM, size in dwords, "DXIL", version,
//!            bitcode offset, bitcode size) – the bitcode is a 4-byte "NOBC" placeholder.
//!            Between the header and the placeholder:
//! XDAT       +0 "XDAT"  +4 u32 count  +8 u32 table offset
//!            table: count × (u32 id, u32 offset, u32 size); offsets relative to the
//!            DXIL part data (= the program header)
//!              0x12  16 B hash
//!              0x0D  XBCH – one per hardware-stage variant (a VS feeding a GS has two: ES + VS)
//!              0x07  nested DxilProgramHeader whose bitcode is "NOBC"
//! XBCH       +0 "XBCH"  +4 u32 version (3)  +8 u32 compressed size  +0x10 u32 raw size
//!            +0x18 zlib stream → FSSM
//! FSSM       +0 "FSSM"  +4 u32 version (0x101)  +8 u32 size of everything after the 16-B header
//!            +0xC0 u64 code offset, +0xC8 u64 code size (offsets relative to +0x10)
//!            +0xD4 u32 flags: bit 5 = the code is for 32-lane waves (wave32)
//!            +0x100 hardware-stage descriptor u64 0x000000C0_0002_AAHH
//!                   (HH = PAL hardware stage, AA = API stage; one more every 0x88 B,
//!                   e.g. GS + its VS copy shader); unused fields are 0xDEADBEEF.
//!            The code is padded with s_code_end; metadata (user data mapping?) and the
//!            shader name follow it.
//! ```
//!
//! Exact semantics of most FSSM fields are unknown, so every step has a fallback: the
//! XDAT table is validated against both plausible bases, the zlib header is searched
//! for, and when the code pointer does not check out the FSSM is scanned for programs.

use serde::{Deserialize, Serialize};

use super::rdna2;
use super::shader::container_parts;

pub const XDAT_HASH: u32 = 0x12;
pub const XDAT_XBCH: u32 = 0x0D;
pub const XDAT_NESTED_HEADER: u32 = 0x07;

const S_ENDPGM: u32 = 0xBF81_0000;
const S_CODE_END: u32 = 0xBF9F_0000;
/// Decompressed FSSM size limit (the largest seen is well under 1 MB).
const MAX_FSSM: usize = 64 << 20;

const DXIL_KINDS: [&str; 16] = [
    "PS", "VS", "GS", "HS", "DS", "CS", "Library", "RayGen", "Intersection", "AnyHit", "ClosestHit", "Miss", "Callable",
    "MS", "AS", "Node",
];

fn rd32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o.checked_add(4)?).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
}

fn rd64(b: &[u8], o: usize) -> Option<u64> {
    b.get(o..o.checked_add(8)?).map(|x| u64::from_le_bytes(x.try_into().unwrap()))
}

/// PAL hardware stage numbering (Ls, Hs, Es, Gs, Vs, Ps, Cs); 7 only appears for mesh
/// shaders, i.e. most likely the NGG (primitive shader) path.
pub fn hw_stage_name(hw: u8) -> String {
    match hw {
        0 => "LS".into(),
        1 => "HS".into(),
        2 => "ES".into(),
        3 => "GS".into(),
        4 => "VS".into(),
        5 => "PS".into(),
        6 => "CS".into(),
        7 => "NGG".into(),
        n => format!("hw#{n}"),
    }
}

/// API stage byte of the descriptor. Only the values seen in real data are named.
pub fn api_stage_name(api: u8) -> String {
    match api {
        0 => "VS".into(),
        1 => "PS".into(),
        2 => "GS".into(),
        3 => "CS".into(),
        6 => "MS".into(),
        n => format!("api#{n}"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XdatEntry {
    pub id: u32,
    /// offset of the entry's data in the whole blob
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HwStage {
    /// offset of the descriptor inside the FSSM
    pub offset: usize,
    pub hw: u8,
    pub api: u8,
}

impl HwStage {
    pub fn label(&self) -> String {
        format!("{} (API {})", hw_stage_name(self.hw), api_stage_name(self.api))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// offset inside the FSSM
    pub offset: usize,
    pub len: usize,
}

/// One decompressed FSSM (native code + metadata for one hardware-stage variant).
#[derive(Debug, Clone, Default)]
pub struct Fssm {
    /// offset of the XBCH header in the whole blob
    pub xbch_offset: usize,
    pub xbch_version: u32,
    pub compressed: usize,
    /// decompressed FSSM (empty when decompression failed)
    pub data: Vec<u8>,
    pub version: u32,
    /// code region (offset, len) inside `data`
    pub code: Option<(usize, usize)>,
    /// `code` came from the header pointer (false: found by scanning)
    pub code_from_header: bool,
    pub stages: Vec<HwStage>,
    pub programs: Vec<Program>,
    /// constant data embedded in the code region (after a program's s_endpgm;
    /// the compiler closes such blocks with another s_endpgm)
    pub data_blocks: Vec<Program>,
    pub error: Option<String>,
}

impl Fssm {
    pub fn code_bytes(&self) -> &[u8] {
        match self.code {
            Some((o, l)) => &self.data[o..o + l],
            None => &[],
        }
    }

    /// The header says the code is for 32-lane waves: bit 5 of the flags word at +0xD4
    /// (measured on a retail pack: set for every program with 32-bit EXEC writes, clear for
    /// every one with 64-bit ones). An unused field (0xDEADBEEF) says nothing.
    pub fn wave32(&self) -> bool {
        self.version == 0x101 && rd32(&self.data, 0xD4).is_some_and(|w| w != 0xDEAD_BEEF && w & 0x20 != 0)
    }

    pub fn stage_label(&self) -> String {
        if self.stages.is_empty() {
            return "?".into();
        }
        self.stages.iter().map(|s| hw_stage_name(s.hw)).collect::<Vec<_>>().join("+")
    }
}

#[derive(Debug, Clone, Default)]
pub struct XboxShader {
    /// API stage from the DXIL program header (PS, VS, MS, …)
    pub stage: String,
    pub shader_model: (u8, u8),
    /// offset of the DXIL part data (program header) in the blob
    pub dxil_offset: usize,
    pub xdat_offset: usize,
    pub entries: Vec<XdatEntry>,
    pub hash: Option<[u8; 16]>,
    pub fssm: Vec<Fssm>,
    pub notes: Vec<String>,
}

/// (DXIL part, offset of XDAT inside it) – the cheap part of [`parse`].
fn find_xdat(blob: &[u8]) -> Option<(super::shader::Part, usize)> {
    let parts = container_parts(blob)?;
    let dxil = parts.into_iter().find(|p| p.fourcc == "DXIL")?;
    let d = &blob[dxil.offset..dxil.offset + dxil.size];
    // XDAT normally sits right after the 24-byte program header
    let x = (0..d.len().min(256).saturating_sub(3)).step_by(4).find(|&o| &d[o..o + 4] == b"XDAT")?;
    Some((dxil, x))
}

/// Is this a console shader container (DXIL part carrying XDAT instead of bitcode)?
pub fn is_xbox(blob: &[u8]) -> bool {
    find_xdat(blob).is_some()
}

/// Parse a console shader container. `None` when the blob is not one (no DXBC container,
/// no DXIL part or no XDAT inside it) – PC shaders included.
pub fn parse(blob: &[u8]) -> Option<XboxShader> {
    let (dxil, x) = find_xdat(blob)?;
    let d = &blob[dxil.offset..dxil.offset + dxil.size];
    let pv = rd32(d, 0).unwrap_or(0);
    let mut sh = XboxShader {
        stage: DXIL_KINDS.get((pv >> 16) as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("kind#{}", pv >> 16)),
        shader_model: (((pv >> 4) & 0xf) as u8, (pv & 0xf) as u8),
        dxil_offset: dxil.offset,
        xdat_offset: dxil.offset + x,
        ..Default::default()
    };
    if x != 0x18 {
        sh.notes.push(format!("XDAT at +{x:#x} in the DXIL part (expected +0x18)"));
    }
    match xdat_entries(d, x) {
        Some(e) => sh.entries = e.into_iter().map(|(id, o, s)| XdatEntry { id, offset: dxil.offset + o, size: s }).collect(),
        None => sh.notes.push("could not read the XDAT table".into()),
    }
    for e in &sh.entries {
        let data = &blob[e.offset..e.offset + e.size];
        if data.starts_with(b"XBCH") {
            if e.id != XDAT_XBCH {
                sh.notes.push(format!("XBCH in an entry with unexpected id {:#x}", e.id));
            }
            sh.fssm.push(parse_xbch(data, e.offset));
        } else if e.id == XDAT_HASH && e.size == 16 {
            sh.hash = Some(data.try_into().unwrap());
        } else if e.id == XDAT_XBCH {
            sh.notes.push(format!("entry {:#x} without an XBCH header", e.id));
        }
    }
    if sh.fssm.is_empty() {
        sh.notes.push("no FSSM (native code) in the container".into());
    }
    Some(sh)
}

/// Read the XDAT table at `x` inside the DXIL part `d`; returns (id, offset in `d`, size).
fn xdat_entries(d: &[u8], x: usize) -> Option<Vec<(u32, usize, usize)>> {
    let count = rd32(d, x + 4)? as usize;
    if count == 0 || count > 64 {
        return None;
    }
    let table_off = rd32(d, x + 8)? as usize;
    // table position: the stated offset (relative to the program header), else right after the XDAT header
    for table in [table_off, x + 12] {
        if table < x + 12 || table + count * 12 > d.len() {
            continue;
        }
        let raw: Vec<(u32, usize, usize)> = (0..count)
            .map(|i| (rd32(d, table + i * 12).unwrap(), rd32(d, table + i * 12 + 4).unwrap() as usize, rd32(d, table + i * 12 + 8).unwrap() as usize))
            .collect();
        // entry offsets: relative to the program header (seen in practice) or to XDAT
        for base in [0usize, x] {
            if raw.iter().all(|&(_, o, s)| base + o >= table + count * 12 && base + o + s <= d.len()) {
                return Some(raw.iter().map(|&(id, o, s)| (id, base + o, s)).collect());
            }
        }
    }
    None
}

fn is_zlib_header(b: &[u8]) -> bool {
    b.len() >= 2 && b[0] & 0x0f == 8 && b[0] >> 4 <= 7 && ((b[0] as u16) << 8 | b[1] as u16) % 31 == 0 && b[1] & 0x20 == 0
}

fn parse_xbch(x: &[u8], blob_offset: usize) -> Fssm {
    let mut f = Fssm { xbch_offset: blob_offset, xbch_version: rd32(x, 4).unwrap_or(0), ..Default::default() };
    let csize = rd32(x, 8).unwrap_or(0) as usize;
    let rsize = rd32(x, 0x10).unwrap_or(0) as usize;
    let start = if x.len() > 0x18 && is_zlib_header(&x[0x18..]) {
        Some(0x18)
    } else {
        (8..x.len().min(64)).find(|&o| x[o] == 0x78 && is_zlib_header(&x[o..]))
    };
    let Some(start) = start else {
        f.error = Some("XBCH: zlib stream not found".into());
        return f;
    };
    let end = if csize > 0 && start + csize <= x.len() { start + csize } else { x.len() };
    f.compressed = end - start;
    let limit = if rsize > 0 && rsize <= MAX_FSSM { rsize } else { MAX_FSSM };
    match miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&x[start..end], limit) {
        Ok(data) => {
            if rsize != 0 && data.len() != rsize {
                f.error = Some(format!("XBCH: decompressed {} B, the header says {rsize} B", data.len()));
            }
            f.data = data;
            analyze_fssm(&mut f);
        }
        Err(e) => f.error = Some(format!("XBCH: zlib error ({:?})", e.status)),
    }
    f
}

fn analyze_fssm(f: &mut Fssm) {
    let d = &f.data;
    let magic_ok = d.starts_with(b"FSSM");
    if magic_ok {
        f.version = rd32(d, 4).unwrap_or(0);
        if let (Some(off), Some(len)) = (rd64(d, 0xC0), rd64(d, 0xC8)) {
            let (off, len) = (off as usize, len as usize);
            let start = off.wrapping_add(0x10);
            if off < d.len() && len > 0 && start % 4 == 0 && start.checked_add(len).is_some_and(|e| e <= d.len()) {
                // a program ends with s_endpgm – or, an ES merged with its GS, with the
                // s_setpc_b64 into the GS code (older kits: Fortnite)
                let has_end = d[start..start + len].chunks_exact(4).any(|w| {
                    let w = u32::from_le_bytes(w.try_into().unwrap());
                    w == S_ENDPGM || is_setpc(w)
                });
                if has_end {
                    f.code = Some((start, len));
                    f.code_from_header = true;
                }
            }
        }
    } else {
        f.error.get_or_insert_with(|| "no FSSM header".into());
    }
    let code_start = f.code.map(|c| c.0).unwrap_or(d.len());
    if magic_ok {
        let mut o = 0x10;
        while o + 8 <= code_start {
            let q = rd64(d, o).unwrap();
            if q >> 16 == 0xC0_0002 && q & 0xff <= 7 {
                f.stages.push(HwStage { offset: o, hw: q as u8, api: (q >> 8) as u8 });
            }
            o += 8;
        }
    }
    if f.code.is_none() {
        f.code = scan_code_region(d);
    }
    if let Some((o, l)) = f.code {
        let code = &f.data[o..o + l];
        // Segments end at s_endpgm. One that is linked to the previous program by a
        // direct branch (either way) is part of it: the shared "all pixels discarded"
        // exit in PS, or a trampoline entered via s_setpc that jumps back.
        let mut targets: Vec<u64> = Vec::new();
        for (i, p) in split_segments(code).into_iter().enumerate() {
            if p.counted_data {
                f.data_blocks.push(Program { offset: o + p.offset, len: p.len });
                continue;
            }
            let seg = &code[p.offset..p.offset + p.len];
            let own = branch_targets(seg, p.offset as u64);
            let range = p.offset as u64..(p.offset + p.len) as u64;
            let reached = targets.iter().any(|t| range.contains(t));
            let jumps_back = f.programs.last().is_some_and(|last| {
                let r = (last.offset - o) as u64..(last.offset - o + last.len) as u64;
                own.iter().any(|t| r.contains(t))
            });
            if i > 0 && !reached && seg.len() == 4 && rd32(seg, 0) == Some(S_ENDPGM) {
                continue; // lone s_endpgm guarding the constant data block that follows
            }
            if i > 0 && !reached && !jumps_back && is_data_block(seg) {
                f.data_blocks.push(Program { offset: o + p.offset, len: p.len });
                continue;
            }
            targets.extend(own);
            match f.programs.last_mut() {
                Some(last) if reached || jumps_back => last.len = o + p.offset + p.len - last.offset,
                _ => f.programs.push(Program { offset: o + p.offset, len: p.len }),
            }
        }
    }
}

/// Static branch targets of a code segment (PCs in the same space as `pc`), far jumps included.
fn branch_targets(seg: &[u8], pc: u64) -> Vec<u64> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= seg.len() {
        match rdna2::decode(&seg[off..], pc + off as u64) {
            Ok(d) => {
                off += d.len;
                insts.push(d);
            }
            Err(_) => off += 4,
        }
    }
    rdna2::resolve_far_jumps(&mut insts);
    insts.iter().filter_map(rdna2::branch_target).collect()
}

/// u32 N, N dwords, s_endpgm – the exact layout of constant data embedded in code.
fn counted_data_block(seg: &[u8]) -> bool {
    rd32(seg, 0).is_some_and(|n| (n as usize).checked_mul(4).and_then(|x| x.checked_add(8)) == Some(seg.len()))
}

/// A segment after the first program that is constant data, not code. Retail layout:
/// u32 N, N dwords, s_endpgm. Otherwise: it starts with an undecodable dword or a
/// quarter of it does not decode (floats, small integers – VOP2 opcode 0 is unassigned).
pub fn is_data_block(seg: &[u8]) -> bool {
    if counted_data_block(seg) {
        return true;
    }
    let mut off = 0usize;
    let (mut words, mut bad) = (0usize, 0usize);
    while off + 4 <= seg.len() {
        match rdna2::decode(&seg[off..], off as u64) {
            Ok(d) => {
                words += d.len / 4;
                off += d.len;
            }
            Err(_) => {
                if off == 0 {
                    return true;
                }
                words += 1;
                bad += 1;
                off += 4;
            }
        }
    }
    bad * 4 >= words.max(1)
}

/// `s_setpc_b64 s[n:n+1]` (SOP1 op 0x20)
fn is_setpc(w: u32) -> bool {
    w & 0xFFFF_FF00 == 0xBE80_2000
}

/// Fallback when the header's code pointer does not check out: the code region ends
/// with the last s_endpgm and starts after the metadata in front of the first one
/// (walking back to 0xDEADBEEF filler or a run of zero dwords).
fn scan_code_region(d: &[u8]) -> Option<(usize, usize)> {
    let word = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let ends: Vec<usize> = (0x10..d.len().saturating_sub(3)).step_by(4).filter(|&o| word(o) == S_ENDPGM).collect();
    let first = *ends.first()?;
    let mut s = first;
    let mut zeros = 0;
    while s >= 0x14 {
        let w = word(s - 4);
        zeros = if w == 0 { zeros + 1 } else { 0 };
        if w == 0xDEAD_BEEF || zeros >= 4 {
            break;
        }
        s -= 4;
    }
    while s < first && word(s) == 0 {
        s += 4;
    }
    let mut end = first + 4;
    for &e in &ends[1..] {
        if (end..e).step_by(4).any(|o| word(o) == 0xDEAD_BEEF) {
            break;
        }
        end = e + 4;
    }
    Some((s, end - s))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub offset: usize,
    pub len: usize,
    /// constant data in the exact retail layout (u32 N, N dwords, s_endpgm)
    pub counted_data: bool,
}

/// Split a code region into segments that end with s_endpgm; s_code_end / zero padding
/// between them is skipped. Counted data blocks are recognised from raw dwords before
/// decoding (their contents may contain s_endpgm-like words). Undecodable dwords stay
/// inside their segment (they are counted, not hidden).
pub fn split_segments(code: &[u8]) -> Vec<Segment> {
    let word = |o: usize| u32::from_le_bytes(code[o..o + 4].try_into().unwrap());
    let mut out: Vec<Segment> = Vec::new();
    let mut off = 0usize;
    loop {
        while off + 4 <= code.len() && matches!(word(off), S_CODE_END | 0) {
            off += 4;
        }
        if off + 4 > code.len() {
            break;
        }
        let start = off;
        if !out.is_empty() {
            let n = word(start) as usize;
            let end = start + 4 + 4 * n;
            if n > 0 && end + 4 <= code.len() && word(end) == S_ENDPGM {
                out.push(Segment { offset: start, len: end + 4 - start, counted_data: true });
                off = end + 4;
                continue;
            }
        }
        let mut ended = false;
        while off + 4 <= code.len() {
            match rdna2::decode(&code[off..], off as u64) {
                Ok(d) => {
                    off += d.len;
                    if d.inst.flags & rdna2::F_TERMINATOR != 0 && word(off - d.len) == S_ENDPGM {
                        ended = true;
                        break;
                    }
                    // an ES that continues into its GS: the jump, then only padding
                    if is_setpc(word(off - d.len)) && (off..code.len()).step_by(4).all(|o| o + 4 > code.len() || matches!(word(o), S_CODE_END | 0)) {
                        ended = true;
                        break;
                    }
                }
                Err(_) => off += 4,
            }
        }
        out.push(Segment { offset: start, len: off.min(code.len()) - start, counted_data: false });
        if !ended {
            break;
        }
    }
    out
}

/// Instruction statistics for a set of programs.
#[derive(Debug, Default, Clone)]
pub struct InstStats {
    pub instructions: u64,
    pub undecoded: u64,
    pub by_mnemonic: std::collections::BTreeMap<String, u64>,
}

impl InstStats {
    pub fn add_code(&mut self, code: &[u8]) {
        let mut off = 0usize;
        while off + 4 <= code.len() {
            match rdna2::decode(&code[off..], off as u64) {
                Ok(d) => {
                    self.instructions += 1;
                    *self.by_mnemonic.entry(rdna2::mnemonic(&d)).or_default() += 1;
                    off += d.len;
                }
                Err(_) => {
                    self.undecoded += 1;
                    off += 4;
                }
            }
        }
    }

    pub fn add_shader(&mut self, sh: &XboxShader) {
        for f in &sh.fssm {
            for p in &f.programs {
                self.add_code(&f.data[p.offset..p.offset + p.len]);
            }
        }
    }

    pub fn merge(&mut self, other: &InstStats) {
        self.instructions += other.instructions;
        self.undecoded += other.undecoded;
        for (k, v) in &other.by_mnemonic {
            *self.by_mnemonic.entry(k.clone()).or_default() += v;
        }
    }

    /// Mnemonics sorted by count (descending).
    pub fn sorted(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self.by_mnemonic.iter().map(|(k, v)| (k.clone(), *v)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }
}

/// Disassembly of every program of a console shader (PCs relative to the code region).
pub fn listing(sh: &XboxShader) -> String {
    let mut out = String::new();
    for (i, f) in sh.fssm.iter().enumerate() {
        let Some((co, _)) = f.code else { continue };
        let mut blocks: Vec<(bool, &Program)> = f.programs.iter().map(|p| (false, p)).chain(f.data_blocks.iter().map(|p| (true, p))).collect();
        blocks.sort_by_key(|b| b.1.offset);
        for (is_data, p) in blocks {
            if is_data {
                out.push_str(&format!("; FSSM #{i}: data in the code +{:#x} (FSSM +{:#x}), {} B\n", p.offset - co, p.offset, p.len));
                for (k, w) in f.data[p.offset..p.offset + p.len].chunks_exact(4).enumerate() {
                    let w = u32::from_le_bytes(w.try_into().unwrap());
                    // show a float only where the bits look like one (|x| ≈ 1e-10 … 1e10)
                    let exp = (w >> 23) & 0xff;
                    let val = if (0x60..=0x9e).contains(&exp) { format!("{}", f32::from_bits(w)) } else { format!("{w}") };
                    out.push_str(&format!("  {:08x}: .long {w:#010x}  ; {val}\n", p.offset - co + k * 4));
                }
                continue;
            }
            out.push_str(&format!(
                "; FSSM #{i} [{}]: program +{:#x} in the code (FSSM +{:#x}), {} B\n",
                f.stage_label(),
                p.offset - co,
                p.offset,
                p.len
            ));
            let code = &f.data[p.offset..p.offset + p.len];
            for l in rdna2::disassemble(code, (p.offset - co) as u64, false) {
                let hex: Vec<String> = l.bytes.iter().map(|b| format!("{b:02x}")).collect();
                out.push_str(&format!("  {:08x}: {:<48} {}\n", l.pc, l.text, hex.join(" ")));
            }
        }
    }
    out
}

/// Non-trivial qwords of an FSSM header (skipping 0 and 0xDEADBEEF filler) – for
/// reverse engineering the metadata.
pub fn header_qwords(f: &Fssm, limit: usize) -> Vec<(usize, u64)> {
    let end = f.code.map(|c| c.0).unwrap_or(f.data.len()).min(limit);
    (0..end / 8).map(|i| (i * 8, rd64(&f.data, i * 8).unwrap())).filter(|&(_, q)| q != 0 && q != 0xDEAD_BEEF_DEAD_BEEF).collect()
}

/// Type of a root-layout entry (u32 at +0 of a 20-byte entry, see [`root_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootKind {
    Constants,
    Descriptor,
    /// one entry per descriptor range of the table
    Table,
    StaticSampler,
    /// 32-bit address of the CBV/SRV/UAV heap set on the list, which the program indexes itself
    ViewHeap,
    /// 32-bit address of the sampler heap set on the list
    SamplerHeap,
}

impl RootKind {
    fn from_raw(t: u32) -> Option<RootKind> {
        Some(match t {
            0 => RootKind::Constants,
            1 => RootKind::Descriptor,
            3 => RootKind::Table,
            4 => RootKind::StaticSampler,
            5 => RootKind::ViewHeap,
            6 => RootKind::SamplerHeap,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            RootKind::Constants => "constants",
            RootKind::Descriptor => "root descriptor",
            RootKind::Table => "table",
            RootKind::StaticSampler => "static sampler",
            RootKind::ViewHeap => "view heap",
            RootKind::SamplerHeap => "sampler heap",
        }
    }
}

/// A root parameter the program reads from its user data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootSlot {
    pub slot: u8,
    pub kind: RootKind,
    /// first user-data dword
    pub dword: u8,
    /// user SGPRs it takes (table 1, root descriptor 2, constants 1 per 32-bit value)
    pub sgprs: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    Srv,
    Sampler,
    Cbv,
    Uav,
}

/// A resource the program accesses (entry of the used-resource table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedResource {
    pub kind: ResourceKind,
    /// how the program reaches it
    pub via: RootKind,
    /// offset in the descriptor table, in dwords
    pub offset: u32,
    pub register: u32,
    /// size of its descriptor in dwords (4: buffer or sampler, 8: image)
    pub dwords: u8,
    pub dword: u8,
    pub slot: u8,
}

/// The user data a stage expects, as the console compiler recorded it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RootLayout {
    /// root parameters by slot (tables, root descriptors, constants)
    pub slots: Vec<RootSlot>,
    pub static_samplers: usize,
    /// user-data dword of the view heap's / the sampler heap's address
    pub view_heap: Option<u8>,
    pub sampler_heap: Option<u8>,
    pub used: Vec<UsedResource>,
}

const LAYOUT_ENTRY: usize = 20;
/// More entries than any stage has (the largest bindless tables have a few hundred ranges).
const MAX_LAYOUT_ENTRIES: usize = 4096;

/// SGPR of user-data dword `dword` in a stage of hardware type `hw`: dword i is s(i) in VS,
/// PS and CS, s(i + 8) in NGG (mesh) stages. None for the stages where it was not checked.
pub fn user_sgpr(hw: u8, dword: u8) -> Option<u32> {
    match hw {
        4..=6 => Some(dword as u32),
        7 => Some(dword as u32 + 8),
        _ => None,
    }
}

/// Entries of one of the stage record's tables: u64 offset (from FSSM+0x10) at `rec + at`,
/// u32 count at `rec + count_at`.
fn layout_entries<'a>(d: &'a [u8], rec: usize, at: usize, count_at: usize, what: &str) -> Result<Vec<&'a [u8]>, String> {
    let n = rd32(d, rec + count_at).ok_or_else(|| format!("{what}: count past the end of the FSSM"))? as usize;
    if n == 0 {
        return Ok(Vec::new()); // the offset is then often 0xDEADBEEF
    }
    if n > MAX_LAYOUT_ENTRIES {
        return Err(format!("{what}: {n} entries (more than {MAX_LAYOUT_ENTRIES})"));
    }
    let off = rd64(d, rec + at).ok_or_else(|| format!("{what}: offset past the end of the FSSM"))?;
    let start = usize::try_from(off).ok().and_then(|o| o.checked_add(0x10));
    match start.and_then(|s| Some((s, s.checked_add(n * LAYOUT_ENTRY)?))) {
        Some((s, e)) if e <= d.len() => Ok(d[s..e].chunks_exact(LAYOUT_ENTRY).collect()),
        _ => Err(format!("{what}: {n} entries at +{off:#x} do not fit in the FSSM ({} B)", d.len())),
    }
}

/// The root layout of `stage` in the FSSM `d`, from the stage record that starts 0x40 before
/// the stage descriptor: +0x28 / +0x30 u64 offsets (from FSSM+0x10) of the root-layout and
/// used-resource tables, +0x38 / +0x3C u32 their entry counts. Both tables have 20-byte entries:
///
/// ```text
/// root layout    u32 type | u32 offset in the table (dwords) | u32 user SGPRs | u32 base register
///                | u8 first user-data dword | u8 root slot | u16 0
///                type: 0 constants, 1 root descriptor, 3 table (one entry per range),
///                4 static sampler, 5 view-heap address, 6 sampler-heap address
/// used resource  u8 kind (0 SRV, 1 sampler, 2 CBV, 3 UAV) | u8 root type | u16 0
///                | u32 offset in the table (dwords) | u32 register | u32 ?
///                | u8 descriptor dwords | u8 first user-data dword | u8 root slot | u8 0
/// ```
///
/// Worked out from every stage of a retail title's shader pack (no entry outside
/// these types; the heap addresses always follow the last root argument, the view heap's first).
pub fn root_layout(d: &[u8], stage: &HwStage) -> Result<RootLayout, String> {
    let rec = stage.offset.checked_sub(0x40).ok_or("stage record before the start of the FSSM")?;
    let root = layout_entries(d, rec, 0x28, 0x38, "root-layout table")?;
    let used = layout_entries(d, rec, 0x30, 0x3C, "used-resource table")?;
    let mut l = RootLayout::default();
    for (i, e) in root.iter().enumerate() {
        let t = rd32(e, 0).unwrap();
        let kind = RootKind::from_raw(t).ok_or_else(|| format!("root-layout entry {i}: unknown type {t}"))?;
        let (dword, slot) = (e[16], e[17]);
        match kind {
            RootKind::StaticSampler => l.static_samplers += 1,
            RootKind::ViewHeap => {
                l.view_heap.get_or_insert(dword);
            }
            RootKind::SamplerHeap => {
                l.sampler_heap.get_or_insert(dword);
            }
            _ if l.slots.iter().any(|s| s.slot == slot) => {} // another range of a table
            _ => l.slots.push(RootSlot { slot, kind, dword, sgprs: rd32(e, 8).unwrap() }),
        }
    }
    l.slots.sort_by_key(|s| s.slot);
    for (i, e) in used.iter().enumerate() {
        let kind = match e[0] {
            0 => ResourceKind::Srv,
            1 => ResourceKind::Sampler,
            2 => ResourceKind::Cbv,
            3 => ResourceKind::Uav,
            k => return Err(format!("used-resource entry {i}: unknown kind {k}")),
        };
        let via = RootKind::from_raw(e[1] as u32).ok_or_else(|| format!("used-resource entry {i}: unknown root type {}", e[1]))?;
        l.used.push(UsedResource { kind, via, offset: rd32(e, 4).unwrap(), register: rd32(e, 8).unwrap(), dwords: e[16], dword: e[17], slot: e[18] });
    }
    Ok(l)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let header = 32 + parts.len() * 4;
        let mut v = vec![0u8; header];
        v[0..4].copy_from_slice(b"DXBC");
        v[20..24].copy_from_slice(&1u32.to_le_bytes());
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

    fn le32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    /// FSSM with the observed layout: code pointer at 0xC0, stage descriptors at 0x100 (+0x88).
    pub(crate) fn fssm(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
        let code_at = 0x308usize;
        let mut v = vec![0u8; code_at];
        v[0..4].copy_from_slice(b"FSSM");
        v[4..8].copy_from_slice(&0x101u32.to_le_bytes());
        for q in (0x18..0xC0).step_by(8) {
            v[q..q + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
        }
        v[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
        v[0xC8..0xD0].copy_from_slice(&(code.len() as u64 + 8).to_le_bytes());
        for (i, &(hw, api)) in stages.iter().enumerate() {
            let o = 0x100 + i * 0x88;
            let q = 0x0000_00C0_0002_0000u64 | (api as u64) << 8 | hw as u64;
            v[o..o + 8].copy_from_slice(&q.to_le_bytes());
        }
        v.extend_from_slice(code);
        v.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf, 0x00, 0x00, 0x9f, 0xbf]); // s_code_end padding
        v.extend_from_slice(b"\x03\0\0\0metadata\0\0\0\0");
        let size = (v.len() - 0x10) as u32;
        v[8..12].copy_from_slice(&size.to_le_bytes());
        v
    }

    fn xbch(fssm: &[u8]) -> Vec<u8> {
        let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm, 6);
        let mut v = b"XBCH".to_vec();
        le32(&mut v, 3);
        le32(&mut v, z.len() as u32);
        le32(&mut v, 0);
        le32(&mut v, fssm.len() as u32);
        le32(&mut v, 0);
        v.extend_from_slice(&z);
        v
    }

    fn program_header(kind: u32, size_dw: u32, bitcode_off: u32) -> Vec<u8> {
        let mut v = Vec::new();
        le32(&mut v, kind << 16 | 0x66);
        le32(&mut v, size_dw);
        v.extend_from_slice(b"DXIL");
        le32(&mut v, 0x106);
        le32(&mut v, bitcode_off);
        le32(&mut v, 4);
        v
    }

    /// DXIL part exactly as found in retail data: header, XDAT (hash, XBCH…, nested header), NOBC.
    pub(crate) fn dxil_part(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
        let mut blobs: Vec<(u32, Vec<u8>)> = vec![(XDAT_HASH, (0u8..16).collect())];
        for f in fssms {
            blobs.push((XDAT_XBCH, xbch(f)));
        }
        let mut nested = program_header(kind, 0x8000_0007, 0x10);
        nested.extend_from_slice(b"NOBC");
        blobs.push((XDAT_NESTED_HEADER, nested));
        let table = 0x18 + 12;
        let mut data_at = table + blobs.len() * 12;
        let mut xdat = b"XDAT".to_vec();
        le32(&mut xdat, blobs.len() as u32);
        le32(&mut xdat, table as u32);
        for (id, b) in &blobs {
            le32(&mut xdat, *id);
            le32(&mut xdat, data_at as u32);
            le32(&mut xdat, b.len() as u32);
            data_at += b.len();
        }
        for (_, b) in &blobs {
            xdat.extend_from_slice(b);
        }
        while (0x18 + xdat.len()) % 4 != 0 {
            xdat.push(0);
        }
        let nobc_at = 0x18 + xdat.len();
        let mut v = program_header(kind, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
        v.extend_from_slice(&xdat);
        v.extend_from_slice(b"NOBC");
        v
    }

    /// s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm
    pub(crate) const CODE: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];

    pub(crate) fn sample(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
        container(&[(b"SFI0", &[0; 8]), (b"HASH", &[0; 20]), (b"DXIL", &dxil_part(kind, fssms)), (b"PRIV", &[7; 16])])
    }

    #[test]
    fn parses_retail_layout() {
        let blob = sample(0, &[fssm(&CODE, &[(5, 1)])]);
        let sh = parse(&blob).expect("xbox container");
        assert_eq!(sh.stage, "PS");
        assert_eq!(sh.shader_model, (6, 6));
        assert!(sh.notes.is_empty(), "{:?}", sh.notes);
        assert_eq!(sh.entries.iter().map(|e| e.id).collect::<Vec<_>>(), vec![0x12, 0x0d, 0x07]);
        assert_eq!(sh.hash, Some(core::array::from_fn(|i| i as u8)));
        assert_eq!(sh.fssm.len(), 1);
        let f = &sh.fssm[0];
        assert_eq!(f.error, None);
        assert_eq!(f.version, 0x101);
        assert!(f.code_from_header);
        assert_eq!(f.stages, vec![HwStage { offset: 0x100, hw: 5, api: 1 }]);
        assert_eq!(f.programs, vec![Program { offset: 0x308, len: 12 }]);
        let l = listing(&sh);
        assert!(l.contains("s_mov_b32 s0, s1") && l.contains("v_add_f32") && l.contains("s_endpgm"), "{l}");
        assert!(!l.contains("s_code_end"), "padding must not be listed: {l}");
        let mut st = InstStats::default();
        st.add_shader(&sh);
        assert_eq!(st.instructions, 3);
        assert_eq!(st.undecoded, 0);
        assert_eq!(st.by_mnemonic.get("s_endpgm"), Some(&1));
    }

    #[test]
    fn gs_with_copy_shader_and_es_variant() {
        // GS: two programs in one code region, two stage descriptors
        let mut two = CODE.to_vec();
        two.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf]);
        two.extend_from_slice(&CODE);
        let gs = parse(&sample(2, &[fssm(&two, &[(3, 2), (4, 2)])])).unwrap();
        let f = &gs.fssm[0];
        assert_eq!(f.stage_label(), "GS+VS");
        assert_eq!(f.programs.len(), 2);
        assert_eq!(f.programs[1].offset - f.programs[0].offset, 16);
        // VS feeding a GS: two XBCH entries (ES + VS)
        let vs = parse(&sample(1, &[fssm(&CODE, &[(2, 0)]), fssm(&CODE, &[(4, 0)])])).unwrap();
        assert_eq!(vs.fssm.iter().map(|f| f.stage_label()).collect::<Vec<_>>(), vec!["ES", "VS"]);
    }

    #[test]
    fn branch_target_after_endpgm_belongs_to_the_program() {
        // s_cbranch_scc0 → +12 ; s_mov ; s_endpgm ; [+12] s_mov ; s_endpgm  (PS discard exit)
        let code = [
            0x02, 0x00, 0x84, 0xbf, 0x01, 0x03, 0x80, 0xbe, 0x00, 0x00, 0x81, 0xbf, 0x01, 0x03, 0x80, 0xbe, 0x00, 0x00, 0x81, 0xbf,
        ];
        let sh = parse(&sample(0, &[fssm(&code, &[(5, 1)])])).unwrap();
        assert_eq!(sh.fssm[0].programs, vec![Program { offset: 0x308, len: 20 }]);
        assert!(sh.fssm[0].data_blocks.is_empty());
        // without the branch the tail is a second program
        let mut code2 = code;
        code2[0..4].copy_from_slice(&[0x00, 0x00, 0x80, 0xbf]); // s_nop
        assert_eq!(parse(&sample(0, &[fssm(&code2, &[(5, 1)])])).unwrap().fssm[0].programs.len(), 2);
        // a branch into the middle of the next segment also links it
        let mut code3 = code;
        code3[0..4].copy_from_slice(&[0x03, 0x00, 0x82, 0xbf]); // s_branch → +16 (second s_mov... s_endpgm)
        assert_eq!(parse(&sample(0, &[fssm(&code3, &[(5, 1)])])).unwrap().fssm[0].programs, vec![Program { offset: 0x308, len: 20 }]);
    }

    #[test]
    fn counted_data_may_contain_endpgm_words() {
        let mut code = CODE.to_vec();
        for w in [0xBF81_0000u32, 3, 0x3f80_0000, 0xBF81_0000, 0x1234_5678, 0xBF81_0000] {
            code.extend_from_slice(&w.to_le_bytes());
        }
        let f = &parse(&sample(0, &[fssm(&code, &[(5, 1)])])).unwrap().fssm[0];
        assert_eq!(f.programs, vec![Program { offset: 0x308, len: 12 }]);
        assert_eq!(f.data_blocks, vec![Program { offset: 0x308 + 16, len: 20 }]);
    }

    #[test]
    fn constant_data_after_program_is_not_code() {
        // as in retail data: program, guard s_endpgm, table (u32 N, N dwords), s_endpgm
        let mut code = CODE.to_vec();
        for w in [0xBF81_0000u32, 4, 0x3e63_3333, 0x3df5_c28f, 0, 1, 0xBF81_0000] {
            code.extend_from_slice(&w.to_le_bytes());
        }
        let sh = parse(&sample(1, &[fssm(&code, &[(4, 0)])])).unwrap();
        let f = &sh.fssm[0];
        assert_eq!(f.programs, vec![Program { offset: 0x308, len: 12 }]);
        assert_eq!(f.data_blocks, vec![Program { offset: 0x308 + 16, len: 24 }]);
        let mut st = InstStats::default();
        st.add_shader(&sh);
        assert_eq!((st.instructions, st.undecoded), (3, 0));
        assert!(listing(&sh).contains(".long 0x3e633333  ; 0.22"), "{}", listing(&sh));
    }

    #[test]
    fn falls_back_when_header_pointer_is_wrong() {
        let mut f = fssm(&CODE, &[(4, 0)]);
        f[0xC0..0xC8].copy_from_slice(&0xFFFF_FFFFu64.to_le_bytes());
        let sh = parse(&sample(1, &[f])).unwrap();
        let f = &sh.fssm[0];
        assert!(!f.code_from_header);
        assert!(f.programs.iter().any(|p| p.offset == 0x308), "{:?}", f.programs);
    }

    /// A wave32 program without an EXEC write (nothing in its code tells the wave size) is
    /// wave32 by its header; a clear flag or an unused field leaves it to the code.
    #[test]
    fn wave_size_comes_from_the_header_flag() {
        let wave64 = |flags: u32| {
            let mut f = fssm(&CODE, &[(6, 5)]);
            f[0xD4..0xD8].copy_from_slice(&flags.to_le_bytes());
            let sh = parse(&sample(5, &[f])).unwrap();
            let f = &sh.fssm[0];
            let p = &f.programs[0];
            (f.wave32(), crate::gpu::recomp::fssm_wave64(f, &f.data[p.offset..p.offset + p.len]))
        };
        assert!(crate::gpu::recomp::detect_wave64(&CODE), "the code alone reads as wave64");
        assert_eq!(wave64(0x20), (true, false));
        assert_eq!(wave64(0x1F | 0x40), (false, true));
        assert_eq!(wave64(0), (false, true));
        assert_eq!(wave64(0xDEAD_BEEF), (false, true));
    }

    #[test]
    fn pc_shaders_are_not_xbox() {
        assert!(parse(&container(&[(b"DXIL", b"\x66\0\0\0\x04\0\0\0DXIL\0\0\0\0")])).is_none());
        assert!(parse(b"random").is_none());
        // truncated XDAT table must not panic
        let mut d = program_header(0, 8, 0x10);
        d.extend_from_slice(b"XDAT\x05\0\0\0\x24\0\0\0");
        let sh = parse(&container(&[(b"DXIL", &d)])).unwrap();
        assert!(sh.fssm.is_empty() && !sh.notes.is_empty());
    }

    #[test]
    fn split_skips_padding_and_keeps_unknown_words() {
        let mut code = vec![0, 0, 0, 0, 0x00, 0x00, 0x9f, 0xbf];
        code.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]); // undecodable
        code.extend_from_slice(&CODE);
        let p = split_segments(&code);
        assert_eq!(p, vec![Segment { offset: 8, len: 16, counted_data: false }]);
        let mut st = InstStats::default();
        st.add_code(&code[8..24]);
        assert_eq!((st.instructions, st.undecoded), (3, 1));
    }

    /// 20-byte root-layout entry: type, offset in the table, user SGPRs, register, dword, slot
    fn root_entry(t: u32, off: u32, sgprs: u32, reg: u32, dword: u8, slot: u8) -> Vec<u8> {
        let mut v = Vec::new();
        for x in [t, off, sgprs, reg] {
            le32(&mut v, x);
        }
        v.extend_from_slice(&[dword, slot, 0, 0]);
        v
    }

    /// 20-byte used-resource entry: kind, root type, offset, register, descriptor dwords, dword, slot
    fn used_entry(kind: u8, via: u8, off: u32, reg: u32, dwords: u8, dword: u8, slot: u8) -> Vec<u8> {
        let mut v = vec![kind, via, 0, 0];
        for x in [off, reg, 1] {
            le32(&mut v, x);
        }
        v.extend_from_slice(&[dwords, dword, slot, 0]);
        v
    }

    /// `fssm` with the two tables appended and the first stage's record (descriptor - 0x40)
    /// pointing at them; offsets and counts can be overridden for the error cases.
    fn with_layout(mut f: Vec<u8>, root: &[Vec<u8>], used: &[Vec<u8>], counts: Option<(u32, u32)>, root_at: Option<u64>) -> Vec<u8> {
        let rec = 0x100 - 0x40;
        let at = (f.len() - 0x10) as u64;
        let used_at = at + (root.len() * LAYOUT_ENTRY) as u64;
        f.extend(root.iter().chain(used).flatten());
        let (nr, nu) = counts.unwrap_or((root.len() as u32, used.len() as u32));
        f[rec + 0x28..rec + 0x30].copy_from_slice(&root_at.unwrap_or(at).to_le_bytes());
        f[rec + 0x30..rec + 0x38].copy_from_slice(&used_at.to_le_bytes());
        f[rec + 0x38..rec + 0x3C].copy_from_slice(&nr.to_le_bytes());
        f[rec + 0x3C..rec + 0x40].copy_from_slice(&nu.to_le_bytes());
        f
    }

    fn layout_of(f: Vec<u8>, kind: u32) -> Result<RootLayout, String> {
        let sh = parse(&sample(kind, &[f])).unwrap();
        let f = &sh.fssm[0];
        root_layout(&f.data, &f.stages[0])
    }

    #[test]
    fn reads_the_root_layout_of_a_compute_stage() {
        // [table (two ranges), 2 constants] + view heap, two static samplers, as a culling CS has
        let root = [
            root_entry(3, 0, 1, 0x42, 4, 0),
            root_entry(3, 8, 1, 0x02, 4, 0),
            root_entry(0, 0, 2, 0x1f4, 5, 1),
            root_entry(4, 0, 0, 0, 0, 0),
            root_entry(4, 0, 0, 1, 0, 1),
            root_entry(5, 0, 1, 0, 7, 0),
        ];
        let used = [used_entry(0, 3, 8, 0x02, 8, 4, 0), used_entry(2, 0, 0, 0x1f4, 4, 5, 1), used_entry(1, 4, 0, 1, 4, 0, 1)];
        let l = layout_of(with_layout(fssm(&CODE, &[(6, 3)]), &root, &used, None, None), 5).unwrap();
        assert_eq!(
            l.slots,
            vec![RootSlot { slot: 0, kind: RootKind::Table, dword: 4, sgprs: 1 }, RootSlot { slot: 1, kind: RootKind::Constants, dword: 5, sgprs: 2 }]
        );
        assert_eq!((l.static_samplers, l.view_heap, l.sampler_heap), (2, Some(7), None));
        assert_eq!(l.used.len(), 3);
        assert_eq!(l.used[0], UsedResource { kind: ResourceKind::Srv, via: RootKind::Table, offset: 8, register: 2, dwords: 8, dword: 4, slot: 0 });
        assert_eq!((l.used[1].kind, l.used[1].via), (ResourceKind::Cbv, RootKind::Constants));
        assert_eq!((l.used[2].kind, l.used[2].via), (ResourceKind::Sampler, RootKind::StaticSampler));
        assert_eq!(user_sgpr(6, 7), Some(7));
    }

    #[test]
    fn mesh_stage_heaps_follow_the_arguments() {
        // [t, t, t, root descriptor, 2 constants] + both heaps from dword 8: s16..s24 in NGG
        let root = [
            root_entry(3, 0, 1, 0, 8, 0),
            root_entry(3, 0, 1, 0, 9, 1),
            root_entry(3, 0, 1, 0, 10, 2),
            root_entry(1, 0, 2, 0, 11, 3),
            root_entry(0, 0, 2, 0, 13, 4),
            root_entry(5, 0, 1, 0, 15, 0),
            root_entry(6, 0, 1, 0, 16, 0),
        ];
        let l = layout_of(with_layout(fssm(&CODE, &[(7, 6)]), &root, &[], None, None), 13).unwrap();
        assert_eq!(l.slots.iter().map(|s| (s.kind, s.dword, s.sgprs)).collect::<Vec<_>>()[3], (RootKind::Descriptor, 11, 2));
        assert_eq!(l.slots.len(), 5);
        assert_eq!((l.view_heap, l.sampler_heap), (Some(15), Some(16)));
        assert_eq!((user_sgpr(7, 8), user_sgpr(7, 15), user_sgpr(7, 16)), (Some(16), Some(23), Some(24)));
        assert_eq!(user_sgpr(3, 8), None);
        // no tables at all (counts 0, offsets filler): an empty layout
        assert_eq!(layout_of(fssm(&CODE, &[(5, 1)]), 0), Ok(RootLayout::default()));
        let f = with_layout(fssm(&CODE, &[(5, 1)]), &[], &[], None, Some(0xDEAD_BEEF));
        assert_eq!(layout_of(f, 0), Ok(RootLayout::default()));
    }

    #[test]
    fn bad_root_layouts_are_errors() {
        let root = [root_entry(3, 0, 1, 0, 8, 0), root_entry(1, 0, 2, 0, 9, 1)];
        let cs = || fssm(&CODE, &[(6, 3)]);
        // a count that runs past the end of the FSSM (truncated table)
        let e = layout_of(with_layout(cs(), &root, &[], Some((3, 0)), None), 5).unwrap_err();
        assert!(e.contains("root-layout table: 3 entries"), "{e}");
        let e = layout_of(with_layout(cs(), &root, &[used_entry(0, 3, 0, 0, 4, 8, 0)], Some((2, 2)), None), 5).unwrap_err();
        assert!(e.contains("used-resource table: 2 entries"), "{e}");
        // counts that would overflow the size computation
        let e = layout_of(with_layout(cs(), &root, &[], Some((u32::MAX, 0)), None), 5).unwrap_err();
        assert!(e.contains("more than"), "{e}");
        // offsets out of bounds, including one that overflows when the 0x10 header is added
        for at in [0xDEAD_BEEF, u64::MAX - 4] {
            let e = layout_of(with_layout(cs(), &root, &[], None, Some(at)), 5).unwrap_err();
            assert!(e.contains("do not fit"), "{e}");
        }
        // types outside the known ones
        let e = layout_of(with_layout(cs(), &[root_entry(2, 0, 1, 0, 4, 0)], &[], None, None), 5).unwrap_err();
        assert_eq!(e, "root-layout entry 0: unknown type 2");
        let e = layout_of(with_layout(cs(), &root, &[used_entry(4, 3, 0, 0, 4, 8, 0)], None, None), 5).unwrap_err();
        assert_eq!(e, "used-resource entry 0: unknown kind 4");
        let e = layout_of(with_layout(cs(), &root, &[used_entry(0, 7, 0, 0, 4, 8, 0)], None, None), 5).unwrap_err();
        assert_eq!(e, "used-resource entry 0: unknown root type 7");
        // a stage descriptor too close to the start for a record
        assert!(root_layout(&cs(), &HwStage { offset: 0x38, hw: 6, api: 3 }).is_err());
    }
}

#[cfg(test)]
mod merged_es_tests {
    use super::*;

    /// An ES merged with its GS ends with the jump into the GS code, not s_endpgm: one
    /// segment up to the jump, the padding after it left out.
    #[test]
    fn es_ending_in_setpc_is_a_program() {
        let mut code: Vec<u8> = Vec::new();
        for w in [0xBF8A_0000u32 /* s_barrier */, 0xBE80_2020 /* s_setpc_b64 s[32:33] */, S_CODE_END, S_CODE_END] {
            code.extend_from_slice(&w.to_le_bytes());
        }
        let segs = split_segments(&code);
        assert_eq!(segs.len(), 1);
        assert_eq!((segs[0].offset, segs[0].len), (0, 8));
        assert!(is_setpc(0xBE80_2020) && !is_setpc(S_ENDPGM));
    }
}
