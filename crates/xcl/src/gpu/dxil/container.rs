//! The DXBC container around a DXIL program: feature flags (`SFI0`),
//! signatures (`ISG1`/`OSG1`), pipeline state validation info (`PSV0`), the
//! program (`DXIL`) and the container digest D3D12 checks.
//!
//! The digest is DXBC's retail hash: MD5 over everything after the digest
//! field, with a non-standard final block (bit count in the first and
//! `(bits >> 2) | 1` in the last dword). Computing it here means shipped
//! builds need neither `dxcompiler.dll` nor `dxil.dll`; the validator is a
//! test oracle only.

/// DXIL shader kinds (`DXIL::ShaderKind`, also PSV's stage numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Pixel = 0,
    Vertex = 1,
    Geometry = 2,
    Hull = 3,
    Domain = 4,
    Compute = 5,
    Mesh = 13,
    Amplification = 14,
}

impl Stage {
    pub fn profile_prefix(self) -> &'static str {
        match self {
            Stage::Pixel => "ps",
            Stage::Vertex => "vs",
            Stage::Geometry => "gs",
            Stage::Hull => "hs",
            Stage::Domain => "ds",
            Stage::Compute => "cs",
            Stage::Mesh => "ms",
            Stage::Amplification => "as",
        }
    }
}

/// PSV resource types (`PSVResourceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsvResType {
    Sampler = 1,
    Cbv = 2,
    SrvTyped = 3,
    SrvRaw = 4,
    SrvStructured = 5,
    UavTyped = 6,
    UavRaw = 7,
    UavStructured = 8,
}

/// One bound resource as PSV0 lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsvResource {
    pub res_type: PsvResType,
    pub space: u32,
    pub lower: u32,
    pub upper: u32,
    /// `DXIL::ResourceKind` (11 raw buffer, 12 structured buffer, 2 texture 2D…).
    pub kind: u32,
    pub flags: u32,
}

/// Pipeline state validation info.
#[derive(Debug, Clone)]
pub struct Psv {
    pub stage: Stage,
    /// Wave lane count range (0 / `u32::MAX` when unconstrained).
    pub min_wave: u32,
    pub max_wave: u32,
    pub numthreads: [u32; 3],
    pub resources: Vec<PsvResource>,
    pub entry: String,
}

/// Retail DXBC container digest of `data` (the bytes after the digest field).
pub fn hash(data: &[u8]) -> [u8; 16] {
    let mut st = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let bits = (data.len() as u32).wrapping_mul(8);
    let left = data.len() % 64;
    let full = data.len() - left;
    for blk in data[..full].chunks_exact(64) {
        md5_block(&mut st, blk);
    }
    let rest = &data[full..];
    let mut blk = [0u8; 64];
    if left >= 56 {
        blk[..left].copy_from_slice(rest);
        blk[left] = 0x80;
        md5_block(&mut st, &blk);
        let mut last = [0u8; 64];
        last[..4].copy_from_slice(&bits.to_le_bytes());
        last[60..].copy_from_slice(&((bits >> 2) | 1).to_le_bytes());
        md5_block(&mut st, &last);
    } else {
        blk[..4].copy_from_slice(&bits.to_le_bytes());
        blk[4..4 + left].copy_from_slice(rest);
        blk[4 + left] = 0x80;
        blk[60..].copy_from_slice(&((bits >> 2) | 1).to_le_bytes());
        md5_block(&mut st, &blk);
    }
    let mut out = [0u8; 16];
    for (i, w) in st.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn md5_block(st: &mut [u32; 4], blk: &[u8]) {
    const S: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    let m: Vec<u32> = blk.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let [mut a, mut b, mut c, mut d] = *st;
    for i in 0..64 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let k = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
        let f = f.wrapping_add(a).wrapping_add(k).wrapping_add(m[g]);
        a = d;
        d = c;
        c = b;
        b = b.wrapping_add(f.rotate_left(S[(i / 16) * 4 + i % 4]));
    }
    st[0] = st[0].wrapping_add(a);
    st[1] = st[1].wrapping_add(b);
    st[2] = st[2].wrapping_add(c);
    st[3] = st[3].wrapping_add(d);
}

fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*s)
}

/// The `DXIL` part: program header, bitcode header, bitcode.
fn program_part(stage: Stage, sm: (u32, u32), dxil_version: (u32, u32), bitcode: &[u8]) -> Vec<u8> {
    assert!(bitcode.len() % 4 == 0);
    let mut p = Vec::with_capacity(24 + bitcode.len());
    let total = 24 + bitcode.len();
    p.extend_from_slice(&(((stage as u32) << 16) | (sm.0 << 4) | sm.1).to_le_bytes());
    p.extend_from_slice(&((total / 4) as u32).to_le_bytes());
    p.extend_from_slice(&fourcc(b"DXIL").to_le_bytes());
    p.extend_from_slice(&((dxil_version.0 << 8) | dxil_version.1).to_le_bytes());
    p.extend_from_slice(&16u32.to_le_bytes());
    p.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
    p.extend_from_slice(bitcode);
    p
}

/// PSV0 with `PSVRuntimeInfo3` (validator 1.8+).
fn psv_part(psv: &Psv) -> Vec<u8> {
    let mut p = Vec::new();
    let w = |p: &mut Vec<u8>, v: u32| p.extend_from_slice(&v.to_le_bytes());
    w(&mut p, 52);
    p.extend_from_slice(&[0; 16]); // stage-specific union (none for compute)
    w(&mut p, psv.min_wave);
    w(&mut p, psv.max_wave);
    p.extend_from_slice(&[psv.stage as u8, 0, 0, 0]); // stage, uses view id, union
    p.extend_from_slice(&[0; 4]); // signature element counts / input vectors
    p.extend_from_slice(&[0; 4]); // output vectors per stream
    for n in psv.numthreads {
        w(&mut p, n);
    }
    w(&mut p, 1); // entry function name: offset 1 in the string table
    w(&mut p, psv.resources.len() as u32);
    if !psv.resources.is_empty() {
        w(&mut p, 24);
        for r in &psv.resources {
            for v in [r.res_type as u32, r.space, r.lower, r.upper, r.kind, r.flags] {
                w(&mut p, v);
            }
        }
    }
    // String table: "" then the entry name, padded to 4 bytes.
    let mut strings = vec![0u8];
    strings.extend_from_slice(psv.entry.as_bytes());
    strings.push(0);
    while strings.len() % 4 != 0 {
        strings.push(0);
    }
    w(&mut p, strings.len() as u32);
    p.extend_from_slice(&strings);
    w(&mut p, 0); // semantic index table
    p
}

/// An empty input / output signature.
fn empty_signature() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&0u32.to_le_bytes());
    p.extend_from_slice(&8u32.to_le_bytes());
    p
}

/// Builds the signed container for a DXIL program.
pub fn container(stage: Stage, sm: (u32, u32), bitcode: &[u8], features: u64, psv: &Psv) -> Vec<u8> {
    let dxil_version = (1, sm.1.min(8));
    let parts: Vec<([u8; 4], Vec<u8>)> = vec![
        (*b"SFI0", features.to_le_bytes().to_vec()),
        (*b"ISG1", empty_signature()),
        (*b"OSG1", empty_signature()),
        (*b"PSV0", psv_part(psv)),
        (*b"DXIL", program_part(stage, sm, dxil_version, bitcode)),
    ];
    let header = 32 + 4 * parts.len();
    let total: usize = header + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut off = header;
    for (_, d) in &parts {
        out.extend_from_slice(&(off as u32).to_le_bytes());
        off += 8 + d.len();
    }
    for (cc, d) in &parts {
        out.extend_from_slice(cc);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    let digest = hash(&out[20..]);
    out[4..20].copy_from_slice(&digest);
    out
}

// ------------------------------------------------- graphics signatures

/// One row of an `ISG1`/`OSG1`/`PSG1` signature (`DxilProgramSignatureElement`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigPartElement {
    pub stream: u32,
    pub name: String,
    pub index: u32,
    /// `D3D_NAME` system value.
    pub system_value: u32,
    /// `D3D_REGISTER_COMPONENT_TYPE` (1 uint, 2 sint, 3 float).
    pub comp_type: u32,
    /// Register (row), `u32::MAX` for unallocated system values (depth, coverage…).
    pub register: u32,
    pub mask: u8,
    /// Inputs: components read; outputs: components never written.
    pub rw_mask: u8,
    pub min_precision: u32,
}

/// A signature part. Elements are written in the given order; each name is
/// stored once, in first-use order of `name_order` and then of the elements
/// (DXC writes names in declaration order while sorting elements by register).
pub fn signature_part(elems: &[SigPartElement], name_order: &[&str]) -> Vec<u8> {
    let mut strings: Vec<u8> = Vec::new();
    let mut offs: Vec<(String, u32)> = Vec::new();
    let base = 8 + 32 * elems.len() as u32;
    for n in name_order.iter().copied().chain(elems.iter().map(|e| e.name.as_str())) {
        if !offs.iter().any(|(s, _)| s == n) {
            offs.push((n.to_string(), base + strings.len() as u32));
            strings.extend_from_slice(n.as_bytes());
            strings.push(0);
        }
    }
    while strings.len() % 4 != 0 {
        strings.push(0);
    }
    let mut p = Vec::new();
    let w = |p: &mut Vec<u8>, v: u32| p.extend_from_slice(&v.to_le_bytes());
    w(&mut p, elems.len() as u32);
    w(&mut p, 8);
    for e in elems {
        let off = offs.iter().find(|(s, _)| *s == e.name).unwrap().1;
        for v in [e.stream, off, e.index, e.system_value, e.comp_type, e.register] {
            w(&mut p, v);
        }
        p.extend_from_slice(&[e.mask, e.rw_mask, 0, 0]);
        w(&mut p, e.min_precision);
    }
    p.extend_from_slice(&strings);
    p
}

/// One signature element as PSV0 lists it (`PSVSignatureElement0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsvSigElement {
    /// Empty for system values (PSV stores no name for them).
    pub name: String,
    pub indices: Vec<u32>,
    pub rows: u8,
    pub start_row: u8,
    pub cols: u8,
    pub start_col: u8,
    pub allocated: bool,
    /// `DXIL::SemanticKind`.
    pub kind: u8,
    /// `PSVComponentType` (1 uint32, 2 sint32, 3 float32, 4 uint16, 5 sint16, 6 float16).
    pub comp_type: u8,
    /// `DXIL::InterpolationMode`.
    pub interp: u8,
    pub dynamic_mask: u8,
    pub stream: u8,
}

/// PSV0 for any stage: `PSVRuntimeInfo3` plus signature elements and the
/// input → output dependency tables.
#[derive(Debug, Clone)]
pub struct PsvFull {
    pub stage: Stage,
    /// The 16-byte stage union of `PSVRuntimeInfo0` (VS/PS/MS/... info).
    pub stage_info: [u8; 16],
    pub min_wave: u32,
    pub max_wave: u32,
    pub uses_view_id: bool,
    /// The 2-byte union of `PSVRuntimeInfo1` (GS max vertices / MS prim vectors + topology).
    pub info1: [u8; 2],
    pub numthreads: [u32; 3],
    pub resources: Vec<PsvResource>,
    pub entry: String,
    pub inputs: Vec<PsvSigElement>,
    pub outputs: Vec<PsvSigElement>,
    pub prims: Vec<PsvSigElement>,
    pub input_vectors: u8,
    pub output_vectors: [u8; 4],
    /// Tables after the signature elements, in order: view-ID output masks
    /// (if `uses_view_id`), input → output dependency tables, and the
    /// patch-constant / primitive tables (dwords).
    pub tables: Vec<Vec<u32>>,
}

/// Writes PSV0 for [`PsvFull`].
pub fn psv_part_full(psv: &PsvFull) -> Vec<u8> {
    let mut p = Vec::new();
    let w = |p: &mut Vec<u8>, v: u32| p.extend_from_slice(&v.to_le_bytes());
    w(&mut p, 52);
    p.extend_from_slice(&psv.stage_info);
    w(&mut p, psv.min_wave);
    w(&mut p, psv.max_wave);
    p.extend_from_slice(&[psv.stage as u8, psv.uses_view_id as u8, psv.info1[0], psv.info1[1]]);
    p.extend_from_slice(&[psv.inputs.len() as u8, psv.outputs.len() as u8, psv.prims.len() as u8, psv.input_vectors]);
    p.extend_from_slice(&psv.output_vectors);
    for n in psv.numthreads {
        w(&mut p, n);
    }
    let all: Vec<&PsvSigElement> = psv.inputs.iter().chain(&psv.outputs).chain(&psv.prims).collect();
    // String table: "" then non-system names in element order (not
    // deduplicated, as DXC writes it), then the entry name.
    let mut strings = vec![0u8];
    let mut name_off = Vec::new();
    for e in &all {
        if e.name.is_empty() {
            name_off.push(0);
        } else {
            name_off.push(strings.len() as u32);
            strings.extend_from_slice(e.name.as_bytes());
            strings.push(0);
        }
    }
    let entry_off = strings.len() as u32;
    strings.extend_from_slice(psv.entry.as_bytes());
    strings.push(0);
    while strings.len() % 4 != 0 {
        strings.push(0);
    }
    w(&mut p, entry_off);
    w(&mut p, psv.resources.len() as u32);
    if !psv.resources.is_empty() {
        w(&mut p, 24);
        for r in &psv.resources {
            for v in [r.res_type as u32, r.space, r.lower, r.upper, r.kind, r.flags] {
                w(&mut p, v);
            }
        }
    }
    w(&mut p, strings.len() as u32);
    p.extend_from_slice(&strings);
    // Semantic index table: each element's index list, reusing an existing run.
    let mut table: Vec<u32> = Vec::new();
    let mut idx_off = Vec::new();
    for e in &all {
        let n = e.indices.len();
        let found = if n == 0 || table.len() < n { None } else { (0..=table.len() - n).find(|&i| table[i..i + n] == e.indices[..]) };
        idx_off.push(match found {
            Some(i) => i as u32,
            None => {
                table.extend_from_slice(&e.indices);
                (table.len() - n) as u32
            }
        });
    }
    w(&mut p, table.len() as u32);
    for v in &table {
        w(&mut p, *v);
    }
    if !all.is_empty() {
        w(&mut p, 16);
        for (i, e) in all.iter().enumerate() {
            w(&mut p, name_off[i]);
            w(&mut p, idx_off[i]);
            let cs = (e.cols & 0xF) | ((e.start_col & 3) << 4) | ((e.allocated as u8) << 6);
            p.extend_from_slice(&[e.rows, e.start_row, cs, e.kind, e.comp_type, e.interp, (e.dynamic_mask & 0xF) | ((e.stream & 3) << 4), 0]);
        }
    }
    for t in &psv.tables {
        for v in t {
            w(&mut p, *v);
        }
    }
    p
}

/// Builds the signed container with explicit signature parts (`psg` for
/// mesh primitive outputs) and a prebuilt PSV0.
#[allow(clippy::too_many_arguments)]
pub fn container_with_signatures(stage: Stage, sm: (u32, u32), bitcode: &[u8], features: u64, isg: Vec<u8>, osg: Vec<u8>, psg: Option<Vec<u8>>, psv: Vec<u8>) -> Vec<u8> {
    let dxil_version = (1, sm.1.min(8));
    let mut parts: Vec<([u8; 4], Vec<u8>)> = vec![(*b"SFI0", features.to_le_bytes().to_vec()), (*b"ISG1", isg), (*b"OSG1", osg)];
    if let Some(p) = psg {
        parts.push((*b"PSG1", p));
    }
    parts.push((*b"PSV0", psv));
    parts.push((*b"DXIL", program_part(stage, sm, dxil_version, bitcode)));
    let header = 32 + 4 * parts.len();
    let total: usize = header + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut off = header;
    for (_, d) in &parts {
        out.extend_from_slice(&(off as u32).to_le_bytes());
        off += 8 + d.len();
    }
    for (cc, d) in &parts {
        out.extend_from_slice(cc);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    let digest = hash(&out[20..]);
    out[4..20].copy_from_slice(&digest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DXC 1.9 output for a 64-thread compute shader storing to two root UAVs
    /// (see `tests/data/dxil/*.hlsl`).
    const DXC_CS: &[u8] = include_bytes!("../../../../../tests/data/dxil/dxc_cs_raw_uav.dxil");

    #[test]
    fn digest_matches_dxc_signed_container() {
        assert_eq!(hash(&DXC_CS[20..]), DXC_CS[4..20]);
    }

    #[test]
    fn digest_covers_both_tail_layouts() {
        // Lengths with < 56 and >= 56 trailing bytes take different final blocks;
        // both must differ from each other and be stable.
        let a = hash(&[7u8; 64 + 10]);
        let b = hash(&[7u8; 64 + 60]);
        assert_ne!(a, b);
        assert_eq!(a, hash(&[7u8; 64 + 10]));
    }

    #[test]
    fn psv_matches_dxc_layout_for_compute_with_two_raw_uavs() {
        let psv = Psv {
            stage: Stage::Compute,
            min_wave: 64,
            max_wave: 64,
            numthreads: [64, 1, 1],
            resources: (0..2)
                .map(|i| PsvResource { res_type: PsvResType::UavRaw, space: 0, lower: i, upper: i, kind: 11, flags: 0 })
                .collect(),
            entry: "main".into(),
        };
        let ours = psv_part(&psv);
        let theirs = part(DXC_CS, b"PSV0");
        assert_eq!(ours, theirs);
    }

    pub(crate) fn part<'a>(c: &'a [u8], cc: &[u8; 4]) -> &'a [u8] {
        let n = u32::from_le_bytes(c[28..32].try_into().unwrap()) as usize;
        for i in 0..n {
            let off = u32::from_le_bytes(c[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
            if &c[off..off + 4] == cc {
                let len = u32::from_le_bytes(c[off + 4..off + 8].try_into().unwrap()) as usize;
                return &c[off + 8..off + 8 + len];
            }
        }
        panic!("no {cc:?} part")
    }
}
