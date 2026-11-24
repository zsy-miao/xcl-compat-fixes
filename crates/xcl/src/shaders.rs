//! `xcl shaders <file|folder> [--top N] [--json] [--recomp] [--compile N]` – statistics over
//! a shader corpus: formats, stages, FSSM variants and the RDNA2 instruction histogram that
//! sets the recompiler's priorities. Inputs: a single blob, a folder (e.g. `<data>\shaders\`)
//! or an RPF9 shader pack (records found by scanning, see `gpu::pack`). `--recomp`
//! translates every program to HLSL (kernel mode) and reports what is still missing;
//! `--compile N` also compiles N of the complete translations with DXC.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::gpu::{dxbc, pack, rdna2, recomp, shader, xbox};

/// `--iface`: how console programs connect to their DXBC signatures (hypothesis checks for
/// the VS/PS interface of the recompiler).
#[derive(Default)]
struct IfaceAgg {
    ps: usize,
    ps_in_regs: usize,
    ps_eq_regs: usize,
    ps_examples: Vec<String>,
    vs: usize,
    vs_param_in_regs: usize,
    vs_param_eq: usize,
    vs_param_examples: Vec<String>,
    vs_inputs: usize,
    vs_in_vec4: usize,
    vs_in_packed: usize,
    vs_input_examples: Vec<String>,
    /// attr / param index = rank among the non-system elements (register order)
    ps_rank: usize,
    vs_rank: usize,
    /// (hw stage, smallest base B with inputs ⊆ {B + 4·register + c}) → programs
    vs_base: BTreeMap<String, usize>,
    /// PS: SPI_PS_INPUT_ENA at FSSM +0x248 explains the I/J VGPRs of v_interp / the initial VGPRs read
    ps_ena_ij: usize,
    ps_ena_reads: usize,
    ps_ena_examples: Vec<String>,
    /// hardware stage → (programs, VGPR read before written → programs, SGPR → programs, export target → programs)
    inputs: BTreeMap<String, StageInputs>,
}

#[derive(Default)]
struct StageInputs {
    programs: usize,
    vgpr: BTreeMap<u64, usize>,
    sgpr: BTreeMap<u64, usize>,
    exports: BTreeMap<u64, usize>,
    sendmsg: BTreeMap<u64, usize>,
    /// PSV0: "kind / max verts / max prims / topology / numthreads / groupshared" → programs
    psv: BTreeMap<String, usize>,
    /// first instruction reading an input register (v0–v7, s0–s3) → programs
    first_use: BTreeMap<String, usize>,
}

impl IfaceAgg {
    fn merge(&mut self, o: IfaceAgg) {
        self.ps += o.ps;
        self.ps_in_regs += o.ps_in_regs;
        self.ps_eq_regs += o.ps_eq_regs;
        self.vs += o.vs;
        self.vs_param_in_regs += o.vs_param_in_regs;
        self.vs_param_eq += o.vs_param_eq;
        self.vs_inputs += o.vs_inputs;
        self.vs_in_vec4 += o.vs_in_vec4;
        self.vs_in_packed += o.vs_in_packed;
        self.ps_rank += o.ps_rank;
        self.ps_ena_ij += o.ps_ena_ij;
        self.ps_ena_reads += o.ps_ena_reads;
        self.ps_ena_examples.extend(o.ps_ena_examples);
        self.ps_ena_examples.truncate(8);
        self.vs_rank += o.vs_rank;
        for (k, v) in o.vs_base {
            *self.vs_base.entry(k).or_default() += v;
        }
        for (a, b) in [(&mut self.ps_examples, o.ps_examples), (&mut self.vs_param_examples, o.vs_param_examples), (&mut self.vs_input_examples, o.vs_input_examples)] {
            a.extend(b);
            a.truncate(8);
        }
        for (k, x) in o.inputs {
            let e = self.inputs.entry(k).or_default();
            e.programs += x.programs;
            for (a, b) in [(&mut e.vgpr, x.vgpr), (&mut e.sgpr, x.sgpr), (&mut e.exports, x.exports), (&mut e.sendmsg, x.sendmsg)] {
                for (r, n) in b {
                    *a.entry(r).or_default() += n;
                }
            }
            for (r, n) in x.psv {
                *e.psv.entry(r).or_default() += n;
            }
            for (r, n) in x.first_use {
                *e.first_use.entry(r).or_default() += n;
            }
        }
    }

    fn add(&mut self, name: &str, blob: &[u8], sh: &xbox::XboxShader) {
        let (isg, osg) = dxbc::signatures(blob);
        let user = |s: &[dxbc::SigElement]| -> Vec<(u32, u8, String)> {
            s.iter().filter(|e| e.system_value == 0 && !e.name.starts_with("SV_")).map(|e| (e.register, e.mask, e.semantic())).collect()
        };
        let sig_txt = |s: &[dxbc::SigElement]| s.iter().map(|e| format!("{}@r{}:{:x}", e.semantic(), e.register, e.mask)).collect::<Vec<_>>().join(" ");
        for f in &sh.fssm {
            let Some(p) = f.programs.first() else { continue };
            let code = &f.data[p.offset..p.offset + p.len];
            let mut attrs = std::collections::BTreeSet::new();
            let mut ij = std::collections::BTreeSet::new();
            let mut params = std::collections::BTreeSet::new();
            let (mut read, mut written) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let (mut sread, mut swritten) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let (mut exps, mut msgs) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let mut first: Vec<String> = Vec::new();
            let mut off = 0;
            while off + 4 <= code.len() {
                let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break };
                off += d.len;
                let n = d.inst.name;
                if n.starts_with("V_INTERP") {
                    let a = rdna2::field_value(d.enc, &d.words, "ATTR")
                        .or_else(|| rdna2::field_value(d.enc, &d.words, "SRC0").map(|v| v & 0x3f));
                    if let Some(a) = a {
                        attrs.insert(a as u32);
                    }
                    if n == "V_INTERP_P1_F32" || n == "V_INTERP_P2_F32" {
                        if let Some(v) = rdna2::field_value(d.enc, &d.words, "VSRC") {
                            ij.insert(v);
                        }
                    }
                }
                if n == "EXP" {
                    let t = rdna2::field_value(d.enc, &d.words, "TGT").unwrap_or(0);
                    if (32..64).contains(&t) {
                        params.insert((t - 32) as u32);
                    }
                }
                let (r, w) = rdna2::vgpr_uses(&d);
                for v in r {
                    if !written.contains(&v) {
                        if v < 8 && !read.contains(&v) {
                            first.push(format!("v{v}: {}", rdna2::format(&d)));
                        }
                        read.insert(v);
                    }
                }
                written.extend(w);
                let (r, w) = rdna2::sgpr_uses(&d);
                for v in r {
                    if !swritten.contains(&v) {
                        if v < 4 && !sread.contains(&v) {
                            first.push(format!("s{v}: {}", rdna2::format(&d)));
                        }
                        sread.insert(v);
                    }
                }
                swritten.extend(w);
                if n == "EXP" {
                    exps.insert(rdna2::field_value(d.enc, &d.words, "TGT").unwrap_or(0));
                }
                if n == "S_SENDMSG" || n == "S_SENDMSGHALT" {
                    msgs.insert(rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) & 0xF);
                }
            }
            {
                let hw0 = f.stages.first().map(|s| s.hw).unwrap_or(99);
                let e = self.inputs.entry(format!("{} {}", sh.stage, xbox::hw_stage_name(hw0))).or_default();
                e.programs += 1;
                for &v in &read {
                    *e.vgpr.entry(v).or_default() += 1;
                }
                for &v in &sread {
                    *e.sgpr.entry(v).or_default() += 1;
                }
                for &t in &exps {
                    *e.exports.entry(t).or_default() += 1;
                }
                for &m in &msgs {
                    *e.sendmsg.entry(m).or_default() += 1;
                }
                let key = match dxbc::psv(blob) {
                    Some(p) if p.kind == Some(13) => format!(
                        "MS v{} p{} topo{} threads{:?} lds{} wave{}-{}",
                        p.ms_max_vertices, p.ms_max_primitives, p.ms_topology, p.numthreads.unwrap_or_default(), p.ms_groupshared, p.min_wave, p.max_wave
                    ),
                    Some(p) => format!("kind {:?} threads{:?} wave{}-{}", p.kind, p.numthreads, p.min_wave, p.max_wave),
                    None => "no PSV0".into(),
                };
                *e.psv.entry(key).or_default() += 1;
                for u in first {
                    *e.first_use.entry(u).or_default() += 1;
                }
            }
            let hw = f.stages.first().map(|s| s.hw).unwrap_or(99);
            if sh.stage == "PS" && hw == 5 && !attrs.is_empty() {
                self.ps += 1;
                let regs: std::collections::BTreeSet<u32> = user(&isg).iter().map(|x| x.0).collect();
                let inside = attrs.is_subset(&regs);
                self.ps_in_regs += inside as usize;
                self.ps_eq_regs += (attrs == regs) as usize;
                let n = user(&isg).len() as u32;
                self.ps_rank += attrs.iter().all(|&a| a < n) as usize;
                // VGPR layout of the enabled PS inputs (hardware order, sizes 2/2/2/3/2/2/2/1…)
                let ena = f.data.get(0x248..0x24C).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
                let sizes = [2u64, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
                let (mut at, mut pairs) = (0u64, std::collections::BTreeSet::new());
                for (bit, sz) in sizes.iter().enumerate() {
                    if ena >> bit & 1 == 1 {
                        if bit <= 6 && bit != 3 {
                            pairs.insert(at);
                            pairs.insert(at + 1);
                        }
                        at += sz;
                    }
                }
                let ij_ok = ij.is_subset(&pairs);
                let rd_ok = read.iter().all(|&v| v < at);
                self.ps_ena_ij += ij_ok as usize;
                self.ps_ena_reads += rd_ok as usize;
                if (!ij_ok || !rd_ok) && self.ps_ena_examples.len() < 8 {
                    self.ps_ena_examples.push(format!("{name}: ena {ena:#x} ({at} VGPR) I/J {ij:?} read before write {:?}", read.iter().take(12).collect::<Vec<_>>()));
                }
                if !inside && self.ps_examples.len() < 8 {
                    self.ps_examples.push(format!("{name}: attr {attrs:?} vs ISG1 {}", sig_txt(&isg)));
                }
            }
            if sh.stage == "VS" && (hw == 4 || hw == 2) {
                self.vs += 1;
                let regs: std::collections::BTreeSet<u32> = user(&osg).iter().map(|x| x.0).collect();
                let inside = params.is_subset(&regs);
                self.vs_param_in_regs += inside as usize;
                self.vs_param_eq += (params == regs) as usize;
                let n = user(&osg).len() as u32;
                let rank_ok = params.iter().all(|&a| a < n);
                self.vs_rank += rank_ok as usize;
                if !rank_ok && self.vs_param_examples.len() < 8 {
                    self.vs_param_examples.push(format!("RANK {name} [{}]: param {params:?} vs OSG1 {}", xbox::hw_stage_name(hw), sig_txt(&osg)));
                }

                // attribute VGPRs the fetch shader fills: read before written, above the 4 system VGPRs
                let ins: std::collections::BTreeSet<u64> = read.iter().copied().filter(|&v| v >= 4).collect();
                let u = user(&isg);
                if !ins.is_empty() && !u.is_empty() {
                    self.vs_inputs += 1;
                    let vec4: std::collections::BTreeSet<u64> =
                        u.iter().flat_map(|(r, m, _)| (0..4).filter(move |c| m >> c & 1 == 1).map(move |c| 4 + 4 * *r as u64 + c)).collect();
                    let mut packed = std::collections::BTreeSet::new();
                    let mut sorted = u.clone();
                    sorted.sort();
                    let mut next = 4u64;
                    for (_, m, _) in &sorted {
                        let n = 8 - m.leading_zeros() as u64; // highest component + 1
                        packed.extend(next..next + n);
                        next += n;
                    }
                    // system values with a register (SV_VertexID / SV_InstanceID) arrive like attributes
                    let sv_regs: Vec<u32> = isg.iter().filter(|e| matches!(e.system_value, 6 | 8)).map(|e| e.register).collect();
                    for b in [4u64, 9] {
                        let all: std::collections::BTreeSet<u64> =
                            u.iter().map(|x| x.0).chain(sv_regs.iter().copied()).flat_map(|r| (0..4).map(move |c| b + 4 * r as u64 + c)).collect();
                        if ins.is_subset(&all) {
                            *self.vs_base.entry(format!("{} matches v{b}+4r+c", xbox::hw_stage_name(hw))).or_default() += 1;
                        } else if b == 4 && hw == 4 && self.vs_input_examples.len() < 8 {
                            let extra: Vec<u64> = ins.difference(&all).copied().collect();
                            self.vs_input_examples.push(format!("v4 NO: {name}: reads {ins:?} outside: {extra:?} | ISG1 {}", sig_txt(&isg)));
                        }
                    }
                    *self.vs_base.entry(format!("{} total", xbox::hw_stage_name(hw))).or_default() += 1;
                    let a = ins.is_subset(&vec4);
                    let b = ins.is_subset(&packed);
                    self.vs_in_vec4 += a as usize;
                    self.vs_in_packed += b as usize;
let _ = b;
                }
            }
        }
    }
}

static HEADER_SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `xcl shaders --example MNEMONIC`
static EXAMPLE_FOR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

#[derive(Default)]
struct RecompAgg {
    programs: usize,
    complete: usize,
    wave32: usize,
    errors: usize,
    error_examples: Vec<String>,
    /// mnemonic → (instructions, programs that use it)
    missing: HashMap<String, (usize, usize)>,
    /// programs whose only missing mnemonic is this one
    sole: HashMap<String, usize>,
    /// "mnemonic: reason" → programs
    why: HashMap<String, usize>,
    /// (name, HLSL, profile) of complete translations kept for the DXC check
    samples: Vec<(String, String, &'static str)>,
    /// programs translated in a graphics mode (VS / PS interface from the container)
    by_mode: BTreeMap<String, (usize, usize)>,
    hlsl_bytes: usize,
    /// VS / PS (the driver picks the wave size): wave32 programs (run split in 64-lane waves)
    /// and wave64 programs whose cross-lane reads break in 32-lane waves (kind → programs)
    graphics_wave32: usize,
    graphics_wave64: usize,
    wave_hazards: BTreeMap<String, usize>,
    /// … and the reads of lanes ≥ 32 the 32-lane form resolves (kind → programs)
    wave_resolved: BTreeMap<String, usize>,
    /// `--example MNEMONIC`: listing of the first program missing it, its HLSL and profile
    example: Option<(String, String, &'static str)>,
}

impl RecompAgg {
    fn merge(&mut self, o: RecompAgg, keep: usize) {
        self.programs += o.programs;
        self.complete += o.complete;
        self.wave32 += o.wave32;
        self.errors += o.errors;
        self.error_examples.extend(o.error_examples);
        self.error_examples.truncate(10);
        for (k, (a, b)) in o.missing {
            let e = self.missing.entry(k).or_default();
            e.0 += a;
            e.1 += b;
        }
        for (k, v) in o.sole {
            *self.sole.entry(k).or_default() += v;
        }
        for (k, v) in o.why {
            *self.why.entry(k).or_default() += v;
        }
        self.samples.extend(o.samples);
        self.samples.truncate(keep);
        if self.example.is_none() {
            self.example = o.example;
        }
        self.graphics_wave32 += o.graphics_wave32;
        self.graphics_wave64 += o.graphics_wave64;
        for (k, v) in o.wave_hazards {
            *self.wave_hazards.entry(k).or_default() += v;
        }
        for (k, v) in o.wave_resolved {
            *self.wave_resolved.entry(k).or_default() += v;
        }
        for (k, (a, b)) in o.by_mode {
            let e = self.by_mode.entry(k).or_default();
            e.0 += a;
            e.1 += b;
        }
        self.hlsl_bytes += o.hlsl_bytes;
    }

    fn add(&mut self, name: &str, code: &[u8], keep: usize, mode: recomp::Mode) {
        // `XCL_RECOMP_ONLY=<hash>` (the hex prefix of an `XCL_DUMP_HLSL` file): that program alone
        if let Ok(only) = std::env::var("XCL_RECOMP_ONLY") {
            if format!("{:016x}", name_hash(name)) != only {
                return;
            }
        }
        self.programs += 1;
        let wave64 = recomp::detect_wave64(code);
        self.wave32 += !wave64 as usize;
        let (label, profile) = match &mode {
            recomp::Mode::Vertex(_) => ("VS", "vs_6_6"),
            recomp::Mode::Pixel(_) => ("PS", "ps_6_6"),
            recomp::Mode::Kernel => ("kernel", "cs_6_6"),
            recomp::Mode::Mesh(_) => ("MS", "cs_6_6"),
            recomp::Mode::Geometry(_) => ("GS", "cs_6_6"),
            recomp::Mode::Compute(_) => ("CS", "cs_6_6"),
        };
        if matches!(label, "VS" | "PS") {
            if wave64 {
                self.graphics_wave64 += 1;
                let (hz, resolved) = recomp::wave_lane_reads(code);
                for k in &hz {
                    *self.wave_hazards.entry(k.to_string()).or_default() += 1;
                }
                for k in &resolved {
                    *self.wave_resolved.entry(k.to_string()).or_default() += 1;
                }
                if !resolved.is_empty() {
                    *self.wave_resolved.entry("(programs)".into()).or_default() += 1;
                }
                if !hz.is_empty() {
                    *self.wave_hazards.entry("(programs at risk)".into()).or_default() += 1;
                }
            } else {
                self.graphics_wave32 += 1;
            }
        }
        // `XCL_RECOMP_WINDOWS=4`: as a title with unified memory translates (`pso::mem_windows`)
        let mem_windows = std::env::var("XCL_RECOMP_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        match recomp::translate(code, &recomp::Options { mode, wave64, mem_windows, ..Default::default() }) {
            Ok(t) => {
                let e = self.by_mode.entry(label.to_string()).or_default();
                e.0 += 1;
                e.1 += t.complete() as usize;
                self.hlsl_bytes += t.hlsl.len();
                for (m, n) in &t.unsupported {
                    let e = self.missing.entry(m.clone()).or_default();
                    e.0 += n;
                    e.1 += 1;
                }
                for k in t.unsupported_why.keys() {
                    *self.why.entry(k.clone()).or_default() += 1;
                }
                if t.unsupported.len() == 1 {
                    *self.sole.entry(t.unsupported.keys().next().unwrap().clone()).or_default() += 1;
                }
                if let (Some(want), None) = (EXAMPLE_FOR.get(), &self.example) {
                    // `--example MNEMONIC` (first program missing it) or `--example name=SUBSTR`
                    let hit = match want.strip_prefix("name=") {
                        Some(sub) => name.contains(sub),
                        None => t.unsupported.contains_key(want),
                    };
                    if hit {
                        let why: Vec<&String> = t.unsupported_why.keys().collect();
                        let mut txt = format!("{name} (mode {label}, wave{}): {why:?}
", if wave64 { 64 } else { 32 });
                        for l in crate::gpu::rdna2::disassemble(code, 0, false) {
                            txt.push_str(&format!("  {:04x}: {}
", l.pc, l.text));
                        }
                        self.example = Some((txt, t.hlsl.clone(), profile));
                    }
                }
                if t.complete() {
                    dump_hlsl(name, label, profile, &t.hlsl);
                    self.complete += 1;
                    if self.samples.len() < keep {
                        self.samples.push((name.to_string(), t.hlsl, profile));
                    }
                }
            }
            Err(e) => {
                self.errors += 1;
                if self.error_examples.len() < 10 {
                    self.error_examples.push(format!("{name}: {e}"));
                }
            }
        }
    }
}

#[derive(Default)]
struct Agg {
    /// translate programs (`--recomp`), keep this many complete HLSL samples
    recomp_on: bool,
    keep: usize,
    rc: RecompAgg,
    iface_on: bool,
    iface: IfaceAgg,
    total: usize,
    by_kind: BTreeMap<String, usize>,
    by_stage: BTreeMap<String, usize>,
    by_hw: BTreeMap<String, usize>,
    fssm: usize,
    programs: usize,
    data_blocks: usize,
    no_code: usize,
    failed: Vec<String>,
    failed_count: usize,
    /// examples: FSSM with several programs, programs with undecodable dwords
    odd: Vec<String>,
    odd_count: usize,
    inflate_errors: usize,
    insts: xbox::InstStats,
    /// in how many shaders a mnemonic appears
    shader_count: HashMap<String, usize>,
    /// distinct mnemonics of every console shader (for coverage)
    sets: Vec<Vec<String>>,
}

impl Agg {
    fn merge(&mut self, o: Agg) {
        self.total += o.total;
        for (k, v) in o.by_kind {
            *self.by_kind.entry(k).or_default() += v;
        }
        for (k, v) in o.by_stage {
            *self.by_stage.entry(k).or_default() += v;
        }
        for (k, v) in o.by_hw {
            *self.by_hw.entry(k).or_default() += v;
        }
        self.fssm += o.fssm;
        self.programs += o.programs;
        self.data_blocks += o.data_blocks;
        self.no_code += o.no_code;
        self.failed_count += o.failed_count;
        self.failed.extend(o.failed);
        self.failed.truncate(20);
        self.odd_count += o.odd_count;
        self.odd.extend(o.odd);
        self.odd.truncate(20);
        self.inflate_errors += o.inflate_errors;
        self.insts.merge(&o.insts);
        for (k, v) in o.shader_count {
            *self.shader_count.entry(k).or_default() += v;
        }
        self.sets.extend(o.sets);
        let keep = self.keep;
        self.rc.merge(o.rc, keep);
        self.iface.merge(o.iface);
    }

    fn add(&mut self, name: &str, blob: &[u8]) {
        self.total += 1;
        let (kind, _) = shader::classify(blob);
        *self.by_kind.entry(kind.label().to_string()).or_default() += 1;
        if kind != shader::Kind::XboxNative {
            return;
        }
        let Some(sh) = xbox::parse(blob) else { return };
        *self.by_stage.entry(sh.stage.clone()).or_default() += 1;
        let mut failed = !sh.notes.is_empty() || sh.fssm.is_empty();
        let mut st = xbox::InstStats::default();
        for f in &sh.fssm {
            self.fssm += 1;
            *self.by_hw.entry(f.stage_label()).or_default() += 1;
            self.programs += f.programs.len();
            self.data_blocks += f.data_blocks.len();
            if f.programs.is_empty() {
                self.no_code += 1;
            }
            failed |= f.error.is_some() || f.programs.is_empty() || !f.code_from_header;
        }
        st.add_shader(&sh);
        let multi = sh.fssm.iter().filter(|f| f.programs.len() > 1).count();
        if multi > 0 || st.undecoded > 0 {
            self.odd_count += 1;
            if self.odd.len() < 20 {
                self.odd.push(format!("{name}: FSSMs with several programs: {multi}, undecoded dwords: {}", st.undecoded));
            }
        }
        if failed {
            self.failed_count += 1;
            if self.failed.len() < 20 {
                let why: Vec<String> =
                    sh.notes.iter().cloned().chain(sh.fssm.iter().filter_map(|f| f.error.clone())).collect();
                self.failed.push(format!("{name}: {}", if why.is_empty() { "no code / code found by scanning".into() } else { why.join("; ") }));
            }
        }
        if self.iface_on {
            self.iface.add(name, blob, &sh);
        }
        // `--example name=SUBSTR` only needs the matching programs
        let wanted = match EXAMPLE_FOR.get().and_then(|w| w.strip_prefix("name=")) {
            Some(sub) => name.contains(sub),
            None => true,
        };
        if self.recomp_on && wanted {
            for (fi, f) in sh.fssm.iter().enumerate() {
                let mode = recomp::interface(blob, &sh, f).unwrap_or(recomp::Mode::Kernel);
                for p in &f.programs {
                    let keep = self.keep;
                    self.rc.add(&format!("{name} FSSM#{fi}"), &f.data[p.offset..p.offset + p.len], keep, mode.clone());
                    // `--example name=…`: FSSM header (up to the code) as dwords, once
                    if EXAMPLE_FOR.get().is_some_and(|w| w.starts_with("name=")) && !HEADER_SHOWN.swap(true, Ordering::Relaxed) {
                        let end = f.code.map(|c| c.0).unwrap_or(0x400).min(f.data.len()).min(0x800);
                        let mut txt = format!("FSSM header {name} (up to the code, {end:#x} B):
");
                        for (i, c) in f.data[..end].chunks(4).enumerate() {
                            if i % 8 == 0 {
                                txt.push_str(&format!("  {:04x}:", i * 4));
                            }
                            let v = u32::from_le_bytes([c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]);
                            txt.push_str(&format!(" {v:08x}"));
                            if i % 8 == 7 {
                                txt.push(10 as char);
                            }
                        }
                        println!("{txt}");
                    }
                }
            }
        }
        let set: Vec<String> = st.by_mnemonic.keys().cloned().collect();
        for m in &set {
            *self.shader_count.entry(m.clone()).or_default() += 1;
        }
        self.sets.push(set);
        self.insts.merge(&st);
    }
}

enum Input {
    Pack { data: Vec<u8>, records: Vec<pack::Record> },
    Files(Vec<PathBuf>),
}

impl Input {
    fn len(&self) -> usize {
        match self {
            Input::Pack { records, .. } => records.len(),
            Input::Files(f) => f.len(),
        }
    }

    /// (name, blob); blob is None when a record does not inflate
    fn get(&self, i: usize) -> (String, Option<Vec<u8>>) {
        match self {
            Input::Pack { data, records } => (format!("{} @{:#x}", records[i].name, records[i].offset), pack::inflate(data, &records[i])),
            Input::Files(f) => (f[i].display().to_string(), std::fs::read(&f[i]).ok()),
        }
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_files(&p, out);
            continue;
        }
        let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        // our own side files in <data>\shaders\ (FSSM dumps, listings)
        if name.ends_with(".txt") || name.contains(".fssm") {
            continue;
        }
        out.push(p);
    }
}

fn analyze(input: &Input, recomp_on: bool, keep: usize, iface_on: bool) -> Agg {
    let n = input.len();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(32);
    let parts: Vec<Agg> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut a = Agg { recomp_on, keep, iface_on, ..Default::default() };
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        match input.get(i) {
                            (name, Some(blob)) => a.add(&name, &blob),
                            (_, None) => a.inflate_errors += 1,
                        }
                    }
                    a
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut agg = Agg { recomp_on, keep, iface_on, ..Default::default() };
    for p in parts {
        agg.merge(p);
    }
    agg
}

fn fmt_map(m: &BTreeMap<String, usize>) -> String {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    v.iter().map(|(k, c)| format!("{k} {c}")).collect::<Vec<_>>().join(", ")
}

/// Mnemonics in implementation order: used by the most shaders first (ties: more
/// frequent first) – the order that unlocks whole shaders soonest.
fn by_shader_count(agg: &Agg, ranked: &[(String, u64)]) -> Vec<String> {
    let mut v: Vec<(&String, usize, u64)> = ranked.iter().map(|(m, c)| (m, agg.shader_count.get(m).copied().unwrap_or(0), *c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then_with(|| a.0.cmp(b.0)));
    v.into_iter().map(|x| x.0.clone()).collect()
}

/// Shaders whose every mnemonic is among the first `k` of `order`.
fn coverage(agg: &Agg, order: &[String], ks: &[usize]) -> Vec<(usize, usize)> {
    let rank: HashMap<&str, usize> = order.iter().enumerate().map(|(i, m)| (m.as_str(), i)).collect();
    let max_rank: Vec<usize> = agg.sets.iter().map(|s| s.iter().map(|m| rank[m.as_str()] + 1).max().unwrap_or(0)).collect();
    ks.iter().map(|&k| (k, max_rank.iter().filter(|&&r| r <= k).count())).collect()
}

pub fn run(path: &Path, args: &[String]) -> i32 {
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
    let top = opt("--top").unwrap_or(60);
    let json = args.iter().any(|a| a == "--json");
    let compile = opt("--compile").unwrap_or(0);
    let recomp_on = compile > 0 || args.iter().any(|a| a == "--recomp");
    let iface_on = args.iter().any(|a| a == "--iface");
    if let Some(m) = args.iter().position(|a| a == "--example").and_then(|i| args.get(i + 1)) {
        let _ = EXAMPLE_FOR.set(m.clone());
    }
    let t0 = std::time::Instant::now();

    let (input, what) = if path.is_dir() {
        let mut files = Vec::new();
        collect_files(path, &mut files);
        let n = files.len();
        (Input::Files(files), format!("folder, {n} files"))
    } else {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("xcl shaders: {}: {e}", path.display());
                return 2;
            }
        };
        if data.starts_with(pack::RPF9_MAGIC) {
            let records = pack::scan(&data, 0x40);
            let what = format!("RPF9 package {:.1} MB, {} shader records", data.len() as f64 / 1048576.0, records.len());
            (Input::Pack { data, records }, what)
        } else {
            let mut a = Agg { recomp_on, keep: compile, iface_on, ..Default::default() };
            a.add(&path.display().to_string(), &data);
            return print(path, "single file", a, top, json, t0, compile);
        }
    };
    if args.iter().any(|a| a == "--precompile") {
        let only = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)).map(|s| s.to_ascii_lowercase());
        let pipelines = args.iter().any(|a| a == "--pipelines");
        // the runtime's key: heap windows unless `XCL_HEAP_WINDOWS=0` (`pso::mem_windows`)
        let windows = opt("--windows").unwrap_or(4) as u32;
        let windows = if windows > 1 && recomp::pso::heap_windows() { windows | recomp::HEAP_WINDOWS } else { windows };
        return precompile(&input, opt("--limit").unwrap_or(usize::MAX), windows, opt("--split").map(|v| v != 0), only.as_deref(), pipelines, t0);
    }
    let agg = analyze(&input, recomp_on, compile, iface_on);
    drop(input);
    print(path, &what, agg, top, json, t0, compile)
}

/// `--precompile`: translate every VS / PS / MS / CS program and compile it into the DXIL cache
/// (`recomp::dxc::cache_dir()`), the same HLSL the runtime will ask for, so a title does not
/// wait for DXC on first use. Resumable: cached programs are skipped. GS halves need their
/// pipeline partner and are left to the runtime.
/// The cache key is the HLSL, so the options are the runtime's: `windows` memory windows as a
/// title's unified memory has (`pso::mem_windows`, 4; `--windows 1` for a title without it),
/// and the split wave64 form when this machine's GPU (`XCL_ADAPTER`) runs only 32-lane waves
/// (`pso::wave32_only_for_device`). Before, the programs were translated with one window and
/// no split form, so a title asked for other HLSL and compiled everything again. Still
/// different at run time: a PS's static samplers after its VS's (sampler base 0 here) and the
/// mesh pass-through, which the runtime builds for its pixel shader.
/// `--pipelines`: each compute program's pipeline made on the GPU as a title's would be (XCL's
/// ABI root signature with its static samplers): the driver keeps what it compiled in its own
/// cache, which the title then finds – a retail game's 1 MB-DXIL programs kept its loading
/// waiting for the driver's compiler for minutes. Nothing runs on the GPU.
struct Pipelines {
    dev: windows::Win32::Graphics::Direct3D12::ID3D12Device,
    signatures: Mutex<std::collections::HashMap<Vec<u32>, windows::Win32::Graphics::Direct3D12::ID3D12RootSignature>>,
    made: AtomicUsize,
    refused: AtomicUsize,
    /// driver compiles running and the most at once (`XCL_PIPELINE_THREADS`, 4): each of the
    /// biggest takes ~1.5 GB in the driver – fifteen at once took 24 GB
    running: (Mutex<usize>, std::sync::Condvar),
    most: usize,
}

// D3D12 devices and root signatures are free-threaded
unsafe impl Sync for Pipelines {}

impl Pipelines {
    fn new() -> Result<Pipelines, String> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let (adapter, _) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next().ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
            let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).map_err(|e| e.to_string())?;
            let most = std::env::var("XCL_PIPELINE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize).max(1);
            Ok(Pipelines { dev, signatures: Mutex::new(Default::default()), made: AtomicUsize::new(0), refused: AtomicUsize::new(0), running: (Mutex::new(0), std::sync::Condvar::new()), most })
        }
    }

    /// The compute pipeline of `c` (made and released).
    fn make(&self, c: &recomp::pso::Compiled) {
        use windows::Win32::Graphics::Direct3D12::*;
        let key: Vec<u32> = c.samplers.iter().flat_map(|(w, cmp)| w.iter().copied().chain([*cmp as u32])).collect();
        let rs = {
            let mut g = self.signatures.lock().unwrap();
            match g.get(&key) {
                Some(rs) => rs.clone(),
                None => match recomp::pso::abi_root_signature_with(&self.dev, true, &c.samplers) {
                    Ok(rs) => {
                        g.insert(key, rs.clone());
                        rs
                    }
                    Err(_) => {
                        self.refused.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                },
            }
        };
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs)),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: c.dxil.as_ptr() as *const _, BytecodeLength: c.dxil.len() },
            ..Default::default()
        };
        {
            let mut n = self.running.0.lock().unwrap();
            while *n >= self.most {
                n = self.running.1.wait(n).unwrap();
            }
            *n += 1;
        }
        let r: windows::core::Result<ID3D12PipelineState> = unsafe { self.dev.CreateComputePipelineState(&desc) };
        *self.running.0.lock().unwrap() -= 1;
        self.running.1.notify_one();
        drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
        let done = match r {
            Ok(_) => self.made.fetch_add(1, Ordering::Relaxed) + 1,
            Err(_) => self.refused.fetch_add(1, Ordering::Relaxed) + 1,
        };
        if done % 25 == 0 {
            eprintln!("  pipelines: {} made, {} refused", self.made.load(Ordering::Relaxed), self.refused.load(Ordering::Relaxed));
        }
    }
}

fn precompile(input: &Input, limit: usize, windows: u32, split: Option<bool>, only: Option<&str>, pipelines: bool, t0: std::time::Instant) -> i32 {
    let pipelines = if pipelines {
        match Pipelines::new() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("xcl shaders --pipelines: no GPU device: {e}");
                return 2;
            }
        }
    } else {
        None
    };
    let pipelines = pipelines.as_ref();
    let n = input.len().min(limit);
    let next = AtomicUsize::new(0);
    let (done, compiled, cached, failed, skipped) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    // the compiles run on the DXC pool; these threads translate and keep it fed
    let threads = recomp::dxc::compile_threads() + 2;
    // no title runs: only a little commit is kept free besides the compiles
    recomp::dxc::set_memory_reserve(2);
    // VS / PS: the 32-lane form wherever the GPU can run 32-lane waves (`pso::graphics_wave32`);
    // `--split` sets both
    let graphics_split = split.unwrap_or_else(recomp::runner::graphics_wave32);
    let split = split.unwrap_or_else(|| recomp::pso::split_wave64_forced() || recomp::runner::wave64_supported().is_ok_and(|w| !w));
    println!(
        "precompiling {n} shaders into {} ({} compile threads; {} memory windows{}{})",
        recomp::dxc::cache_dir().display(),
        recomp::dxc::compile_threads(),
        windows & !recomp::HEAP_WINDOWS,
        if windows & recomp::HEAP_WINDOWS != 0 { " from the descriptor heap" } else { "" },
        if split { ", split wave64 for a GPU with 32-lane waves" } else { "" }
    );
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let (name, blob) = input.get(i);
                let Some(blob) = blob else { continue };
                let Some(sh) = xbox::parse(&blob) else { continue };
                if only.is_some_and(|o| !sh.stage.eq_ignore_ascii_case(o)) {
                    done.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                // a compute program as a title's pipeline translates it (`pso::console_to_dxil_in`):
                // into the translation cache too, so the title does not translate it again
                if sh.fssm.iter().any(|f| matches!(recomp::interface(&blob, &sh, f), Some(recomp::Mode::Compute(_)))) {
                    match recomp::pso::console_to_dxil_in(&blob, recomp::pso::Stage::Compute, 0, split, windows) {
                        Ok(c) => {
                            if let Some(p) = pipelines {
                                p.make(&c);
                            }
                            compiled.fetch_add(1, Ordering::Relaxed)
                        }
                        Err(e) => {
                            let mut g = errors.lock().unwrap();
                            if g.len() < 10 {
                                g.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or(e.lines().next().unwrap_or(""))));
                            }
                            failed.fetch_add(1, Ordering::Relaxed)
                        }
                    };
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 200 == 0 {
                        eprintln!("  {d}/{n} ({:.0} s)", t0.elapsed().as_secs_f64());
                    }
                    continue;
                }
                for f in &sh.fssm {
                    let Some(mode) = recomp::interface(&blob, &sh, f) else { continue };
                    let jobs: Vec<(&str, Option<String>)> = match &mode {
                        recomp::Mode::Vertex(_) => vec![("vs_6_6", None)],
                        recomp::Mode::Pixel(_) => vec![("ps_6_6", None)],
                        recomp::Mode::Compute(_) => vec![("cs_6_6", None)],
                        recomp::Mode::Mesh(ms) => vec![("cs_6_6", None), ("ms_6_6", Some(recomp::mesh_passthrough(ms)))],
                        _ => {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        // as `pso` translates: with the title's memory windows (`console_translation`,
                        // `translate_mesh`)
                        let split = if matches!(mode, recomp::Mode::Vertex(_) | recomp::Mode::Pixel(_)) { graphics_split } else { split };
                        let opt = recomp::Options { mode: mode.clone(), wave64: recomp::detect_wave64(code), split_wave64: split, mem_windows: windows, ..Default::default() };
                        let Ok(t) = recomp::translate(code, &opt) else { continue };
                        if !t.complete() {
                            continue;
                        }
                        for (profile, src) in &jobs {
                            match recomp::dxc::compile_cached(src.as_deref().unwrap_or(&t.hlsl), "main", profile) {
                                Ok((_, true)) => cached.fetch_add(1, Ordering::Relaxed),
                                Ok(_) => compiled.fetch_add(1, Ordering::Relaxed),
                                Err(e) => {
                                    let mut g = errors.lock().unwrap();
                                    if g.len() < 10 {
                                        g.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or("")));
                                    }
                                    failed.fetch_add(1, Ordering::Relaxed)
                                }
                            };
                        }
                    }
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 200 == 0 {
                    eprintln!("  {d}/{n} ({:.0} s)", t0.elapsed().as_secs_f64());
                }
            });
        }
    });
    println!(
        "compiled {}, already cached {}, errors {}, skipped (GS – paired at pipeline creation) {}, time {:.1} s",
        compiled.into_inner(),
        cached.into_inner(),
        failed.into_inner(),
        skipped.into_inner(),
        t0.elapsed().as_secs_f64()
    );
    if let Some(p) = pipelines {
        println!("pipelines made on the GPU {}, refused {}", p.made.load(Ordering::Relaxed), p.refused.load(Ordering::Relaxed));
    }
    for e in errors.into_inner().unwrap() {
        println!("  {e}");
    }
    0
}

/// Compile the kept HLSL samples with DXC on all cores: (ok, failures with the first error line).
fn compile_samples(samples: &[(String, String, &'static str)]) -> (usize, Vec<String>) {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let times: Mutex<Vec<(f64, usize, String)>> = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let parts: Vec<(usize, Vec<String>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let (mut ok, mut bad) = (0usize, Vec::new());
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((name, hlsl, profile)) = samples.get(i) else { break };
                        let t0 = std::time::Instant::now();
                        match recomp::dxc::compile(hlsl, "main", profile, &[]) {
                            Ok(_) => ok += 1,
                            Err(e) => bad.push(format!("{name}: {}", e.lines().find(|l| l.contains("error")).unwrap_or(e.as_str()))),
                        }
                        times.lock().unwrap().push((t0.elapsed().as_secs_f64(), hlsl.len(), name.clone()));
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 100 == 0 {
                            eprintln!("DXC: {n}/{}", samples.len());
                        }
                    }
                    (ok, bad)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ok = parts.iter().map(|p| p.0).sum();
    let mut t = times.into_inner().unwrap();
    t.sort_by(|a, b| a.0.total_cmp(&b.0));
    if !t.is_empty() {
        let q = |f: f64| t[((t.len() - 1) as f64 * f) as usize].0;
        let sum: f64 = t.iter().map(|x| x.0).sum();
        eprintln!(
            "DXC time per shader: median {:.2} s, p90 {:.2} s, max {:.2} s, sum {:.0} s (threads: in parallel)",
            q(0.5),
            q(0.9),
            q(1.0),
            sum
        );
        for (secs, len, name) in t.iter().rev().take(5) {
            eprintln!("  {secs:.1} s, HLSL {} KB: {name}", len / 1024);
        }
    }
    (ok, parts.into_iter().flat_map(|p| p.1).collect())
}

#[allow(clippy::too_many_arguments)]
fn print(path: &Path, what: &str, agg: Agg, top: usize, json: bool, t0: std::time::Instant, compile: usize) -> i32 {
    let rc = &agg.rc;
    let compiled = if compile > 0 && !rc.samples.is_empty() { Some(compile_samples(&rc.samples)) } else { None };
    let mut missing: Vec<(&String, &(usize, usize))> = rc.missing.iter().collect();
    missing.sort_by(|a, b| b.1 .1.cmp(&a.1 .1).then_with(|| a.0.cmp(b.0)));
    let ranked = agg.insts.sorted();
    let ks: Vec<usize> = [10, 25, 50, 75, 100, 150, 200, 300].into_iter().filter(|&k| k < ranked.len()).chain([ranked.len()]).collect();
    let order = by_shader_count(&agg, &ranked);
    let cov = coverage(&agg, &order, &ks);
    let n_xbox = agg.sets.len();
    let total_inst = agg.insts.instructions.max(1);

    if json {
        let mnemonics: Vec<serde_json::Value> = ranked
            .iter()
            .map(|(m, c)| serde_json::json!({ "mnemonic": m, "count": c, "shaders": agg.shader_count.get(m).copied().unwrap_or(0) }))
            .collect();
        let v = serde_json::json!({
            "path": path.display().to_string(),
            "source": what,
            "shaders": agg.total,
            "inflate_errors": agg.inflate_errors,
            "by_kind": agg.by_kind,
            "by_dxil_stage": agg.by_stage,
            "by_hw_stage": agg.by_hw,
            "fssm": agg.fssm,
            "programs": agg.programs,
            "embedded_data_blocks": agg.data_blocks,
            "fssm_without_code": agg.no_code,
            "parse_problems": agg.failed_count,
            "parse_problem_examples": agg.failed,
            "unusual": agg.odd_count,
            "unusual_examples": agg.odd,
            "instructions": agg.insts.instructions,
            "undecoded_dwords": agg.insts.undecoded,
            "distinct_mnemonics": ranked.len(),
            "implementation_order": order,
            "coverage_top_k": cov.iter().map(|(k, c)| serde_json::json!({ "k": k, "shaders": c })).collect::<Vec<_>>(),
            "mnemonics": mnemonics,
            "recomp": agg.recomp_on.then(|| serde_json::json!({
                "programs": rc.programs,
                "complete": rc.complete,
                "wave32_programs": rc.wave32,
                "translate_errors": rc.errors,
                "translate_error_examples": rc.error_examples,
                "hlsl_bytes": rc.hlsl_bytes,
                "missing": missing.iter().map(|(m, (n, p))| serde_json::json!({
                    "mnemonic": m, "instructions": n, "programs": p, "sole_blocker": rc.sole.get(*m).copied().unwrap_or(0)
                })).collect::<Vec<_>>(),
                "dxc_compiled": compiled.as_ref().map(|c| c.0),
                "dxc_failed": compiled.as_ref().map(|c| c.1.clone()),
            })),
            "seconds": t0.elapsed().as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
        return 0;
    }

    println!("{} – {what}", path.display());
    println!("shaders: {}{}", agg.total, if agg.inflate_errors > 0 { format!(" (+{} records could not be unpacked)", agg.inflate_errors) } else { String::new() });
    println!("formats: {}", fmt_map(&agg.by_kind));
    if n_xbox == 0 {
        println!("no console containers (xbox-native)");
        return 0;
    }
    println!("stages (DXIL): {}", fmt_map(&agg.by_stage));
    println!("FSSM variants by hardware stage: {}", fmt_map(&agg.by_hw));
    println!("FSSM: {}, programs: {}, data blocks in code: {}, FSSM without code: {}", agg.fssm, agg.programs, agg.data_blocks, agg.no_code);
    println!(
        "RDNA2 instructions: {}, undecoded dwords: {} ({:.4}%), distinct mnemonics: {}",
        agg.insts.instructions,
        agg.insts.undecoded,
        agg.insts.undecoded as f64 * 100.0 / (agg.insts.instructions + agg.insts.undecoded).max(1) as f64,
        ranked.len()
    );
    if agg.failed_count > 0 {
        println!("parse problems: {} containers, e.g.:", agg.failed_count);
        for f in &agg.failed {
            println!("  {f}");
        }
    }
    if agg.odd_count > 0 {
        println!("unusual: {} containers (several programs in an FSSM or undecoded dwords), e.g.:", agg.odd_count);
        for o in agg.odd.iter().take(10) {
            println!("  {o}");
        }
    }
    println!("coverage: shaders made up only of the first K mnemonics (order: number of shaders it appears in):");
    for (k, c) in &cov {
        let last = order.get(k.saturating_sub(1)).map(String::as_str).unwrap_or("");
        println!("  K = {k:>4}: {c:>6} / {n_xbox} ({:>5.1}%)   K-th: {last}", *c as f64 * 100.0 / n_xbox as f64);
    }
    println!("top {} mnemonics (count, % of instructions, cumulative %, number of shaders):", top.min(ranked.len()));
    let mut cum = 0u64;
    for (i, (m, c)) in ranked.iter().take(top).enumerate() {
        cum += c;
        let sc = agg.shader_count.get(m).copied().unwrap_or(0);
        println!(
            "  {:>4}. {m:<34} {c:>10} {:>6.2}% {:>6.2}%  {sc:>6} ({:.0}%)",
            i + 1,
            *c as f64 * 100.0 / total_inst as f64,
            cum as f64 * 100.0 / total_inst as f64,
            sc as f64 * 100.0 / n_xbox as f64
        );
    }
    if agg.iface_on {
        let i = &agg.iface;
        println!("interface (hypotheses):");
        println!(
            "  PS: {} programs with v_interp; attributes ⊆ ISG1 registers: {}, = ISG1 registers: {}, attribute < number of non-SV elements (rank): {}",
            i.ps, i.ps_in_regs, i.ps_eq_regs, i.ps_rank
        );
        for e in &i.ps_examples {
            println!("    {e}");
        }
        println!(
            "  PS: SPI_PS_INPUT_ENA from FSSM+0x248 explains the v_interp I/J: {} / {}, all VGPRs read before write fit the layout: {} / {}",
            i.ps_ena_ij, i.ps, i.ps_ena_reads, i.ps
        );
        for e in &i.ps_ena_examples {
            println!("    {e}");
        }
        println!(
            "  VS: {} programs; exp param ⊆ OSG1 registers: {}, = OSG1 registers: {}, param < number of non-SV elements (rank): {}",
            i.vs, i.vs_param_in_regs, i.vs_param_eq, i.vs_rank
        );
        for e in &i.vs_param_examples {
            println!("    {e}");
        }
        println!(
            "  VS inputs (VGPRs ≥ v4 read before write): {} programs; matching v4+4·register: {}, with consecutive attributes packed: {}",
            i.vs_inputs, i.vs_in_vec4, i.vs_in_packed
        );
        for e in &i.vs_input_examples {
            println!("    {e}");
        }
        println!("  inputs by stage (API hardware stage: register read before write → number of programs; exports; s_sendmsg):");
        let sreg = |r: u64| match r {
            106 => "vcc_lo".to_string(),
            107 => "vcc_hi".to_string(),
            124 => "m0".to_string(),
            126 => "exec_lo".to_string(),
            127 => "exec_hi".to_string(),
            r => format!("s{r}"),
        };
        let tgt = |t: u64| match t {
            0..=7 => format!("mrt{t}"),
            8 => "mrtz".into(),
            9 => "null".into(),
            12..=15 => format!("pos{}", t - 12),
            20 => "prim".into(),
            32..=63 => format!("param{}", t - 32),
            t => format!("tgt{t}"),
        };
        for (k, e) in &i.inputs {
            let pct = |n: usize| n * 100 / e.programs.max(1);
            let list = |m: &BTreeMap<u64, usize>, f: &dyn Fn(u64) -> String| {
                m.iter().filter(|(_, &n)| pct(n) >= 1).map(|(r, &n)| format!("{} {}%", f(*r), pct(n))).collect::<Vec<_>>().join(", ")
            };
            println!("    {k}: {} programs", e.programs);
            println!("      VGPR: {}", list(&e.vgpr, &|r| format!("v{r}")));
            println!("      SGPR: {}", list(&e.sgpr, &sreg));
            println!("      exp: {}", list(&e.exports, &tgt));
            println!("      sendmsg: {}", list(&e.sendmsg, &|m| m.to_string()));
            let mut ps: Vec<_> = e.psv.iter().collect();
            ps.sort_by(|a, b| b.1.cmp(a.1));
            println!("      PSV0: {}", ps.iter().take(6).map(|(k, n)| format!("{k}: {n}")).collect::<Vec<_>>().join("; "));
            let mut fu: Vec<_> = e.first_use.iter().collect();
            fu.sort_by(|a, b| a.0.split(':').next().cmp(&b.0.split(':').next()).then(b.1.cmp(a.1)));
            let mut shown: BTreeMap<String, usize> = BTreeMap::new();
            for (k, n) in fu {
                let reg = k.split(':').next().unwrap_or("").to_string();
                let c = shown.entry(reg).or_default();
                if *c < 4 && *n * 100 >= e.programs {
                    println!("      first use {k}  ({n})");
                }
                *c += 1;
            }
        }
        println!("  VS input layout (smallest base B: inputs ⊆ B + 4·register + component): {}", fmt_map(&i.vs_base));
    }
    if agg.recomp_on {
        println!(
            "recompiler (kernel mode): programs {}, fully translated {} ({:.1}%), translation errors {}, wave32: {}, HLSL: {:.1} MB",
            rc.programs,
            rc.complete,
            rc.complete as f64 * 100.0 / rc.programs.max(1) as f64,
            rc.errors,
            rc.wave32,
            rc.hlsl_bytes as f64 / 1048576.0
        );
        for e in &rc.error_examples {
            println!("  error: {e}");
        }
        for (m, (n, c)) in &rc.by_mode {
            println!("  mode {m}: {n} programs, complete {c} ({:.1}%)", *c as f64 * 100.0 / (*n).max(1) as f64);
        }
        println!("missing instructions (number of programs; programs it blocks as the only missing one; occurrences):");
        for (m, (n, p)) in missing.iter().take(top) {
            println!("  {m:<34} {p:>6} {:>6} {n:>9}", rc.sole.get(*m).copied().unwrap_or(0));
        }
        let mut why: Vec<(&String, &usize)> = rc.why.iter().collect();
        why.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        println!("reasons (instruction: reason → number of programs):");
        for (k, v) in why.iter().take(top) {
            println!("  {k:<60} {v:>6}");
        }
        println!(
            "VS/PS and the wave size (the driver picks it): wave32 {} (handled – the halves of a 64-wide wave run separately), wave64 {}; hazards in 32-thread waves: {:?}; reads of lanes ≥ 32 resolved (32-lane GPUs): {:?}",
            rc.graphics_wave32, rc.graphics_wave64, rc.wave_hazards, rc.wave_resolved
        );
        if let Some((x, hlsl, profile)) = &rc.example {
            println!("example:
{x}");
            match recomp::dxc::compile(hlsl, "main", profile, &[]) {
                Ok(b) => println!("DXC ({profile}): OK, {} B DXIL", b.len()),
                Err(e) => println!("DXC ({profile}): {}", e.split("--- HLSL ---").next().unwrap_or("").trim()),
            }
            if let Ok(path) = std::env::var("XCL_EXAMPLE_HLSL") {
                let _ = std::fs::write(&path, hlsl);
                println!("HLSL saved: {path}");
            }
        }
        if let Some((ok, bad)) = &compiled {
            println!("DXC (vs/ps/cs_6_6) on a sample of complete translations: {ok} OK, {} errors", bad.len());
            for b in bad.iter().take(10) {
                println!("  {b}");
            }
        }
    }
    println!("time: {:.1} s", t0.elapsed().as_secs_f64());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::xbox::tests::{fssm, sample, CODE};

    #[test]
    fn aggregates_console_shaders() {
        let mut a = Agg::default();
        a.add("PS_a", &sample(0, &[fssm(&CODE, &[(5, 1)])]));
        a.add("VS_b", &sample(1, &[fssm(&CODE, &[(2, 0)]), fssm(&CODE, &[(4, 0)])]));
        a.add("junk", b"not a shader");
        assert_eq!(a.total, 3);
        assert_eq!(a.by_kind.get("xbox-native"), Some(&2));
        assert_eq!(a.by_hw.get("ES"), Some(&1));
        assert_eq!(a.fssm, 3);
        assert_eq!(a.programs, 3);
        assert_eq!(a.insts.instructions, 9);
        assert_eq!(a.failed_count, 0, "{:?}", a.failed);
        assert_eq!(a.shader_count.get("s_endpgm"), Some(&2));
        let ranked = a.insts.sorted();
        let order = by_shader_count(&a, &ranked);
        assert_eq!(order.len(), 3);
        let cov = coverage(&a, &order, &[1, 3]);
        assert_eq!(cov, vec![(1, 0), (3, 2)]);
    }

    /// Real data check (skipped unless XCL_SAMPLE_RPF points at an RPF9 shader pack).
    #[test]
    fn retail_pack_sample() {
        let Ok(p) = std::env::var("XCL_SAMPLE_RPF") else {
            println!("XCL_SAMPLE_RPF not set – skipping");
            return;
        };
        let data = std::fs::read(&p).unwrap();
        let mut records = pack::scan(&data, 0x40);
        records.truncate(300);
        let agg = analyze(&Input::Pack { data, records }, false, 0, false);
        assert_eq!(agg.inflate_errors, 0);
        assert_eq!(agg.by_kind.get("xbox-native"), Some(&agg.total), "{:?}", agg.by_kind);
        assert_eq!(agg.failed_count, 0, "{:?}", agg.failed);
        assert_eq!(agg.insts.undecoded, 0);
    }
}

/// `XCL_DUMP_HLSL=<folder>` (`--recomp`): every `XCL_DUMP_EVERY`-th complete translation (by a
/// hash of its name, default 100) is written there – samples for timing DXC on its own.
fn dump_hlsl(name: &str, label: &str, profile: &str, hlsl: &str) {
    let Some(dir) = std::env::var_os("XCL_DUMP_HLSL") else { return };
    let every: u64 = std::env::var("XCL_DUMP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(100).max(1);
    let h = name_hash(name);
    if h % every != 0 {
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{h:016x}_{label}_{profile}.hlsl")), hlsl);
}

/// FNV-1a of a program's name (`XCL_DUMP_HLSL` file names, `XCL_RECOMP_ONLY`).
fn name_hash(name: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}
