//! Console shaders in pipeline states: a shader blob a title passes to Create*PipelineState
//! (DXBC container with XDAT/XBCH/FSSM, no usable DXIL) becomes DXIL through the recompiler,
//! and the pipeline gets XCL's shader ABI root signature instead of the title's one:
//! root UAVs u0 `io` (initial SGPRs = user data), u1 `res`, u2 `mem` (flat memory), plus the
//! directly indexed descriptor heaps (images / samplers). Mapping the title's root arguments
//! onto that ABI at draw time is the next step (needs the console runtime's user-data layout).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

use super::{dxc, interface, translate, Mode, Options};
use crate::gpu::xbox;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Pixel,
    Compute,
}

/// A console (xbox-native) shader container?
pub fn is_console(blob: &[u8]) -> bool {
    xbox::is_xbox(blob) || fspx(blob).is_some()
}

/// A compute program in the console driver's own container (an `ACCC` part holding `FSPX`),
/// what AMD FidelityFX (FSR4) creates its passes from on the console: no DXIL, the RDNA2
/// code with its thread geometry and COMPUTE_PGM_RSRC2.
pub struct Fspx<'a> {
    pub threads: [u32; 3],
    pub rsrc1: u32,
    pub rsrc2: u32,
    /// RDNA2 code, from its first instruction to the last `s_endpgm`
    pub code: &'a [u8],
}

/// `FSPX`: 'FSPX', version, entry count, size, then 16-byte entries (type, offset, size, 0)
/// from +0x10, offsets from the start of FSPX. Type 0 (0x1C bytes): numthreads as u16 x3 at
/// +0x0C, RSRC1 at +0x14, RSRC2 at +0x18; type 1: a 0x14-byte hash; type 6: the code behind
/// a 0x24-byte header (the v_nop hash prologue); type 7: resource declarations (32-byte
/// records). Read from FSR4's pipelines (the console upscaler's "NewBE" path).
pub fn fspx(blob: &[u8]) -> Option<Fspx<'_>> {
    let parts = crate::gpu::shader::container_parts(blob)?;
    let p = parts.iter().find(|p| p.fourcc == "ACCC")?;
    let part = blob.get(p.offset..p.offset + p.size)?;
    let u32_at = |b: &[u8], o: usize| b.get(o..o + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()));
    if part.get(..4)? != b"FSPX" {
        return None;
    }
    let n = u32_at(part, 8)? as usize;
    let entry = |ty: u32| -> Option<&[u8]> {
        (0..n.min(32)).find_map(|e| {
            let at = 0x10 + 16 * e;
            (u32_at(part, at)? == ty).then(|| {
                let (o, s) = (u32_at(part, at + 4)? as usize, u32_at(part, at + 8)? as usize);
                part.get(o..o.checked_add(s)?)
            })?
        })
    };
    let head = entry(0)?;
    let u16_at = |o: usize| head.get(o..o + 2).map(|x| u16::from_le_bytes(x.try_into().unwrap()) as u32);
    let threads = [u16_at(0x0C)?, u16_at(0x0E)?, u16_at(0x10)?];
    if threads.iter().any(|&t| t == 0 || t > 1024) || threads.iter().product::<u32>() > 1024 {
        return None;
    }
    let code = entry(6)?.get(0x24..)?;
    // up to the last s_endpgm (the driver pads with s_code_end)
    let end = (0..code.len() / 4).rev().find(|&i| u32_at(code, 4 * i) == Some(0xBF81_0000))?;
    Some(Fspx { threads, rsrc1: u32_at(head, 0x14)?, rsrc2: u32_at(head, 0x18)?, code: &code[..4 * (end + 1)] })
}

/// Memory windows of the title's unified memory (1 without it), with `HEAP_WINDOWS` when
/// shaders take them from the descriptor heap (`heap_windows`).
pub fn mem_windows() -> u32 {
    let n = crate::gpu::uma::get().map_or(1, |u| u.shader_windows() as u32);
    if n > 1 && heap_windows() && !SWITCH_FORM.with(|s| s.get()) {
        n | super::HEAP_WINDOWS
    } else {
        n
    }
}

thread_local! {
    /// this thread translates in the switch form (`in_switch_form`)
    static SWITCH_FORM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with this thread's translations in the switch form over root UAVs (`mem_windows`
/// without `HEAP_WINDOWS`, so other cache keys): for a pipeline the driver would not build with
/// heap windows. The root signature and the binds are the same for both forms.
pub fn in_switch_form<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SWITCH_FORM.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SWITCH_FORM.with(|s| s.replace(true)));
    f()
}

/// Whether translated shaders reach the memory windows through the shared descriptor heap
/// (`Options::heap_windows`; descriptors `MEM_HEAP_BASE`.. made by `console_heap::set_window_views`)
/// instead of a `switch` over root UAVs. On unless `XCL_HEAP_WINDOWS=0` or the views could not
/// be made (no shared heap: then the switch, as before).
pub fn heap_windows() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("XCL_HEAP_WINDOWS").map_or(true, |v| v != "0")) && !crate::hle::d3d12x::console_heap::window_views_failed()
}

/// A static sampler of a translated program: (S#, comparison) – see `Translation`.
pub type StaticSampler = ([u32; 4], bool);

/// A console shader as PC DXIL and the static samplers it needs in its root signature.
#[derive(Clone)]
pub struct Compiled {
    pub dxil: Vec<u8>,
    pub samplers: Vec<StaticSampler>,
}

/// Translate a console shader for `stage` to signed DXIL (DXC, disk-cached); its static
/// samplers take registers from `sampler_base` (space 1).
/// Fails when the blob is not a console shader or the program is not fully translated.
pub fn console_to_dxil(blob: &[u8], stage: Stage, sampler_base: u32) -> Result<Compiled, String> {
    console_to_dxil_for(blob, stage, sampler_base, false)
}

/// `console_to_dxil` for a GPU that runs only 32-lane waves when `split_wave64`
/// (`wave32_only_for_device`): a wave64 compute program gets the split form
/// (`Options::split_wave64`) instead of `[WaveSize(64)]`, which such a driver rejects.
/// Finished shaders are also kept in memory (`ShaderCache`): pipelines share their shaders.
pub fn console_to_dxil_for(blob: &[u8], stage: Stage, sampler_base: u32, split_wave64: bool) -> Result<Compiled, String> {
    console_to_dxil_in(blob, stage, sampler_base, split_wave64, mem_windows())
}

/// `console_to_dxil_for` with the title's memory windows given (`xcl shaders --precompile`
/// fills the caches without a title: the same keys as at its run).
pub fn console_to_dxil_in(blob: &[u8], stage: Stage, sampler_base: u32, split_wave64: bool, windows: u32) -> Result<Compiled, String> {
    let compile_one = || compile_console_shader(blob, stage, sampler_base, windows, split_wave64);
    if !cache_allowed(stage, |name| std::env::var_os(name).is_some()) {
        return compile_one();
    }
    static CACHE: OnceLock<Mutex<ShaderCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(ShaderCache::new(64 << 20)));
    let key = ShaderKey {
        hash: crate::gpu::shader::fnv64(blob),
        stage,
        sampler_base,
        windows,
        split_wave64,
        cache_dir: dxc::cache_dir(),
    };
    let from_disk = || translated::cached(&dxc::cache_dir(), blob, stage, sampler_base, windows, split_wave64, compile_one);
    cached_shader(cache, key, blob, from_disk).map(|shader| shader.as_ref().clone())
}

/// Translations on disk: a console program's (DXIL, static samplers) by its bytes, so a run
/// does not translate again what an earlier run of the same XCL build translated. The DXIL
/// cache is keyed by the generated HLSL, so it needs the translation first: a retail game's
/// ~2 800 programs (611 of more than 10 000 instructions) took its loading threads 0.1 - 0.3 s
/// each at every start. Keyed by two hashes and the length of the bytes, the stage and the
/// translation parameters; translator sources and the DXC toolchain identify compatible builds.
/// `XCL_TRANSLATED_CACHE=0` turns it off.
mod translated {
    use super::{Compiled, Stage};
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    const MAGIC: &[u8; 8] = b"XCLTRAN2";

    pub(super) fn identity_for(source: &str, toolchain: &str) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for part in ["xcl-translated-cache-v2", source, toolchain] {
            for byte in part.bytes().chain([0]) {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h | 1
    }

    /// Translation semantics and the DXC DLLs, independent of link metadata and build profile.
    /// Zero means the disk cache is disabled.
    fn identity() -> u64 {
        static I: OnceLock<u64> = OnceLock::new();
        *I.get_or_init(|| {
            if std::env::var("XCL_TRANSLATED_CACHE").is_ok_and(|v| v == "0") {
                return 0;
            }
            identity_for(env!("XCL_TRANSLATOR_SOURCE_ID"), super::dxc::toolchain_identity())
        })
    }

    /// a second hash of the bytes, independent of `fnv64`'s
    fn hash2(b: &[u8]) -> u64 {
        b.iter().fold(0x9E37_79B9_7F4A_7C15u64, |h, &x| (h ^ x as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29))
    }

    pub(super) fn file(dir: &Path, ident: u64, blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split: bool) -> PathBuf {
        dir.join("translated")
            .join("v2")
            .join(format!("{ident:016x}"))
            .join(format!("{:016x}_{stage:?}_{sampler_base}_{windows}_{}.bin", crate::gpu::shader::fnv64(blob), split as u8))
    }

    fn header(ident: u64, blob: &[u8]) -> Vec<u8> {
        let mut h = MAGIC.to_vec();
        for v in [ident, blob.len() as u64, hash2(blob)] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h
    }

    pub(super) fn load_in(path: &Path, ident: u64, blob: &[u8]) -> Option<Compiled> {
        let b = std::fs::read(path).ok()?;
        let head = header(ident, blob);
        let mut r = b.strip_prefix(head.as_slice())?;
        let u32_ = |r: &mut &[u8]| -> Option<u32> {
            let (v, rest) = r.split_first_chunk::<4>()?;
            *r = rest;
            Some(u32::from_le_bytes(*v))
        };
        let n = u32_(&mut r)? as usize;
        let mut samplers = Vec::with_capacity(n.min(64));
        for _ in 0..n {
            let s = [u32_(&mut r)?, u32_(&mut r)?, u32_(&mut r)?, u32_(&mut r)?];
            let (&cmp, rest) = r.split_first()?;
            r = rest;
            samplers.push((s, cmp != 0));
        }
        let len = u32_(&mut r)? as usize;
        (r.len() == len && len > 0).then(|| Compiled { dxil: r.to_vec(), samplers })
    }

    pub(super) fn store_in(path: &Path, ident: u64, blob: &[u8], c: &Compiled) {
        let mut b = header(ident, blob);
        b.extend_from_slice(&(c.samplers.len() as u32).to_le_bytes());
        for (s, cmp) in &c.samplers {
            s.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
            b.push(*cmp as u8);
        }
        b.extend_from_slice(&(c.dxil.len() as u32).to_le_bytes());
        b.extend_from_slice(&c.dxil);
        let Some(dir) = path.parent() else { return };
        let tmp = dir.join(format!("{}.{}.{:?}.tmp", path.file_name().unwrap_or_default().to_string_lossy(), std::process::id(), std::thread::current().id()));
        // written whole, then renamed: a reader never sees half a file
        if std::fs::create_dir_all(dir).is_ok() && std::fs::write(&tmp, &b).is_ok() && std::fs::rename(&tmp, path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    pub(super) fn cached_in(
        dir: &Path,
        ident: u64,
        blob: &[u8],
        stage: Stage,
        sampler_base: u32,
        windows: u32,
        split: bool,
        translate: impl FnOnce() -> Result<Compiled, String>,
    ) -> Result<Compiled, String> {
        if ident == 0 {
            return translate();
        }
        let path = file(dir, ident, blob, stage, sampler_base, windows, split);
        if let Some(c) = load_in(&path, ident, blob) {
            super::dxc::note_memory_hit();
            return Ok(c);
        }
        let c = translate()?;
        store_in(&path, ident, blob, &c);
        Ok(c)
    }

    /// `translate` through the disk (`identity()` zero: not used).
    pub(super) fn cached(
        dir: &Path,
        blob: &[u8],
        stage: Stage,
        sampler_base: u32,
        windows: u32,
        split: bool,
        translate: impl FnOnce() -> Result<Compiled, String>,
    ) -> Result<Compiled, String> {
        cached_in(dir, identity(), blob, stage, sampler_base, windows, split, translate)
    }
}

// Changes here, in `fspx`, or in `console_translation` that alter the cached result must bump
// `TRANSLATION_GLUE_REVISION` in build.rs; cache serialization changes need not bump it.
fn compile_console_shader(blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split_wave64: bool) -> Result<Compiled, String> {
    let (mut t, profile) = console_translation(blob, stage, sampler_base, windows, split_wave64)?;
    if let Some([r, g, b]) = (stage == Stage::Pixel).then(|| debug_color(blob)).flatten() {
        t.hlsl = t.hlsl.replacen("    return xo;\n}", &format!("    xo.t0 = float4({r:.3}, {g:.3}, {b:.3}, 1.0);\n    return xo;\n}}"), 1);
    }
    // `XCL_DEBUG_LOD0=hash,…`: those pixel shaders sample at LOD 0 instead of with implicit
    // derivatives – tells derivative (helper lane) trouble apart from the rest
    if stage == Stage::Pixel && std::env::var("XCL_DEBUG_LOD0").is_ok_and(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).any(|h| h == crate::gpu::shader::fnv64(blob))) {
        t.hlsl = sample_at_lod0(&t.hlsl);
    }
    // `XCL_DEBUG_PS_SUB=hash:from=>to||from=>to…` (one pixel shader): every `from` in its HLSL
    // becomes `to`
    if let Some(subs) = (stage == Stage::Pixel).then(|| std::env::var("XCL_DEBUG_PS_SUB").ok()).flatten().and_then(|v| {
        let (h, rest) = v.split_once(':')?;
        (u64::from_str_radix(h.trim(), 16).ok()? == crate::gpu::shader::fnv64(blob)).then(|| rest.to_string())
    }) {
        for sub in subs.split("||") {
            let Some((from, to)) = sub.split_once("=>") else { continue };
            let n = t.hlsl.matches(from).count();
            crate::xlog_info!("[SHADER] XCL_DEBUG_PS_SUB: {n} occurrence(s) of {from:?} replaced");
            t.hlsl = t.hlsl.replace(from, to);
        }
    }
    // `XCL_DEBUG_PS_EXPR=hash:expression` (one pixel shader): its target 0 gets that HLSL
    // expression, e.g. `float4(xi.a1.rgb, 1)` – an input or a register made visible
    if let Some(expr) = (stage == Stage::Pixel).then(|| std::env::var("XCL_DEBUG_PS_EXPR").ok()).flatten().and_then(|v| {
        let (h, e) = v.split_once(':')?;
        (u64::from_str_radix(h.trim(), 16).ok()? == crate::gpu::shader::fnv64(blob)).then(|| e.to_string())
    }) {
        t.hlsl = t.hlsl.replacen("    if (!xcl_live) discard;\n    return xo;\n}", &format!("    xo.t0 = {expr};\n    return xo;\n}}"), 1);
    }
    let dxil = dxc::compile_cached(&t.hlsl, "main", profile)?.0;
    Ok(Compiled { dxil, samplers: t.static_samplers })
}

/// Every `X_T.Sample(args)` of generated HLSL as `X_T.SampleLevel(args, 0.0f)`.
fn sample_at_lod0(hlsl: &str) -> String {
    let mut out = String::with_capacity(hlsl.len());
    let mut rest = hlsl;
    while let Some(at) = rest.find("_T.Sample(") {
        let open = at + "_T.Sample(".len();
        out.push_str(&rest[..at]);
        out.push_str("_T.SampleLevel(");
        let mut depth = 1;
        let mut end = open;
        for (k, ch) in rest[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&rest[open..end]);
        out.push_str(", 0.0f)");
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `XCL_DEBUG_MAGENTA=hash[:rrggbb],…` (dump hashes, hex; magenta without a colour): those
/// pixel shaders write that opaque colour to target 0 wherever they run – where a draw lands,
/// told apart from what it shades (a colour per shader: which one drew a pixel).
fn debug_color(blob: &[u8]) -> Option<[f32; 3]> {
    static S: std::sync::OnceLock<Vec<(u64, [f32; 3])>> = std::sync::OnceLock::new();
    let wanted = S.get_or_init(|| {
        std::env::var("XCL_DEBUG_MAGENTA")
            .unwrap_or_default()
            .split(',')
            .filter_map(|item| {
                let item = item.trim();
                let (h, c) = item.split_once(':').unwrap_or((item, "ff00ff"));
                let c = u32::from_str_radix(c, 16).ok()?;
                let ch = |shift: u32| ((c >> shift) & 0xFF) as f32 / 255.0;
                Some((u64::from_str_radix(h, 16).ok()?, [ch(16), ch(8), ch(0)]))
            })
            .collect()
    });
    let h = crate::gpu::shader::fnv64(blob);
    wanted.iter().find(|(w, _)| *w == h).map(|(_, c)| *c)
}

/// The HLSL (and its DXC profile) `console_to_dxil` compiles.
pub fn console_to_hlsl(blob: &[u8], stage: Stage) -> Result<(String, &'static str), String> {
    console_translation(blob, stage, 0, mem_windows(), false).map(|(t, p)| (t.hlsl, p))
}

#[derive(Clone, PartialEq, Eq)]
struct ShaderKey {
    hash: u64,
    stage: Stage,
    sampler_base: u32,
    windows: u32,
    split_wave64: bool,
    cache_dir: PathBuf,
}

struct ShaderEntry {
    key: ShaderKey,
    blob: Vec<u8>,
    shader: Arc<Compiled>,
}

impl ShaderEntry {
    fn bytes(&self) -> usize {
        self.blob.len()
            + self.key.cache_dir.capacity()
            + self.shader.dxil.capacity()
            + self.shader.samplers.capacity() * std::mem::size_of::<StaticSampler>()
    }
}

struct ShaderCache {
    entries: VecDeque<ShaderEntry>,
    bytes: usize,
    limit: usize,
}

impl ShaderCache {
    fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }

    fn get(&mut self, key: &ShaderKey, blob: &[u8]) -> Option<Arc<Compiled>> {
        let at = self
            .entries
            .iter()
            .rposition(|e| &e.key == key && e.blob == blob)?;
        let entry = self.entries.remove(at)?;
        let shader = entry.shader.clone();
        self.entries.push_back(entry);
        Some(shader)
    }

    fn insert(&mut self, key: ShaderKey, blob: &[u8], shader: Arc<Compiled>) {
        let bytes = blob.len()
            + key.cache_dir.capacity()
            + shader.dxil.capacity()
            + shader.samplers.capacity() * std::mem::size_of::<StaticSampler>();
        if bytes > self.limit {
            return;
        }
        while self.bytes + bytes > self.limit || self.entries.len() >= 128 {
            let Some(old) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= old.bytes();
        }
        self.entries.push_back(ShaderEntry {
            key,
            blob: blob.to_vec(),
            shader,
        });
        self.bytes += bytes;
    }
}

fn cached_shader(
    cache: &Mutex<ShaderCache>,
    key: ShaderKey,
    blob: &[u8],
    compile_one: impl FnOnce() -> Result<Compiled, String>,
) -> Result<Arc<Compiled>, String> {
    if let Some(shader) = cache.lock().unwrap().get(&key, blob) {
        dxc::note_memory_hit();
        return Ok(shader);
    }
    // Keep compilation outside the cache lock, including cached-only probes.
    let shader = Arc::new(compile_one()?);
    let mut cache = cache.lock().unwrap();
    if let Some(existing) = cache.get(&key, blob) {
        return Ok(existing);
    }
    cache.insert(key, blob, shader.clone());
    Ok(shader)
}

fn cache_allowed(stage: Stage, present: impl Fn(&str) -> bool) -> bool {
    !["XCL_RECOMP_NO_STRUCTURED", "XCL_RECOMP_WHY", "XCL_X64_ALL_MASKS", "XCL_WAVE64_SPIN"]
        .iter()
        .any(|name| present(name))
        && (stage != Stage::Pixel
            || ![
                "XCL_DEBUG_MAGENTA",
                "XCL_DEBUG_LOD0",
                "XCL_DEBUG_PS_SUB",
                "XCL_DEBUG_PS_EXPR",
            ]
            .iter()
            .any(|name| present(name)))
}

/// User SGPRs of a console compute shader (COMPUTE_PGM_RSRC2 in FSPX or FSSM):
/// s0…s3 are the driver's, so what is above 4 is the root arguments its compiler passed in
/// registers. None: not a console compute shader.
pub fn compute_user_sgprs(blob: &[u8]) -> Option<u32> {
    if let Some(f) = fspx(blob) {
        return Some(super::CsInterface::from_rsrc2(f.threads, f.rsrc2).user_sgprs);
    }
    let sh = xbox::parse(blob)?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6))?;
    match interface(blob, &sh, f)? {
        Mode::Compute(cs) => Some(cs.user_sgprs),
        _ => None,
    }
}

fn console_translation(blob: &[u8], stage: Stage, sampler_base: u32, windows: u32, split_wave64: bool) -> Result<(super::Translation, &'static str), String> {
    if let Some(f) = fspx(blob) {
        if stage != Stage::Compute {
            return Err("an FSPX (driver) container in a graphics slot".into());
        }
        let cs = super::CsInterface::from_rsrc2(f.threads, f.rsrc2);
        let t = translate(f.code, &Options { mode: Mode::Compute(cs), wave64: super::detect_wave64(f.code), split_wave64, static_sampler_base: sampler_base, mem_windows: windows, ..Default::default() })?;
        if !t.complete() {
            let missing: Vec<&String> = t.unsupported.keys().collect();
            return Err(format!("incomplete translation, missing: {missing:?}"));
        }
        return Ok((t, "cs_6_6"));
    }
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    // the hardware variant for this pipeline slot: VS (not the ES variant used with GS), PS
    let want = match stage {
        Stage::Vertex => 4,
        Stage::Pixel => 5,
        Stage::Compute => 6,
    };
    let f = sh
        .fssm
        .iter()
        .find(|f| f.stages.first().is_some_and(|s| s.hw == want))
        .ok_or_else(|| format!("no hardware variant {}", xbox::hw_stage_name(want)))?;
    let mode = interface(blob, &sh, f).ok_or("unknown shader interface")?;
    let profile = match (&mode, stage) {
        (Mode::Vertex(_), Stage::Vertex) => "vs_6_6",
        (Mode::Pixel(_), Stage::Pixel) => "ps_6_6",
        (Mode::Compute(_), Stage::Compute) => "cs_6_6",
        _ => return Err("the shader stage does not match its slot in the PSO".into()),
    };
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    let t = translate(code, &Options { mode, wave64: super::detect_wave64(code), split_wave64, static_sampler_base: sampler_base, mem_windows: windows, ..Default::default() })?;
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    Ok((t, profile))
}

/// A console geometry stage as XCL runs it, in two passes: the console program as a compute
/// shader writes one record per thread group (vertices + primitives, `record`) into `res`,
/// then a pass-through mesh shader rasterises the records with the pixel shader.
pub struct TwoPass {
    /// compute pass (cs_6_6 DXIL)
    pub cs: Vec<u8>,
    /// pass-through mesh shader (ms_6_6 DXIL)
    pub ms: Vec<u8>,
    pub record: super::MsInterface,
    /// mesh groups of the pass-through per record (> 1: the record is over D3D12's mesh
    /// output limit and goes out in runs of triangles)
    pub mesh_groups_per_record: u32,
    /// static samplers of the compute pass
    pub samplers: Vec<StaticSampler>,
}

fn finish_two_pass(t: super::Translation, record: super::MsInterface, ps: Option<&[u8]>) -> Result<TwoPass, String> {
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    let cs = dxc::compile_cached(&t.hlsl, "main", "cs_6_6")?.0;
    let ps_inputs = ps.map(|p| crate::gpu::dxbc::signatures(p).0);
    let ms = dxc::compile_cached(&super::emit::mesh_passthrough_for(&record, ps_inputs.as_deref()), "main", "ms_6_6")?.0;
    let mesh_groups_per_record = super::emit::passthrough_split(&record, super::emit::passthrough_elements(&record, ps_inputs.as_deref()));
    Ok(TwoPass { cs, ms, record, mesh_groups_per_record, samplers: t.static_samplers })
}

/// The mesh-shader (NGG) program of a console MS container; the pass-through carries what
/// the pixel shader `ps` (its container: console or PC) reads.
pub fn console_mesh(blob: &[u8], ps: Option<&[u8]>, split_wave64: bool) -> Result<TwoPass, String> {
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 7)).ok_or("no NGG hardware variant")?;
    let mode = interface(blob, &sh, f).ok_or("unknown MS interface (e.g. lines / points)")?;
    let Mode::Mesh(ms) = &mode else { return Err("the container is not an MS".into()) };
    let record = ms.clone();
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    finish_two_pass(super::translate_mesh(code, ms.clone(), super::detect_wave64(code), split_wave64, mem_windows())?, record, ps)
}

/// Whether the adapter runs only 32-lane waves (its wave range lacks 64): wave64 compute,
/// mesh and geometry programs then need the split form.
pub fn wave32_only(wave_min: u32, wave_max: u32) -> bool {
    !(wave_min <= 64 && wave_max >= 64)
}

/// `XCL_SPLIT_WAVE64=1`: the split wave64 form also where the GPU has 64-lane waves – to check
/// the path of GPUs with 32-lane waves only (NVIDIA) on an RDNA GPU, which runs both sizes.
pub fn split_wave64_forced() -> bool {
    std::env::var("XCL_SPLIT_WAVE64").is_ok_and(|v| v == "1")
}

/// Whether a wave64 VS / PS can run in 32-lane waves on the adapter: D3D12 has no `WaveSize`
/// for graphics shaders, so the driver picks, and one whose range includes 32 may pick 32 –
/// an RDNA2 driver (RX 6600 XT, issue #42) runs pixel shaders in waves of 32 although it
/// reports 32–64. Such a VS / PS needs the `upper_lanes` form (`Options::split_wave64`), which
/// checks the wave size at run time, so a driver that picks 64 lanes runs it as before.
pub fn graphics_wave32(wave_min: u32, wave_max: u32) -> bool {
    wave32_only(wave_min, wave_max) || wave_min <= 32
}

/// The device's wave lane range; `None` when it cannot be asked.
fn wave_range(device: &ID3D12Device) -> Option<(u32, u32)> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            &mut options as *mut _ as _,
            std::mem::size_of_val(&options) as u32,
        )
    }
    .ok()
    .map(|_| (options.WaveLaneCountMin, options.WaveLaneCountMax))
}

pub fn wave32_only_for_device(device: &ID3D12Device) -> bool {
    split_wave64_forced() || wave_range(device).is_none_or(|(min, max)| wave32_only(min, max))
}

/// `graphics_wave32` for the device: the split flag for a wave64 VS / PS. Unknown: assume so.
pub fn graphics_wave32_for_device(device: &ID3D12Device) -> bool {
    split_wave64_forced() || wave_range(device).is_none_or(|(min, max)| graphics_wave32(min, max))
}

/// A legacy GS pipeline: the ES variant of a console VS + the GS as one program.
pub fn console_geometry(vs: &[u8], gs: &[u8], ps: Option<&[u8]>, split_wave64: bool) -> Result<TwoPass, String> {
    let (code, g) = super::geometry_program(vs, gs)?;
    let record = g.record();
    let translation = super::translate_geometry(&code, g, split_wave64, mem_windows())?;
    finish_two_pass(translation, record, ps)
}

/// The geometry shader (gs_6_6 DXIL) that draws a console rect list's rectangles behind the
/// console VS `vs` (its container): `emit::rect_list_gs` for the VS's output signature, which
/// is what the translated VS's output struct is declared from.
pub fn rect_list_gs(vs: &[u8]) -> Result<Vec<u8>, String> {
    let outputs = crate::gpu::dxbc::signatures(vs).1;
    if outputs.is_empty() {
        return Err("the VS has no output signature".into());
    }
    Ok(dxc::compile_cached(&super::emit::rect_list_gs(&outputs), "main", "gs_6_6")?.0)
}

/// XCL shader ABI root signature (see the module docs); `graphics` allows an input layout.
pub fn abi_root_signature(dev: &ID3D12Device, graphics: bool) -> Result<ID3D12RootSignature, String> {
    abi_root_signature_with(dev, graphics, &[])
}

/// A static sampler (register `k`, space 1) from its S#; static samplers have only three
/// border colours (others: transparent black).
pub fn static_sampler_desc(k: u32, (w, comparison): &StaticSampler) -> D3D12_STATIC_SAMPLER_DESC {
    let d = crate::gpu::desc::sampler_d3d12(w, *comparison);
    let border = match d.BorderColor {
        [0.0, 0.0, 0.0, 1.0] => D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        [1.0, 1.0, 1.0, 1.0] => D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        _ => D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
    };
    D3D12_STATIC_SAMPLER_DESC {
        Filter: d.Filter,
        AddressU: d.AddressU,
        AddressV: d.AddressV,
        AddressW: d.AddressW,
        MipLODBias: d.MipLODBias,
        MaxAnisotropy: d.MaxAnisotropy,
        ComparisonFunc: d.ComparisonFunc,
        BorderColor: border,
        MinLOD: d.MinLOD,
        MaxLOD: d.MaxLOD,
        ShaderRegister: k,
        RegisterSpace: 1,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// The ABI root signature with the pipeline's static samplers (registers 0.., space 1).
pub fn abi_root_signature_with(dev: &ID3D12Device, graphics: bool, samplers: &[StaticSampler]) -> Result<ID3D12RootSignature, String> {
    let statics: Vec<D3D12_STATIC_SAMPLER_DESC> = samplers.iter().enumerate().map(|(k, s)| static_sampler_desc(k as u32, s)).collect();
    unsafe {
        // u0 io, u1 res, u2..u5 the memory windows (`gpu::uma`; unused ones need no binding)
        let params: Vec<D3D12_ROOT_PARAMETER1> = (0..6)
            .map(|i| D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: i, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            })
            .collect();
        let mut flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        if graphics {
            flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: statics.len() as u32,
                    pStaticSamplers: if statics.is_empty() { std::ptr::null() } else { statics.as_ptr() },
                    Flags: flags,
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(|e| format!("{e}"))?;
        let blob = blob.ok_or("root signature")?;
        dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("CreateRootSignature: {e}"))
    }
}

#[cfg(test)]
mod shader_cache_tests {
    use super::*;

    #[test]
    fn translated_identity_tracks_sources_and_dxc() {
        let built_source = env!("XCL_TRANSLATOR_SOURCE_ID");
        assert_eq!(built_source.len(), 16);
        assert!(u64::from_str_radix(built_source, 16).is_ok());
        let a = translated::identity_for("source-a", "dxc-a");
        assert_ne!(a, 0);
        assert_eq!(a, translated::identity_for("source-a", "dxc-a"));
        assert_ne!(a, translated::identity_for("source-b", "dxc-a"));
        assert_ne!(a, translated::identity_for("source-a", "dxc-b"));
    }

    #[test]
    fn translated_namespace_tracks_every_request_input() {
        let dir = PathBuf::from("cache");
        let base = translated::file(&dir, 42, b"console", Stage::Compute, 3, 2, false);
        let paths = [
            base.clone(),
            translated::file(&dir, 43, b"console", Stage::Compute, 3, 2, false),
            translated::file(&dir, 42, b"consolf", Stage::Compute, 3, 2, false),
            translated::file(&dir, 42, b"console", Stage::Vertex, 3, 2, false),
            translated::file(&dir, 42, b"console", Stage::Compute, 4, 2, false),
            translated::file(&dir, 42, b"console", Stage::Compute, 3, 3, false),
            translated::file(&dir, 42, b"console", Stage::Compute, 3, 2, true),
        ];
        assert_eq!(paths.iter().collect::<std::collections::HashSet<_>>().len(), paths.len());
        assert!(base.starts_with(dir.join("translated").join("v2").join("000000000000002a")));
    }

    /// A translation on disk comes back for the same bytes and semantic identity only: other
    /// bytes of the same length, another translator/toolchain or a cut file are misses.
    #[test]
    fn translations_on_disk_need_the_same_bytes_and_identity() {
        let dir = std::env::temp_dir().join(format!("xcl-translated-{}", std::process::id()));
        let path = dir.join("t.bin");
        let c = Compiled { dxil: vec![1, 2, 3, 4, 5], samplers: vec![([7, 8, 9, 10], true), ([1, 2, 3, 4], false)] };
        translated::store_in(&path, 42, b"console", &c);
        let back = translated::load_in(&path, 42, b"console").expect("stored");
        assert_eq!((back.dxil, back.samplers), (c.dxil.clone(), c.samplers.clone()));
        assert!(translated::load_in(&path, 43, b"console").is_none(), "another identity");
        assert!(translated::load_in(&path, 42, b"consolf").is_none(), "other bytes");
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 1]).unwrap();
        assert!(translated::load_in(&path, 42, b"console").is_none(), "a cut file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn translated_v2_ignores_valid_entries_outside_its_namespace() {
        let dir = std::env::temp_dir().join(format!("xcl-translated-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let blob = b"console";
        let ident = 42;
        let unnamespaced = dir.join("translated").join(format!("{:016x}_Compute_0_1_0.bin", crate::gpu::shader::fnv64(blob)));
        // Use a valid v2 payload so the miss proves the namespace, rather than header parsing,
        // kept the old root-level location out of consideration.
        let old = compiled(3);
        translated::store_in(&unnamespaced, ident, blob, &old);

        let mut calls = 0;
        let first = translated::cached_in(&dir, ident, blob, Stage::Compute, 0, 1, false, || {
            calls += 1;
            Ok(compiled(7))
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(first.dxil[0], 7);
        assert!(unnamespaced.exists());
        assert!(translated::file(&dir, ident, blob, Stage::Compute, 0, 1, false).exists());

        let second = translated::cached_in(&dir, ident, blob, Stage::Compute, 0, 1, false, || panic!("v2 entry was not reused")).unwrap();
        assert_eq!(second.dxil, first.dxil);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn key(hash: u64) -> ShaderKey {
        ShaderKey {
            hash,
            stage: Stage::Compute,
            sampler_base: 0,
            windows: 1,
            split_wave64: false,
            cache_dir: "cache".into(),
        }
    }

    fn compiled(marker: u8) -> Compiled {
        Compiled {
            dxil: vec![marker; 16],
            samplers: vec![([marker as u32; 4], false)],
        }
    }

    #[test]
    fn shader_cache_skips_repeated_translation_and_disk_lookups() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let mut calls = 0;
        let first = cached_shader(&cache, key(1), b"a", || {
            calls += 1;
            Ok(compiled(7))
        })
        .unwrap();
        for _ in 0..100 {
            let next = cached_shader(&cache, key(1), b"a", || {
                panic!("repeated shader reached the compiler path")
            })
            .unwrap();
            assert!(Arc::ptr_eq(&first, &next));
            assert_eq!(next.dxil, first.dxil);
            assert_eq!(next.samplers, first.samplers);
        }
        assert_eq!(calls, 1);
        let mut owned = first.as_ref().clone();
        owned.dxil[0] = 99;
        owned.samplers.clear();
        assert_eq!(first.dxil[0], 7);
        assert_eq!(first.samplers.len(), 1);
    }

    #[test]
    fn shader_cache_checks_context_and_blob_bytes() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let original = key(7);
        let a = cached_shader(&cache, original.clone(), b"a", || Ok(compiled(1))).unwrap();
        for changed in [
            ShaderKey {
                stage: Stage::Vertex,
                ..original.clone()
            },
            ShaderKey {
                sampler_base: 1,
                ..original.clone()
            },
            ShaderKey {
                windows: 2,
                ..original.clone()
            },
            ShaderKey {
                cache_dir: "other cache".into(),
                ..original.clone()
            },
        ] {
            let b = cached_shader(&cache, changed, b"a", || Ok(compiled(2))).unwrap();
            assert!(!Arc::ptr_eq(&a, &b));
            assert_eq!(b.dxil[0], 2);
        }
        let b = cached_shader(&cache, original.clone(), b"b", || Ok(compiled(3))).unwrap();
        assert_eq!(b.dxil[0], 3);
        let a_again =
            cached_shader(&cache, original, b"a", || panic!("original entry lost")).unwrap();
        assert!(Arc::ptr_eq(&a, &a_again));
    }

    #[test]
    fn shader_cache_retries_failed_and_deferred_compiles() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        assert!(cached_shader(&cache, key(1), b"a", || Err("compile failed".into())).is_err());
        let dir = std::env::temp_dir().join(format!("xcl-shader-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (result, missed) = dxc::cached_only(|| {
            cached_shader(&cache, key(1), b"a", || {
                let dxil = dxc::compile_cached_in(&dir, "probe", "main", "cs_6_6")?.0;
                Ok(Compiled {
                    dxil,
                    samplers: Vec::new(),
                })
            })
        });
        assert!(result.is_err() && missed);
        assert!(cache.lock().unwrap().entries.is_empty());
        let t = cached_shader(&cache, key(1), b"a", || Ok(compiled(9))).unwrap();
        assert_eq!(t.dxil[0], 9);
    }

    #[test]
    fn shader_cache_evicts_old_entries_and_skips_oversized_shaders() {
        let t = Arc::new(compiled(1));
        let limit = 2 * ShaderEntry {
            key: key(1),
            blob: b"a".to_vec(),
            shader: t.clone(),
        }
        .bytes();
        let mut cache = ShaderCache::new(limit);
        cache.insert(key(1), b"a", t.clone());
        cache.insert(key(2), b"b", t.clone());
        assert!(cache.get(&key(1), b"a").is_some());
        cache.insert(key(3), b"c", t.clone());
        assert!(cache.get(&key(2), b"b").is_none());
        assert!(cache.get(&key(1), b"a").is_some());
        assert!(cache.get(&key(3), b"c").is_some());
        cache.insert(
            key(4),
            b"d",
            Arc::new(Compiled {
                dxil: vec![0; limit + 1],
                samplers: Vec::new(),
            }),
        );
        assert!(cache.get(&key(4), b"d").is_none());
        assert_eq!(cache.bytes, limit);
        assert_eq!(cache.entries.len(), 2);
        let mut cache = ShaderCache::new(1 << 20);
        for i in 0..129 {
            cache.insert(key(i), b"a", t.clone());
        }
        assert_eq!(cache.entries.len(), 128);
        assert!(cache.get(&key(0), b"a").is_none());
    }

    #[test]
    fn shader_cache_bypasses_debug_overrides() {
        for name in ["XCL_RECOMP_NO_STRUCTURED", "XCL_RECOMP_WHY", "XCL_X64_ALL_MASKS", "XCL_WAVE64_SPIN"] {
            assert!(!cache_allowed(Stage::Compute, |v| v == name));
            assert!(!cache_allowed(Stage::Vertex, |v| v == name));
            assert!(!cache_allowed(Stage::Pixel, |v| v == name));
        }
        for name in [
            "XCL_DEBUG_MAGENTA",
            "XCL_DEBUG_LOD0",
            "XCL_DEBUG_PS_SUB",
            "XCL_DEBUG_PS_EXPR",
        ] {
            assert!(!cache_allowed(Stage::Pixel, |v| v == name));
            assert!(cache_allowed(Stage::Compute, |v| v == name));
        }
        assert!(cache_allowed(Stage::Pixel, |_| false));
    }

    #[test]
    fn shader_cache_probe_does_not_wait_for_an_in_flight_compile() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let dir = std::env::temp_dir().join(format!("xcl-shader-pending-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            let worker = scope.spawn(move || {
                cached_shader(cache, key(1), b"a", || {
                    started_tx.send(()).unwrap();
                    release_rx
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    Ok(compiled(1))
                })
            });
            started_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            let probe = scope.spawn(move || {
                let (result, missed) = dxc::cached_only(|| {
                    cached_shader(cache, key(1), b"a", || {
                        let dxil =
                            dxc::compile_cached_in(&dir, "pending probe", "main", "cs_6_6")?.0;
                        Ok(Compiled {
                            dxil,
                            samplers: Vec::new(),
                        })
                    })
                });
                probe_tx.send(result.is_err() && missed).unwrap();
            });
            let result = probe_rx.recv_timeout(std::time::Duration::from_secs(2));
            release_tx.send(()).unwrap();
            worker.join().unwrap().unwrap();
            probe.join().unwrap();
            assert!(result.unwrap());
        });
    }

    #[test]
    fn shader_cache_does_not_serialize_different_compiles() {
        let cache = Mutex::new(ShaderCache::new(1 << 20));
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let mut threads = Vec::new();
            for hash in [1, 2] {
                let cache = &cache;
                let barrier = &barrier;
                threads.push(scope.spawn(move || {
                    cached_shader(cache, key(hash), b"a", || {
                        barrier.wait();
                        Ok(compiled(1))
                    })
                }));
            }
            for thread in threads {
                assert_eq!(thread.join().unwrap().unwrap().dxil[0], 1);
            }
        });
    }
}

#[cfg(test)]
mod compile_time_bench {
    use super::*;

    #[test]
    #[ignore]
    fn repeated_pso_shader_compile() {
        use crate::gpu::xbox::tests::{fssm, sample, CODE};
        let mut code = Vec::new();
        for _ in 0..256 {
            code.extend_from_slice(&CODE[..8]);
        }
        code.extend_from_slice(&CODE[8..]);
        let mut f = fssm(&code, &[(6, 0)]);
        for offset in [0x13C, 0x140, 0x144] {
            f[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        }
        let blob = sample(5, &[f]);
        let first = console_to_dxil(&blob, Stage::Compute, 0).unwrap();
        let mut batches = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            for _ in 0..100 {
                let next = console_to_dxil(&blob, Stage::Compute, 0).unwrap();
                assert_eq!(next.dxil, first.dxil);
                assert_eq!(next.samplers, first.samplers);
            }
            batches.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("100 repeated PSO shader requests, warm DXIL cache, ms: {batches:?}");
    }
}
