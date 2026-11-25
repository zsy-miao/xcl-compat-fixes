//! Coverage measurement for the direct DXIL translator.
//!
//! * [`Report`]: per-stage outcome counts (translated / unsupported / validator-failed /
//!   panicked), a ranked histogram of unsupported constructs, translation times and DXIL
//!   sizes (direct vs the HLSL + DXC path). Used by `xcl shaders … --direct`.
//! * [`translate_mode`]: one console program through the direct translator for a
//!   `recomp::Mode`, with panics caught and classified.
//! * Live fallbacks (`XCL_DIRECT_DXIL`): [`record_fallback`] counts the HLSL fallbacks of
//!   `pso.rs` by (stage, reason); with `XCL_DIRECT_DXIL_STATS=1` [`dump_fallbacks`] prints
//!   the summary (called at process exit; harmless to call more than once).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use super::kernel::{self, KernelOptions};
use crate::gpu::recomp::Mode;

/// The stage name of a recompiler mode (as the corpus CLI prints it).
pub fn stage_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Kernel => "kernel",
        Mode::Compute(_) => "CS",
        Mode::Vertex(_) => "VS",
        Mode::Pixel(_) => "PS",
        Mode::Mesh(_) => "MS",
        Mode::Geometry(_) => "GS",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    Translated,
    Unsupported,
    ValidatorFailed,
    Panicked,
    /// the sweep gave up waiting (`xcl shaders --direct --timeout`)
    Timeout,
}

/// Classifies a translator error string.
pub fn classify(err: &str) -> Outcome {
    if err.contains("translator panicked") || err.starts_with("panic:") {
        Outcome::Panicked
    } else if err.starts_with("timeout:") {
        Outcome::Timeout
    } else if err.contains("failed validation") || err.starts_with("validator:") {
        Outcome::ValidatorFailed
    } else {
        Outcome::Unsupported
    }
}

/// The histogram key of a translator error: the instruction mnemonic plus the reason with
/// operands, offsets and numbers stripped, so equal constructs collapse into one bucket.
/// Errors look like `"v_foo v1, v2: unsupported …"` (`lower::unsupported`) or free text.
pub fn reason_key(err: &str) -> String {
    let first = err.lines().next().unwrap_or("").trim();
    let first = first.strip_prefix("the direct DXIL translator panicked: ").map(|m| m.trim()).unwrap_or(first);
    let (head, why) = match first.split_once(": ") {
        Some((h, w)) => (h.trim(), w.trim()),
        None => ("", first),
    };
    // an instruction head: mnemonic first (`s_`, `v_`, `ds_`, `buffer_`, …); drop its operands
    let mnemonic = head.split_whitespace().next().filter(|m| m.contains('_') && m.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    let norm = |s: &str| -> String {
        let mut out = String::new();
        let mut last_num = false;
        for tok in s.split_whitespace() {
            let t = tok.trim_matches(|c: char| ",;:()[]".contains(c));
            let numeric = !t.is_empty() && (t.starts_with("0x") || t.starts_with("+0x") || t.chars().all(|c| c.is_ascii_digit() || c == '-' || c == '+'));
            if numeric {
                if !last_num {
                    out.push_str(" N");
                }
                last_num = true;
                continue;
            }
            last_num = false;
            out.push(' ');
            out.push_str(tok);
        }
        let out = out.trim().to_string();
        if out.len() > 120 { format!("{}…", &out[..out.char_indices().take_while(|(i, _)| *i < 120).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(0)]) } else { out }
    };
    match mnemonic {
        Some(m) => format!("{m}: {}", norm(why)),
        None => norm(first),
    }
}

/// One program's result.
#[derive(Debug, Clone)]
pub struct Sample {
    pub stage: &'static str,
    /// `Ok(dxil size)` or the error
    pub result: Result<usize, String>,
    pub time: Duration,
    /// DXIL size of the HLSL + DXC path (`--compare`), when it compiled
    pub hlsl_size: Option<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct StageCounts {
    pub translated: usize,
    pub unsupported: usize,
    pub validator_failed: usize,
    pub panicked: usize,
    pub timeout: usize,
}

impl StageCounts {
    pub fn total(&self) -> usize {
        self.translated + self.unsupported + self.validator_failed + self.panicked + self.timeout
    }
}

#[derive(Debug, Default, Clone)]
pub struct Report {
    pub stages: BTreeMap<&'static str, StageCounts>,
    /// reason key → (programs, stages it was seen in, one example program)
    pub reasons: BTreeMap<String, (usize, BTreeMap<&'static str, usize>, String)>,
    pub times_us: Vec<u64>,
    /// (direct DXIL bytes, HLSL-path DXIL bytes) for programs both paths produced
    pub sizes: Vec<(usize, usize)>,
    /// the slowest programs: (microseconds, program), slowest first, at most [`SLOWEST`]
    pub slowest: Vec<(u64, String)>,
}

/// How many of the slowest programs a [`Report`] keeps.
pub const SLOWEST: usize = 10;

impl Report {
    pub fn add(&mut self, name: &str, s: &Sample) {
        let c = self.stages.entry(s.stage).or_default();
        let us = s.time.as_micros() as u64;
        self.times_us.push(us);
        if self.slowest.len() < SLOWEST || self.slowest.last().is_some_and(|l| l.0 < us) {
            let at = self.slowest.partition_point(|l| l.0 >= us);
            self.slowest.insert(at, (us, name.to_string()));
            self.slowest.truncate(SLOWEST);
        }
        match &s.result {
            Ok(size) => {
                c.translated += 1;
                if let Some(h) = s.hlsl_size {
                    self.sizes.push((*size, h));
                }
            }
            Err(e) => {
                match classify(e) {
                    Outcome::Panicked => c.panicked += 1,
                    Outcome::Timeout => c.timeout += 1,
                    Outcome::ValidatorFailed => c.validator_failed += 1,
                    _ => c.unsupported += 1,
                }
                let r = self.reasons.entry(reason_key(e)).or_insert_with(|| (0, BTreeMap::new(), name.to_string()));
                r.0 += 1;
                *r.1.entry(s.stage).or_default() += 1;
            }
        }
    }

    pub fn total(&self) -> StageCounts {
        let mut t = StageCounts::default();
        for c in self.stages.values() {
            t.translated += c.translated;
            t.unsupported += c.unsupported;
            t.validator_failed += c.validator_failed;
            t.panicked += c.panicked;
            t.timeout += c.timeout;
        }
        t
    }

    /// Reasons ranked by programs (ties by name).
    pub fn ranked(&self) -> Vec<(&String, usize)> {
        let mut v: Vec<_> = self.reasons.iter().map(|(k, v)| (k, v.0)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }

    /// (p50, p95, max) translation time in microseconds.
    pub fn time_percentiles(&self) -> (u64, u64, u64) {
        percentiles(&self.times_us)
    }

    pub fn text(&self, top: usize) -> String {
        let mut s = String::from("direct DXIL coverage\n  stage     total  translated  unsupported  validator  panicked  timeout\n");
        let row = |s: &mut String, name: &str, c: &StageCounts| {
            let pct = if c.total() > 0 { 100.0 * c.translated as f64 / c.total() as f64 } else { 0.0 };
            s.push_str(&format!("  {name:<7} {:>7} {:>8} ({pct:>5.1}%) {:>10} {:>10} {:>9} {:>8}\n", c.total(), c.translated, c.unsupported, c.validator_failed, c.panicked, c.timeout));
        };
        for (k, c) in &self.stages {
            row(&mut s, k, c);
        }
        row(&mut s, "all", &self.total());
        let (p50, p95, max) = self.time_percentiles();
        s.push_str(&format!("  translation time per program: p50 {:.2} ms, p95 {:.2} ms, max {:.2} ms\n", p50 as f64 / 1e3, p95 as f64 / 1e3, max as f64 / 1e3));
        for (us, name) in &self.slowest {
            s.push_str(&format!("    slow: {:>9.2} ms  {name}
", *us as f64 / 1e3));
        }
        if !self.sizes.is_empty() {
            let (d, h): (usize, usize) = self.sizes.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            s.push_str(&format!("  DXIL size, {} programs: direct {} B, HLSL+DXC {} B (direct/HLSL {:.2})\n", self.sizes.len(), d, h, d as f64 / h.max(1) as f64));
        }
        let ranked = self.ranked();
        if !ranked.is_empty() {
            s.push_str(&format!("  unsupported constructs ({} kinds, top {}):\n", ranked.len(), top.min(ranked.len())));
            for (k, n) in ranked.iter().take(top) {
                let r = &self.reasons[*k];
                let stages: Vec<String> = r.1.iter().map(|(s, n)| format!("{s} {n}")).collect();
                s.push_str(&format!("  {n:>6}  {k}  [{}]  e.g. {}\n", stages.join(", "), r.2));
            }
        }
        s
    }

    pub fn json(&self, top: usize) -> String {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        let counts = |c: &StageCounts| format!("{{\"total\":{},\"translated\":{},\"unsupported\":{},\"validator_failed\":{},\"panicked\":{},\"timeout\":{}}}", c.total(), c.translated, c.unsupported, c.validator_failed, c.panicked, c.timeout);
        let stages: Vec<String> = self.stages.iter().map(|(k, c)| format!("\"{k}\":{}", counts(c))).collect();
        let reasons: Vec<String> = self.ranked().iter().take(top).map(|(k, n)| format!("{{\"reason\":\"{}\",\"programs\":{n},\"example\":\"{}\"}}", esc(k), esc(&self.reasons[*k].2))).collect();
        let (p50, p95, max) = self.time_percentiles();
        let (d, h): (usize, usize) = self.sizes.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        format!(
            "{{\"stages\":{{{}}},\"all\":{},\"time_us\":{{\"p50\":{p50},\"p95\":{p95},\"max\":{max}}},\"size\":{{\"programs\":{},\"direct\":{d},\"hlsl\":{h}}},\"unsupported\":[{}]}}",
            stages.join(","),
            counts(&self.total()),
            self.sizes.len(),
            reasons.join(",")
        )
    }
}

pub fn percentiles(v: &[u64]) -> (u64, u64, u64) {
    if v.is_empty() {
        return (0, 0, 0);
    }
    let mut v = v.to_vec();
    v.sort_unstable();
    let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    (at(0.5), at(0.95), *v.last().unwrap())
}

/// One program through the direct translator for `mode`. Panics (outside the translator's
/// own thread) come back as `"panic: …"`; `validate` runs DXC's validator on the result
/// (the translator itself only validates in debug / test builds).
pub fn translate_mode(code: &[u8], mode: &Mode, o: KernelOptions, validate: bool) -> Result<Vec<u8>, String> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match mode {
        Mode::Kernel => kernel::translate_full(code, o),
        Mode::Compute(cs) => kernel::translate_compute(code, o, cs),
        Mode::Vertex(vs) => kernel::translate_vertex(code, KernelOptions { split: false, ..o }, vs),
        Mode::Pixel(ps) => kernel::translate_pixel(code, KernelOptions { split: false, ..o }, ps),
        Mode::Mesh(ms) => kernel::translate_mesh(code, o, ms),
        Mode::Geometry(g) => kernel::translate_geometry(code, o, g),
    }));
    let k = match r {
        Ok(r) => r?,
        Err(e) => {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            return Err(format!("panic: {msg}"));
        }
    };
    if validate {
        super::validate::validate(&k.dxil).map_err(|e| format!("validator: {}", e.lines().next().unwrap_or("")))?;
    }
    Ok(k.dxil)
}

/// Deterministic sample of `n` of `0..len` (splitmix64-driven partial Fisher–Yates), in
/// ascending order. `n >= len` returns everything.
pub fn sample_indices(len: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..len).collect();
    if n >= len {
        return idx;
    }
    let mut st = seed;
    let mut next = || {
        st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in 0..n {
        let j = i + (next() % (len - i) as u64) as usize;
        idx.swap(i, j);
    }
    idx.truncate(n);
    idx.sort_unstable();
    idx
}

// ---- live fallbacks ----------------------------------------------------------------------

static FALLBACKS: Mutex<BTreeMap<(String, String), usize>> = Mutex::new(BTreeMap::new());
static TRANSLATED: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());

/// `XCL_DIRECT_DXIL_STATS=1`: keep and print the live fallback summary.
pub fn stats_enabled() -> bool {
    std::env::var("XCL_DIRECT_DXIL_STATS").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// A live shader the direct path translated (`stage`: "CS", "VS", "PS", "MS").
pub fn record_translated(stage: &str) {
    if let Ok(mut m) = TRANSLATED.lock() {
        *m.entry(stage.to_string()).or_default() += 1;
    }
}

/// A live shader that fell back to the HLSL path. Returns the reason key (for the log line).
pub fn record_fallback(stage: &str, err: &str) -> String {
    let key = reason_key(err);
    if let Ok(mut m) = FALLBACKS.lock() {
        *m.entry((stage.to_string(), key.clone())).or_default() += 1;
    }
    key
}

/// The live summary: per stage direct / fallback counts and the fallback reasons ranked.
pub fn fallback_summary() -> String {
    let fb = FALLBACKS.lock().map(|m| m.clone()).unwrap_or_default();
    let ok = TRANSLATED.lock().map(|m| m.clone()).unwrap_or_default();
    let mut per_stage: BTreeMap<String, (usize, usize)> = ok.iter().map(|(k, v)| (k.clone(), (*v, 0))).collect();
    for ((st, _), n) in &fb {
        per_stage.entry(st.clone()).or_default().1 += n;
    }
    let mut s = String::from("[SHADER] direct DXIL summary:");
    for (st, (d, f)) in &per_stage {
        s.push_str(&format!(" {st} direct {d} / HLSL fallback {f};"));
    }
    let mut v: Vec<_> = fb.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for ((st, why), n) in v {
        s.push_str(&format!("\n  {n:>5}  {st}  {why}"));
    }
    s
}

/// Prints [`fallback_summary`] to stderr and the log when `XCL_DIRECT_DXIL_STATS` is set.
pub fn dump_fallbacks() {
    if stats_enabled() {
        let s = fallback_summary();
        eprintln!("{s}");
        crate::xlog_info!("{}", s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2;

    #[test]
    fn reason_keys_collapse_operands() {
        let a = reason_key("v_foo_f32 v1, v2, 0x10: unsupported modifier 3");
        let b = reason_key("v_foo_f32 v7, s3, 0x20: unsupported modifier 4");
        assert_eq!(a, b);
        assert_eq!(a, "v_foo_f32: unsupported modifier N");
        assert_eq!(reason_key("geometry mode (ES+GS): not supported"), "geometry mode (ES+GS): not supported");
        assert_eq!(classify("the direct DXIL translator panicked: boom"), Outcome::Panicked);
        assert_eq!(classify("direct DXIL failed validation: x"), Outcome::ValidatorFailed);
        assert_eq!(classify("v_x: y"), Outcome::Unsupported);
        assert_eq!(classify("timeout: over 10 s"), Outcome::Timeout);
    }

    #[test]
    fn sampling_is_seeded_and_bounded() {
        let a = sample_indices(1000, 50, 7);
        assert_eq!(a.len(), 50);
        assert_eq!(a, sample_indices(1000, 50, 7));
        assert_ne!(a, sample_indices(1000, 50, 8));
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(sample_indices(5, 50, 1), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn report_counts_and_ranks() {
        let mut r = Report::default();
        let t = Duration::from_micros;
        r.add("a", &Sample { stage: "CS", result: Ok(100), time: t(10), hlsl_size: Some(200) });
        r.add("b", &Sample { stage: "CS", result: Err("v_bar v1: no".into()), time: t(20), hlsl_size: None });
        r.add("c", &Sample { stage: "PS", result: Err("v_bar v3: no".into()), time: t(30), hlsl_size: None });
        r.add("d", &Sample { stage: "PS", result: Err("s_baz: no".into()), time: t(40), hlsl_size: None });
        r.add("e", &Sample { stage: "VS", result: Err("direct DXIL failed validation: bad".into()), time: t(50), hlsl_size: None });
        r.add("f", &Sample { stage: "GS", result: Err("panic: oops".into()), time: t(1000), hlsl_size: None });
        let all = r.total();
        assert_eq!((all.translated, all.unsupported, all.validator_failed, all.panicked), (1, 3, 1, 1));
        assert_eq!(r.stages["PS"].unsupported, 2);
        assert_eq!(r.ranked()[0], (&"v_bar: no".to_string(), 2));
        assert_eq!(r.time_percentiles(), (40, 1000, 1000));
        let txt = r.text(10);
        assert!(txt.contains("v_bar: no") && txt.contains("direct/HLSL 0.50"), "{txt}");
        let j = r.json(10);
        assert!(j.contains("\"CS\":{\"total\":2,\"translated\":1") && j.contains("\"reason\":\"v_bar: no\",\"programs\":2"), "{j}");
    }

    /// Synthetic programs from the RDNA2 encoder through the real translator.
    #[test]
    fn synthetic_programs_through_translate_mode() {
        let o = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let ok = rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).expect("encode");
        let mut r = Report::default();
        for (name, code, mode) in [("ok", ok.clone(), Mode::Kernel), ("gs", ok, Mode::Geometry(Default::default()))] {
            let t0 = std::time::Instant::now();
            let res = translate_mode(&code, &mode, o, false).map(|d| d.len());
            r.add(name, &Sample { stage: stage_name(&mode), result: res, time: t0.elapsed(), hlsl_size: None });
        }
        assert_eq!(r.stages["kernel"].translated, 1, "{}", r.text(5));
        assert_eq!(r.stages["GS"].translated, 1, "{}", r.text(5));
    }

    #[test]
    fn live_fallback_summary() {
        record_translated("CS");
        let k = record_fallback("PS", "v_qq v1: unsupported");
        assert_eq!(k, "v_qq: unsupported");
        let s = fallback_summary();
        assert!(s.contains("PS  v_qq: unsupported"), "{s}");
    }
}
