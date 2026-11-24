//! `xcl run <game folder | exe> [-- args]` – prepares data folders and config, then
//! starts the host process with the game's own command line (argv[0] = game exe).

use std::path::{Path, PathBuf};

use crate::config::{Boot, Config};
use crate::util::wide;

/// Data root: `%XCL_DATA_ROOT%` or `%LOCALAPPDATA%\XCL`.
pub fn data_root() -> PathBuf {
    if let Ok(p) = std::env::var("XCL_DATA_ROOT") {
        return PathBuf::from(p);
    }
    let base = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    PathBuf::from(base).join("XCL")
}

/// Values of `attr` on every `<tag …>` element (so "<Executable" != "<ExecutableList").
pub(crate) fn xml_attrs(xml: &str, tag: &str, attr: &str) -> Vec<String> {
    let open = format!("<{tag}");
    let key = format!("{attr}=\"");
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = xml[from..].find(&open).map(|i| from + i) {
        from = i + open.len();
        let Some(next) = xml[from..].chars().next() else { break };
        if !(next.is_whitespace() || next == '/' || next == '>') {
            continue;
        }
        let Some(end) = xml[from..].find('>') else { break };
        let t = &xml[from..from + end];
        // attribute must start at a whitespace boundary ("Name=" but not "DisplayName=")
        let mut pos = 0;
        while let Some(k) = t[pos..].find(&key).map(|k| pos + k) {
            if k == 0 || t[..k].ends_with(char::is_whitespace) {
                let a = k + key.len();
                if let Some(b) = t[a..].find('"') {
                    out.push(xml_unescape(&t[a..a + b]));
                }
                break;
            }
            pos = k + key.len();
        }
    }
    out
}

/// An attribute value with XML's character references replaced: the five named ones
/// ("Ratchet &amp; Clank") and numeric ones (`&#39;`, `&#x27;`). Anything else stays as written.
fn xml_unescape(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    let mut rest = v;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let decoded = rest.find(';').and_then(|end| {
            let c = match &rest[1..end] {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                "apos" => '\'',
                r => {
                    let n = r.strip_prefix("#x").map(|h| u32::from_str_radix(h, 16)).or_else(|| r.strip_prefix('#').map(|d| d.parse()))?;
                    char::from_u32(n.ok()?)?
                }
            };
            Some((c, end + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn xml_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    xml_attrs(xml, tag, attr).into_iter().next()
}

/// MicrosoftGame.config as text, without its comments: a commented-out `<Executable>` or
/// `<Identity>` is not the game's.
pub(crate) fn decode_game_config(raw: &[u8]) -> Option<String> {
    decode_text(raw).map(|xml| without_comments(&xml))
}

fn without_comments(xml: &str) -> String {
    let mut out = String::with_capacity(xml.len());
    let mut rest = xml;
    while let Some(i) = rest.find("<!--") {
        out.push_str(&rest[..i]);
        match rest[i + 4..].find("-->") {
            Some(j) => rest = &rest[i + 4 + j + 3..],
            None => return out, // unterminated: the rest is comment
        }
    }
    out.push_str(rest);
    out
}

fn decode_text(raw: &[u8]) -> Option<String> {
    let (body, little_endian) = if let Some(body) = raw.strip_prefix(&[0xFF, 0xFE]) {
        (body, true)
    } else if let Some(body) = raw.strip_prefix(&[0xFE, 0xFF]) {
        (body, false)
    } else {
        return String::from_utf8(raw.to_vec()).ok();
    };
    if body.len() % 2 != 0 {
        return None;
    }
    let units: Vec<u16> = body
        .chunks_exact(2)
        .map(|b| if little_endian { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
        .collect();
    String::from_utf16(&units).ok()
}

/// (executables, title) from MicrosoftGame.config if present.
/// The game's name for people (`ShellVisuals DefaultDisplayName` of MicrosoftGame.config):
/// None when it is a resource reference (`ms-resource:…`) or missing.
pub fn display_name(root: &Path) -> Option<String> {
    let xml = std::fs::read(root.join("MicrosoftGame.config")).ok().and_then(|raw| decode_game_config(&raw))?;
    xml_attr(&xml, "ShellVisuals", "DefaultDisplayName").filter(|n| !n.trim().is_empty() && !n.starts_with("ms-resource:"))
}

pub fn read_game_config(root: &Path) -> (Vec<String>, Option<String>) {
    let p = root.join("MicrosoftGame.config");
    let Some(xml) = std::fs::read(p).ok().and_then(|raw| decode_game_config(&raw)) else { return (Vec::new(), None) };
    (xml_attrs(&xml, "Executable", "Name"), xml_attr(&xml, "Identity", "Name"))
}

/// `xcl.exe` itself (by name, path or identical bytes) must never be picked as the game.
fn is_xcl_copy(p: &Path) -> bool {
    if p.file_name().map(|n| n.to_string_lossy().eq_ignore_ascii_case("xcl.exe")).unwrap_or(false) {
        return true;
    }
    let Ok(me) = std::env::current_exe() else { return false };
    if std::fs::canonicalize(p).ok() == std::fs::canonicalize(&me).ok() {
        return true;
    }
    match (std::fs::metadata(p), std::fs::metadata(&me)) {
        (Ok(a), Ok(b)) if a.len() == b.len() => std::fs::read(p).ok() == std::fs::read(&me).ok(),
        _ => false,
    }
}

fn is_exe(p: &Path) -> bool {
    p.is_file() && p.extension().map(|x| x.eq_ignore_ascii_case("exe")).unwrap_or(false)
}

/// Executables in `dir` and its subfolders (up to `depth` levels), excluding XCL itself.
fn find_exes(dir: &Path, depth: u32, skipped_self: &mut bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() && depth > 0 {
            out.extend(find_exes(&p, depth - 1, skipped_self));
        } else if is_exe(&p) {
            if is_xcl_copy(&p) {
                *skipped_self = true;
            } else {
                out.push(p);
            }
        }
    }
    out
}

fn pick_exe(root: &Path) -> Result<PathBuf, String> {
    let (cfg_exes, _) = read_game_config(root);
    if !cfg_exes.is_empty() {
        for e in &cfg_exes {
            let p = root.join(e);
            if is_exe(&p) && !is_xcl_copy(&p) {
                return Ok(p);
            }
        }
        return Err(format!(
            "MicrosoftGame.config names {}, but there is no such file in {}",
            cfg_exes.join(", "),
            root.display()
        ));
    }
    let mut skipped_self = false;
    let rel = |p: &PathBuf| p.strip_prefix(root).unwrap_or(p).display().to_string();
    let top = find_exes(root, 0, &mut skipped_self);
    let candidates = if top.is_empty() { find_exes(root, 2, &mut skipped_self) } else { top };
    match candidates.len() {
        1 => Ok(candidates.into_iter().next().unwrap()),
        0 => Err(format!(
            "no game .exe file in {}{}",
            root.display(),
            if skipped_self { " (only xcl.exe – that is XCL itself; do not copy it into the game folder)" } else { "" }
        )),
        _ => Err(format!(
            "several .exe files ({}) – point to the game exe directly or set 'exe' in xcl.toml",
            candidates.iter().map(rel).collect::<Vec<_>>().join(", ")
        )),
    }
}

fn sanitize(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    if t.is_empty() {
        "title".into()
    } else {
        t
    }
}

/// The package folder of `exe`: the nearest folder above it (up to 4 levels) whose
/// MicrosoftGame.config names it – Unreal titles keep theirs in `<Game>/Binaries/XSX`, and the
/// package root is what the title's paths (and its D: drive) start from.
fn config_root(exe: &Path) -> Option<PathBuf> {
    let mut dir = exe.parent()?;
    for _ in 0..5 {
        let (exes, _) = read_game_config(dir);
        if exes.iter().any(|e| std::fs::canonicalize(dir.join(e)).ok().is_some_and(|p| PathBuf::from(p.to_string_lossy().trim_start_matches("\\\\?\\")) == exe)) {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
    None
}

fn resolve_location(target: &Path) -> Result<(PathBuf, Option<PathBuf>, String), String> {
    let target = std::fs::canonicalize(target).map_err(|e| format!("{}: {e}", target.display()))?;
    let target = PathBuf::from(target.to_string_lossy().trim_start_matches("\\\\?\\"));
    let (root, exe) = if target.is_dir() {
        (target, None)
    } else {
        if is_xcl_copy(&target) {
            return Err(format!("{} is XCL itself, not a game – give the game exe or folder", target.display()));
        }
        (config_root(&target).unwrap_or(target.parent().ok_or("no parent folder")?.to_path_buf()), Some(target))
    };
    let (_, identity) = read_game_config(&root);
    let title = sanitize(&identity.unwrap_or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
    Ok((root, exe, title))
}

pub fn resolve_target(target: &Path) -> Result<(PathBuf, PathBuf, String), String> {
    let (root, explicit_exe, title) = resolve_location(target)?;
    let exe = match explicit_exe {
        Some(exe) => exe,
        None => pick_exe(&root)?,
    };
    Ok((root, exe, title))
}

fn quote(a: &str) -> String {
    if !a.is_empty() && !a.contains([' ', '\t', '"']) {
        return a.to_string();
    }
    let mut s = String::from("\"");
    let mut bs = 0;
    for c in a.chars() {
        match c {
            '\\' => bs += 1,
            '"' => {
                s.push_str(&"\\".repeat(bs * 2 + 1));
                s.push('"');
                bs = 0;
            }
            _ => {
                s.push_str(&"\\".repeat(bs));
                bs = 0;
                s.push(c);
            }
        }
    }
    s.push_str(&"\\".repeat(bs * 2));
    s.push('"');
    s
}

pub fn run(target: &Path, args: &[String]) -> i32 {
    let (root, explicit_exe, title) = match resolve_location(target) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("xcl: {e}");
            return 2;
        }
    };
    let data_dir = data_root().join(&title);
    let mut config = match Config::load_or_create(&data_dir.join("xcl.toml")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("xcl: error in xcl.toml: {e}");
            return 2;
        }
    };
    // the exe given on the command line, else the one `xcl.toml` names (written on every run,
    // relative to the package root), else the folder's own pick
    let configured_exe = root.join(&config.exe);
    let exe = if let Some(exe) = explicit_exe {
        exe
    } else if !config.exe.is_empty() && is_exe(&configured_exe) && !is_xcl_copy(&configured_exe) {
        configured_exe
    } else {
        match pick_exe(&root) {
            Ok(exe) => exe,
            Err(error) => {
                eprintln!("xcl: {error}");
                return 2;
            }
        }
    };
    for d in ["logs", "cache", "overlay", "saves", "pls", "temp"] {
        let _ = std::fs::create_dir_all(data_dir.join(d));
    }
    config.exe = exe.strip_prefix(&root).ok().or(exe.file_name().map(Path::new)).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    let manifest = crate::manifest::load(&root);
    let from_manifest = config.resolve(manifest.as_ref());
    crate::tls::cleanup_cache(&data_dir.join("cache"));
    let boot = Boot { title: title.clone(), exe_path: exe.clone(), game_root: root.clone(), data_dir: data_dir.clone(), config };
    let boot_path = data_dir.join("cache").join("boot.json");
    if let Err(e) = std::fs::write(&boot_path, serde_json::to_string_pretty(&boot).unwrap()) {
        eprintln!("xcl: writing {}: {e}", boot_path.display());
        return 2;
    }
    // the previous run's log and report stay as *.prev.*: a failure that a retry does not
    // repeat is still there to read
    let logs = boot.logs_dir();
    let _ = std::fs::rename(logs.join("xcl.log"), logs.join("xcl.prev.log"));
    let _ = std::fs::rename(boot.report_path(), logs.join("report.prev.json"));
    let _ = std::fs::remove_file(boot.report_path());

    println!("XCL {} – starting '{title}'", env!("CARGO_PKG_VERSION"));
    println!("  exe:   {}", exe.display());
    println!("  data:  {}", data_dir.display());
    if !from_manifest.is_empty() {
        println!("  MicrosoftGame.Config: {}", from_manifest.join(", "));
    }
    match crate::inspect::console_imports(&exe) {
        Some(c) if c.is_empty() => println!(
            "  WARNING: this file imports no console library (kernelx / xgameruntime / d3d12_xs) –\n           it is most likely a regular Windows program, not an Xbox Series game."
        ),
        Some(c) => println!("  console: imports {}", c.join(", ")),
        None => println!("  WARNING: this is not a valid x64 PE file (encrypted? from an XVC package?)"),
    }
    let environment = crate::preflight::run(&boot.config.d3d12_adapter);
    crate::preflight::print(&environment);

    // the console's CPU is a Zen 2: its titles are compiled for it and fault with an illegal
    // instruction (0xC000001D) on a PC without its extensions – said here instead
    let missing = missing_console_cpu_features();
    if !missing.is_empty() {
        if std::env::var_os("XCL_SKIP_CPU_CHECK").is_some() {
            println!("  WARNING: this CPU lacks {} (XCL_SKIP_CPU_CHECK set, continuing)", missing.join(", "));
        } else {
            eprintln!("xcl: {}", format_missing_cpu_message(&missing));
            return 2;
        }
    }

    let mut cmdline = quote(&exe.to_string_lossy());
    for a in args {
        cmdline.push(' ');
        cmdline.push_str(&quote(a));
    }
    let code = match spawn_host(&cmdline, &root, &boot_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("xcl: failed to start the game process: {e}");
            return 2;
        }
    };
    summarize(&boot, code, &environment);
    code as i32
}

/// Formats a friendly error message listing the missing console Zen 2 CPU features.
pub fn format_missing_cpu_message(missing: &[&str]) -> String {
    format!(
        "this CPU lacks {}: Xbox Series X|S games are compiled for the console's Zen 2 CPU\n     and need them (they would stop with an illegal instruction). XCL_SKIP_CPU_CHECK=1 runs anyway.",
        missing.join(", ")
    )
}

/// Helper to evaluate CPU features using a query closure, returning missing feature names.
pub fn check_cpu_features<F: Fn(&str) -> bool>(has: F) -> Vec<&'static str> {
    const REQUIRED: &[(&str, &str)] = &[
        ("sse4.2", "SSE4.2"),
        ("popcnt", "POPCNT"),
        ("avx", "AVX"),
        ("avx2", "AVX2"),
        ("fma", "FMA"),
        ("f16c", "F16C"),
        ("bmi1", "BMI1"),
        ("bmi2", "BMI2"),
        ("lzcnt", "LZCNT"),
        ("movbe", "MOVBE"),
    ];
    let mut missing = Vec::new();
    for &(feature, name) in REQUIRED {
        if !has(feature) {
            missing.push(name);
        }
    }
    missing
}

/// Instruction set extensions of the console's Zen 2 that this CPU does not have (titles are
/// compiled for them: AVX2, FMA, F16C, BMI1 / BMI2, LZCNT, POPCNT, MOVBE, SSE4.2).
pub fn missing_console_cpu_features() -> Vec<&'static str> {
    check_cpu_features(|f| match f {
        "sse4.2" => std::arch::is_x86_feature_detected!("sse4.2"),
        "popcnt" => std::arch::is_x86_feature_detected!("popcnt"),
        "avx" => std::arch::is_x86_feature_detected!("avx"),
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "fma" => std::arch::is_x86_feature_detected!("fma"),
        "f16c" => std::arch::is_x86_feature_detected!("f16c"),
        "bmi1" => std::arch::is_x86_feature_detected!("bmi1"),
        "bmi2" => std::arch::is_x86_feature_detected!("bmi2"),
        "lzcnt" => std::arch::is_x86_feature_detected!("lzcnt"),
        "movbe" => std::arch::is_x86_feature_detected!("movbe"),
        _ => false,
    })
}

fn spawn_host(cmdline: &str, cwd: &Path, boot_path: &Path) -> Result<u32, String> {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError};
    use windows_sys::Win32::System::Threading::{
        CreateProcessW, GetExitCodeProcess, WaitForSingleObject, INFINITE, PROCESS_INFORMATION, STARTUPINFOW,
    };
    let me = std::env::current_exe().map_err(|e| e.to_string())?;
    std::env::set_var("XCL_HOST", boot_path);
    let app = wide(&me.to_string_lossy());
    let mut cmd = wide(cmdline);
    let dir = wide(&cwd.to_string_lossy());
    unsafe {
        let mut si: STARTUPINFOW = std::mem::zeroed();
        si.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
        let mut pi: PROCESS_INFORMATION = std::mem::zeroed();
        let ok = CreateProcessW(
            app.as_ptr(),
            cmd.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            0,
            std::ptr::null(),
            dir.as_ptr(),
            &si,
            &mut pi,
        );
        std::env::remove_var("XCL_HOST");
        if ok == 0 {
            return Err(format!("CreateProcessW: error {}", GetLastError()));
        }
        WaitForSingleObject(pi.hProcess, INFINITE);
        let mut code = 0u32;
        GetExitCodeProcess(pi.hProcess, &mut code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        Ok(code)
    }
}

fn summarize(boot: &Boot, code: u32, environment: &[crate::preflight::Check]) {
    let path = boot.report_path();
    let mut report = crate::report::load(&path);
    if let Some(r) = report.as_mut() {
        r.exit_code = Some(code as i64);
        r.environment = environment.to_vec();
        if r.dxc.dir.is_empty() {
            // the game compiled nothing: the DXC the checks above found for it
            r.dxc = crate::gpu::recomp::dxc::toolchain();
        }
        let _ = std::fs::write(&path, serde_json::to_string_pretty(r).unwrap_or_default());
    }
    println!();
    println!("Game exited with code {code:#x} ({code})");
    // repeated here: what the game printed in between has scrolled them away
    let missing: Vec<_> = environment.iter().filter(|c| matches!(c.level, crate::preflight::Level::Warning | crate::preflight::Level::Error)).cloned().collect();
    if !missing.is_empty() {
        println!("Missing on this PC (see the start of the run):");
        crate::preflight::print(&missing);
    }
    let Some(r) = report else {
        println!("No report – see the log: {}", boot.logs_dir().join("xcl.log").display());
        return;
    };
    if let Some(c) = &r.crash {
        println!("CRASH: {c}");
    }
    let total: usize = r.imports_by_source.values().sum();
    println!("Imports: {total} ({})", r.imports_by_source.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", "));
    if !r.called_stubs.is_empty() {
        println!("Unimplemented functions called ({}):", r.called_stubs.len());
        for s in r.called_stubs.iter().take(15) {
            println!("  {:>8}×  {}!{}   (first call: {})", s.calls, s.dll, s.function, s.first_caller);
        }
    }
    if !r.vfs.unmapped_paths.is_empty() {
        println!("Unmapped console paths: {}", r.vfs.unmapped_paths.len());
        for p in r.vfs.unmapped_paths.iter().take(5) {
            println!("  {p}");
        }
    }
    if !r.vfs.redirected_writes.is_empty() {
        println!("Writes to game files redirected to the overlay: {}", r.vfs.redirected_writes.len());
    }
    println!("Report: {}", path.display());
    println!("Log:    {}", boot.logs_dir().join("xcl.log").display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn microsoft_game_config_parsing() {
        let xml = r#"<?xml version="1.0"?><Game configVersion="1"><Identity Name="Contoso.Game" Publisher="CN=X" Version="1.0.0.0"/>
            <ExecutableList><Executable Name="game_scarlett.exe" TargetDeviceFamily="Scarlett" Id="Game"/></ExecutableList></Game>"#;
        assert_eq!(xml_attr(xml, "Identity", "Name").as_deref(), Some("Contoso.Game"));
        assert_eq!(xml_attr(xml, "Executable", "Name").as_deref(), Some("game_scarlett.exe"));
    }

    #[test]
    fn several_executables_in_config() {
        let xml = r#"<ExecutableList><Executable Name="Bin\game.exe" Id="Game"/><Executable Name="tool.exe" Id="T"/></ExecutableList>
            <ShellVisuals DefaultDisplayName="X" Name="no"/>"#;
        assert_eq!(xml_attrs(xml, "Executable", "Name"), vec!["Bin\\game.exe".to_string(), "tool.exe".to_string()]);
        assert_eq!(xml_attr(xml, "ShellVisuals", "Name").as_deref(), Some("no"), "DisplayName= must not match Name=");
    }

    #[test]
    fn attribute_values_have_their_character_references_replaced() {
        assert_eq!(xml_unescape("Tom &amp; Jerry&apos;s &lt;&quot;x&quot;&gt;"), "Tom & Jerry's <\"x\">");
        assert_eq!(xml_unescape("It&#39;s &#x2122;"), "It's \u{2122}");
        // not a reference: kept as written
        assert_eq!(xml_unescape("A & B &nbsp; &#xZZ; &"), "A & B &nbsp; &#xZZ; &");
        let xml = r#"<ExecutableList><Executable Name="Q&amp;A.exe" Id="Game"/></ExecutableList>"#;
        assert_eq!(xml_attrs(xml, "Executable", "Name"), ["Q&A.exe"]);
    }

    #[test]
    fn display_name_is_the_shell_name_unless_it_is_a_resource() {
        let dir = TempDir::new("display_name");
        let config = |name: &str| std::fs::write(dir.0.join("MicrosoftGame.config"), format!(r#"<Game><Identity Name="Contoso.Test"/><ShellVisuals DefaultDisplayName="{name}" Square44x44Logo="a.png"/></Game>"#)).unwrap();
        config("Minecraft Preview");
        assert_eq!(display_name(&dir.0).as_deref(), Some("Minecraft Preview"));
        config("Ratchet &amp; Clank: Rift Apart");
        assert_eq!(display_name(&dir.0).as_deref(), Some("Ratchet & Clank: Rift Apart"));
        config("ms-resource:AppName");
        assert_eq!(display_name(&dir.0), None);
        std::fs::remove_file(dir.0.join("MicrosoftGame.config")).unwrap();
        assert_eq!(display_name(&dir.0), None);
    }

    #[test]
    fn commented_out_entries_are_not_the_games() {
        let dir = TempDir::new("config_comments");
        std::fs::write(dir.0.join("old.exe"), b"MZ").unwrap();
        std::fs::write(dir.0.join("game.exe"), b"MZ").unwrap();
        let xml = r#"<Game>
  <!-- <Identity Name="Contoso.Old"/> -->
  <Identity Name="Contoso.Test"/>
  <ExecutableList>
    <!--<Executable Name="old.exe" Id="Game"/>-->
    <Executable Name="game.exe" Id="Game"/>
  </ExecutableList>
  <ShellVisuals DefaultDisplayName="Contoso Game"/>
  <!-- <ShellVisuals DefaultDisplayName="Old Name"/>
</Game>"#;
        std::fs::write(dir.0.join("MicrosoftGame.config"), xml).unwrap();
        assert_eq!(read_game_config(&dir.0), (vec!["game.exe".into()], Some("Contoso.Test".into())));
        assert_eq!(pick_exe(&dir.0).unwrap(), dir.0.join("game.exe"));
        assert_eq!(display_name(&dir.0).as_deref(), Some("Contoso Game"));
        assert_eq!(without_comments("a<!--b-->c<!--d-->e<!--f"), "ace");
    }

    #[test]
    fn game_config_accepts_bom_marked_utf16_and_rejects_malformed_text() {
        let dir = TempDir::new("utf16_config");
        let xml = r#"<Game><Identity Name="Contoso.Test"/><Executable Name="Bin\game.exe"/></Game>"#;
        for (bom, bytes) in [
            ([0xFF, 0xFE], xml.encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>()),
            ([0xFE, 0xFF], xml.encode_utf16().flat_map(u16::to_be_bytes).collect::<Vec<_>>()),
        ] {
            let mut raw = bom.to_vec();
            raw.extend(bytes);
            let decoded = decode_game_config(&raw).unwrap();
            assert_eq!(xml_attr(&decoded, "Executable", "Name").as_deref(), Some(r"Bin\game.exe"));
            assert_eq!(xml_attr(&decoded, "Identity", "Name").as_deref(), Some("Contoso.Test"));
            std::fs::write(dir.0.join("MicrosoftGame.config"), &raw).unwrap();
            assert_eq!(read_game_config(&dir.0), (vec![r"Bin\game.exe".into()], Some("Contoso.Test".into())));
        }
        assert!(decode_game_config(&[0xFF, 0xFE, b'<']).is_none());
        assert!(decode_game_config(&[0xFF, 0xFE, 0x00, 0xD8]).is_none());
        assert!(decode_game_config(&[0xFF, 0xFF]).is_none());
        std::fs::write(dir.0.join("MicrosoftGame.config"), [0xFF, 0xFE, b'<']).unwrap();
        assert_eq!(read_game_config(&dir.0), (Vec::new(), None));
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> TempDir {
            let p = std::env::temp_dir().join(format!("xcl_launcher_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn never_picks_xcl_itself() {
        let d = TempDir::new("self");
        let me = std::env::current_exe().unwrap();
        std::fs::copy(&me, d.0.join("xcl.exe")).unwrap();
        std::fs::copy(&me, d.0.join("renamed_copy.exe")).unwrap(); // byte-identical copy of XCL
        let e = resolve_target(&d.0).unwrap_err();
        assert!(e.contains("xcl.exe") && e.contains("XCL itself"), "{e}");
        assert!(resolve_target(&d.0.join("xcl.exe")).unwrap_err().contains("XCL itself"));
        // a real game exe next to it wins
        std::fs::write(d.0.join("game.exe"), b"MZ not really a game").unwrap();
        let (_, exe, _) = resolve_target(&d.0).unwrap();
        assert_eq!(exe.file_name().unwrap(), "game.exe");
    }

    #[test]
    fn finds_exe_in_subfolder_and_via_config() {
        let d = TempDir::new("sub");
        std::fs::create_dir_all(d.0.join("Bin")).unwrap();
        std::fs::write(d.0.join("Bin").join("game.exe"), b"MZ").unwrap();
        let (_, exe, _) = resolve_target(&d.0).unwrap();
        assert!(exe.ends_with("Bin\\game.exe"), "{exe:?}");
        std::fs::write(
            d.0.join("MicrosoftGame.config"),
            r#"<Game><Identity Name="Contoso.Test"/><ExecutableList><Executable Name="Bin\missing.exe"/></ExecutableList></Game>"#,
        )
        .unwrap();
        let e = resolve_target(&d.0).unwrap_err();
        assert!(e.contains("missing.exe"), "{e}");
    }

    #[test]
    fn console_detection() {
        // a plain Windows binary imports no console DLLs
        let me = std::env::current_exe().unwrap();
        assert_eq!(crate::inspect::console_imports(&me), Some(vec![]));
        let sys = std::path::PathBuf::from(std::env::var("SystemRoot").unwrap_or("C:\\Windows".into())).join("System32\\notepad.exe");
        assert_eq!(crate::inspect::console_imports(&sys), Some(vec![]));
        let d = TempDir::new("pe");
        std::fs::write(d.0.join("junk.exe"), b"not a PE").unwrap();
        assert_eq!(crate::inspect::console_imports(&d.0.join("junk.exe")), None);
    }

    #[test]
    fn command_line_quoting() {
        assert_eq!(quote("plain"), "plain");
        assert_eq!(quote("C:\\Games\\My Game\\a.exe"), "\"C:\\Games\\My Game\\a.exe\"");
        assert_eq!(quote("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(quote(""), "\"\"");
    }

    #[test]
    fn cpu_feature_detection_identifies_missing_features() {
        let all_present = check_cpu_features(|_| true);
        assert!(all_present.is_empty());

        let none_present = check_cpu_features(|_| false);
        assert_eq!(
            none_present,
            vec!["SSE4.2", "POPCNT", "AVX", "AVX2", "FMA", "F16C", "BMI1", "BMI2", "LZCNT", "MOVBE"]
        );

        let missing_avx2 = check_cpu_features(|f| f != "avx2");
        assert_eq!(missing_avx2, vec!["AVX2"]);

        let missing_fma = check_cpu_features(|f| f != "fma");
        assert_eq!(missing_fma, vec!["FMA"]);

        let missing_bmi = check_cpu_features(|f| f != "bmi1" && f != "bmi2");
        assert_eq!(missing_bmi, vec!["BMI1", "BMI2"]);

        let missing_zen2_subset = check_cpu_features(|f| {
            !["avx2", "fma", "bmi1", "bmi2", "popcnt", "sse4.2", "lzcnt"].contains(&f)
        });
        assert_eq!(
            missing_zen2_subset,
            vec!["SSE4.2", "POPCNT", "AVX2", "FMA", "BMI1", "BMI2", "LZCNT"]
        );
    }

    #[test]
    fn missing_cpu_message_format() {
        let msg = format_missing_cpu_message(&["AVX2", "FMA", "BMI1", "BMI2", "POPCNT", "SSE4.2", "LZCNT"]);
        assert!(msg.contains("this CPU lacks AVX2, FMA, BMI1, BMI2, POPCNT, SSE4.2, LZCNT"));
        assert!(msg.contains("Zen 2 CPU"));
        assert!(msg.contains("XCL_SKIP_CPU_CHECK=1"));
    }
}
