//! `xcl rpf <package> [--top N] [--find FRAGMENT] [--extract FRAGMENT --out FOLDER]`:
//! lists the records of an RPF9 package (name, sizes, signature of the unpacked data) and
//! extracts the matching ones. Same record scan as the shader pack (`gpu::pack`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};

use crate::gpu::pack;

const AUTO_PROBE_SIZE: u64 = 1024 * 1024 * 1024;

/// Printable signature of unpacked data: 4 ASCII bytes or hex.
fn magic(b: &[u8]) -> String {
    let m = &b[..b.len().min(4)];
    if m.len() == 4 && m.iter().all(|c| c.is_ascii_alphanumeric() || *c == b' ') {
        String::from_utf8_lossy(m).into_owned()
    } else {
        m.iter().map(|c| format!("{c:02x}")).collect()
    }
}

/// Choose a deterministic output path without overwriting another record or an
/// existing file. Duplicate inline names are legal because the scanner does not
/// rely on a table of contents; retain both by suffixing later records with their
/// archive offset.
fn extraction_path(out: &Path, name: &str, offset: usize, used: &mut BTreeSet<PathBuf>) -> PathBuf {
    let safe = name.replace(['/', '\\', ':'], "_");
    let first = out.join(&safe);
    if !first.exists() && used.insert(first.clone()) {
        return first;
    }

    let mut n = 0usize;
    loop {
        let suffix = if n == 0 {
            format!("{safe}.{offset:08x}")
        } else {
            format!("{safe}.{offset:08x}.{n}")
        };
        let path = out.join(suffix);
        if !path.exists() && used.insert(path.clone()) {
            return path;
        }
        n += 1;
    }
}

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempPath(PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn atomic_write_with(
    path: &Path,
    data: &[u8],
    before_commit: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "output has no parent"))?;
    let name = path.file_name().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "output has no file name"))?;
    let (temporary, mut file) = loop {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{}.tmp.{}.{id}", name.to_string_lossy(), std::process::id()));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => break (TempPath(candidate), file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };

    file.write_all(data)?;
    file.sync_all()?;
    drop(file);
    if path.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output appeared during extraction"));
    }
    before_commit(&temporary.0)?;
    let from: Vec<u16> = temporary.0.as_os_str().encode_wide().chain([0]).collect();
    let to: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_WRITE_THROUGH) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn atomic_write(path: &Path, data: &[u8]) -> io::Result<()> {
    atomic_write_with(path, data, |_| Ok(()))
}

struct RpfOptions {
    top: usize,
    force_full: bool,
    probe: bool,
    find: Option<String>,
    extract: Option<String>,
    out: Option<PathBuf>,
    probe_conflict: Option<&'static str>,
}

fn parse_args(args: &[String]) -> Result<RpfOptions, String> {
    let mut opts = RpfOptions {
        top: 30,
        force_full: false,
        probe: false,
        find: None,
        extract: None,
        out: None,
        probe_conflict: None,
    };
    let mut p = crate::cli::Parser::new(args);
    while let Some(arg) = p.next_arg() {
        match arg {
            "--top" => opts.top = p.parse_usize("--top")?,
            "--full" => {
                opts.force_full = true;
                opts.probe_conflict.get_or_insert("--full");
            }
            "--probe" => opts.probe = true,
            "--find" => {
                opts.find = Some(p.value("--find")?.to_string());
                opts.probe_conflict.get_or_insert("--find");
            }
            "--extract" => {
                opts.extract = Some(p.value("--extract")?.to_string());
                opts.probe_conflict.get_or_insert("--extract");
            }
            "--out" => {
                opts.out = Some(PathBuf::from(p.value("--out")?));
                opts.probe_conflict.get_or_insert("--out");
            }
            other => return Err(crate::cli::Parser::unexpected(other)),
        }
    }
    if opts.probe {
        if let Some(conflict) = opts.probe_conflict {
            return Err(format!("--probe cannot be combined with {conflict}"));
        }
    }
    Ok(opts)
}

pub fn run(path: &Path, args: &[String]) -> i32 {
    let opts = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("xcl rpf: {e}");
            return 2;
        }
    };
    let top = opts.top;
    let force_full = opts.force_full;
    if opts.probe {
        return crate::rpf_probe::run(path, top);
    }
    let find = opts.find;
    let extract = opts.extract;
    let out = opts.out.unwrap_or_else(|| PathBuf::from("rpf_out"));
    let len = match std::fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            eprintln!("xcl rpf: {}: {error}", path.display());
            return 2;
        }
    };
    if len > AUTO_PROBE_SIZE && !force_full {
        if find.is_some() || extract.is_some() {
            eprintln!("xcl rpf: detailed options on archives larger than 1 GB require --full");
            return 2;
        }
        println!("archive exceeds 1 GB; using bounded-memory probe (pass --full to request a detailed scan)");
        return crate::rpf_probe::run(path, top);
    }
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("xcl rpf: {}: {e}", path.display());
            return 2;
        }
    };
    if !data.starts_with(pack::RPF9_MAGIC) {
        eprintln!("xcl rpf: {} is not an RPF9 package", path.display());
        return 2;
    }
    let recs = pack::scan(&data, 0x40);
    if recs.is_empty() {
        println!("{}: {:.1} MB, no records with plain names", path.display(), data.len() as f64 / 1048576.0);
        crate::rpf_probe::summarize(&data, top);
        return 0;
    }
    let (csum, usum): (usize, usize) = recs.iter().fold((0, 0), |a, r| (a.0 + r.csize, a.1 + r.usize));
    println!(
        "{}: {:.1} MB, {} records, packed {:.1} MB → {:.1} MB",
        path.display(),
        data.len() as f64 / 1048576.0,
        recs.len(),
        csum as f64 / 1048576.0,
        usum as f64 / 1048576.0
    );
    // kinds by extension / name prefix and by signature
    let mut by_ext: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut by_magic: BTreeMap<String, usize> = BTreeMap::new();
    let mut shown = 0;
    let mut extracted = 0;
    let mut extraction_errors = 0;
    let mut output_ready = false;
    let mut extraction_paths = BTreeSet::new();
    for r in &recs {
        let ext = r.name.rsplit_once('.').map(|x| x.1.to_ascii_lowercase()).unwrap_or_else(|| "(no extension)".into());
        let e = by_ext.entry(ext).or_default();
        e.0 += 1;
        e.1 += r.usize;
        let hit = find.as_deref().is_none_or(|f| r.name.contains(f));
        let want = extract.as_deref().is_some_and(|f| r.name.contains(f));
        if !(hit && shown < top) && !want && by_magic.len() > 64 {
            continue;
        }
        let Some(blob) = pack::inflate(&data, r) else {
            if want {
                extraction_errors += 1;
                eprintln!("xcl rpf: cannot unpack {} at {:#x}", r.name, r.offset);
            }
            continue;
        };
        *by_magic.entry(magic(&blob)).or_default() += 1;
        if hit && shown < top {
            println!("  {:#010x} {:>9} → {:>9} B  [{}]  {}", r.offset, r.csize, r.usize, magic(&blob), r.name);
            shown += 1;
        }
        if want {
            if !output_ready {
                if let Err(error) = std::fs::create_dir_all(&out) {
                    eprintln!("xcl rpf: cannot create {}: {error}", out.display());
                    return 2;
                }
                output_ready = true;
            }
            let file = extraction_path(&out, &r.name, r.offset, &mut extraction_paths);
            match atomic_write(&file, &blob) {
                Ok(()) => extracted += 1,
                Err(error) => {
                    extraction_errors += 1;
                    eprintln!("xcl rpf: cannot extract {} to {}: {error}", r.name, file.display());
                }
            }
        }
    }
    println!("extensions (records, unpacked size):");
    for (k, (n, s)) in &by_ext {
        println!("  {k:<20} {n:>7} {:>10.1} MB", *s as f64 / 1048576.0);
    }
    let mut m: Vec<_> = by_magic.into_iter().collect();
    m.sort_by(|a, b| b.1.cmp(&a.1));
    println!("data signatures (from a sample): {}", m.iter().take(12).map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    if extract.is_some() {
        println!("extracted {extracted} to {}", out.display());
    }
    if extraction_errors == 0 { 0 } else { 2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn record(name: &str, payload: &[u8]) -> Vec<u8> {
        let compressed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let mut bytes = vec![(name.len() + 1) as u8];
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.push(50);
        for value in [
            compressed.len() as u32,
            0,
            compressed.len() as u32,
            payload.len() as u32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&compressed);
        bytes
    }

    #[test]
    fn extraction_preserves_duplicate_record_names() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("xcl-rpf-{}-{nonce}", std::process::id()));
        let archive = root.join("synthetic.rpf");
        let out = root.join("out");
        std::fs::create_dir_all(&root).unwrap();

        let mut data = b"9FPR".to_vec();
        data.resize(0x40, 0);
        data.extend(record("PS_Duplicate", b"first"));
        data.extend(record("PS_Duplicate", b"second"));
        let records = pack::scan(&data, 0x40);
        assert_eq!(records.len(), 2);
        std::fs::write(&archive, data).unwrap();

        let args = vec![
            "--extract".into(),
            "PS_Duplicate".into(),
            "--out".into(),
            out.to_string_lossy().into_owned(),
        ];
        assert_eq!(run(&archive, &args), 0);
        assert_eq!(std::fs::read(out.join("PS_Duplicate")).unwrap(), b"first");
        assert_eq!(
            std::fs::read(out.join(format!("PS_Duplicate.{:08x}", records[1].offset))).unwrap(),
            b"second"
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_extraction_cleans_up_an_interrupted_write() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("xcl-rpf-atomic-{}-{nonce}", std::process::id()));
        let output = root.join("record.bin");
        std::fs::create_dir_all(&root).unwrap();

        let error = atomic_write_with(&output, b"complete payload", |_| {
            Err(io::Error::new(io::ErrorKind::Other, "injected before commit"))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(!output.exists());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);

        let error = atomic_write_with(&output, b"replacement", |_| std::fs::write(&output, b"appeared concurrently")).unwrap_err();
        assert!(matches!(error.kind(), io::ErrorKind::AlreadyExists | io::ErrorKind::PermissionDenied));
        assert_eq!(std::fs::read(&output).unwrap(), b"appeared concurrently");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn extraction_reports_an_unusable_output_directory() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("xcl-rpf-error-{}-{nonce}", std::process::id()));
        let archive = root.join("synthetic.rpf");
        let output_file = root.join("not-a-directory");
        std::fs::create_dir_all(&root).unwrap();

        let mut data = b"9FPR".to_vec();
        data.resize(0x40, 0);
        data.extend(record("PS_OutputProbe", b"payload"));
        std::fs::write(&archive, data).unwrap();
        std::fs::write(&output_file, b"preserve me").unwrap();

        let args = vec![
            "--extract".into(),
            "PS_OutputProbe".into(),
            "--out".into(),
            output_file.to_string_lossy().into_owned(),
        ];
        assert_eq!(run(&archive, &args), 2);
        assert_eq!(std::fs::read(&output_file).unwrap(), b"preserve me");

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn extraction_reports_a_corrupt_matching_record() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("xcl-rpf-corrupt-{}-{nonce}", std::process::id()));
        let archive = root.join("synthetic.rpf");
        let out = root.join("out");
        std::fs::create_dir_all(&root).unwrap();

        let name = "PS_CorruptProbe";
        let mut bad_record = record(name, b"payload");
        let data_offset = name.len() + 19;
        bad_record[data_offset..].fill(0xff);
        let mut data = b"9FPR".to_vec();
        data.resize(0x40, 0);
        data.extend(bad_record);
        std::fs::write(&archive, data).unwrap();

        let args = vec!["--extract".into(), name.into(), "--out".into(), out.to_string_lossy().into_owned()];
        assert_eq!(run(&archive, &args), 2);
        assert!(!out.exists());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn truncated_rpf_headers_are_reported_without_panicking() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("xcl-rpf-header-{}-{nonce}", std::process::id()));
        let archive = root.join("truncated.rpf");
        std::fs::create_dir_all(&root).unwrap();

        std::fs::write(&archive, b"9FPR").unwrap();
        assert_eq!(run(&archive, &[]), 0);

        let mut nested = vec![0u8; 0x800];
        nested[..4].copy_from_slice(b"9FPR");
        nested.extend_from_slice(b"9FPR");
        std::fs::write(&archive, nested).unwrap();
        assert_eq!(run(&archive, &[]), 0);

        std::fs::remove_dir_all(root).unwrap();
    }
}
