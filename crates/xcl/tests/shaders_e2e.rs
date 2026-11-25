use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm
const CODE: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("xcl-shaders-{tag}-{}-{nonce}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn le32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn dxbc_container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
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

fn fssm(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
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

fn xbch(fssm_bytes: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm_bytes, 6);
    let mut v = b"XBCH".to_vec();
    le32(&mut v, 3);
    le32(&mut v, z.len() as u32);
    le32(&mut v, 0);
    le32(&mut v, fssm_bytes.len() as u32);
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

fn xbox_shader(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    let mut blobs: Vec<(u32, Vec<u8>)> = vec![(0x12, (0u8..16).collect())];
    for f in fssms {
        blobs.push((0x0D, xbch(f)));
    }
    let mut nested = program_header(kind, 0x8000_0007, 0x10);
    nested.extend_from_slice(b"NOBC");
    blobs.push((0x07, nested));
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
    let mut dxil = program_header(kind, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
    dxil.extend_from_slice(&xdat);
    dxil.extend_from_slice(b"NOBC");
    dxbc_container(&[(b"SFI0", &[0; 8]), (b"HASH", &[0; 20]), (b"DXIL", &dxil), (b"PRIV", &[7; 16])])
}

fn rpf9_record(name: &str, payload: &[u8], usize_override: Option<u32>) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec(payload, 6);
    let mut v = vec![(name.len() + 1) as u8];
    v.extend_from_slice(name.as_bytes());
    v.push(0);
    v.push(50);
    for x in [z.len() as u32, 0xddce7, z.len() as u32, usize_override.unwrap_or(payload.len() as u32)] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.extend_from_slice(&z);
    v
}

fn run_shaders(target: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("shaders")
        .arg(target)
        .args(extra)
        .env_remove("XCL_DUMP_HLSL")
        .env_remove("XCL_RECOMP_ONLY")
        .env_remove("XCL_EXAMPLE_HLSL")
        .output()
        .expect("spawn xcl shaders")
}

#[test]
fn shaders_single_file_reports_text_top_json_recomp_and_dxc_compile() {
    let dir = TempDir::new("single");
    let file = dir.path().join("ps.bin");
    std::fs::write(&file, xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    // Default text output
    let text_out = run_shaders(&file, &["--recomp"]);
    assert!(text_out.status.success(), "{text_out:?}");
    let text = String::from_utf8(text_out.stdout).unwrap();
    assert!(text.contains("single file"), "{text}");
    assert!(text.contains("shaders: 1"), "{text}");
    assert!(text.contains("formats: xbox-native 1"), "{text}");
    assert!(text.contains("stages (DXIL): PS 1"), "{text}");
    assert!(text.contains("FSSM variants by hardware stage: PS 1"), "{text}");
    assert!(text.contains("RDNA2 instructions: 3, undecoded dwords: 0 (0.0000%), distinct mnemonics: 3"), "{text}");
    assert!(text.contains("top 3 mnemonics"), "{text}");
    assert!(text.contains("recompiler (kernel mode): programs 1, fully translated 1 (100.0%), translation errors 0"), "{text}");

    // --top 1 limits the printed mnemonic table
    let top_out = run_shaders(&file, &["--top", "1"]);
    assert!(top_out.status.success(), "{top_out:?}");
    let top_text = String::from_utf8(top_out.stdout).unwrap();
    assert!(top_text.contains("top 1 mnemonics"), "{top_text}");
    let ranked_lines: Vec<&str> = top_text
        .lines()
        .skip_while(|l| !l.starts_with("top 1 mnemonics"))
        .skip(1)
        .take_while(|l| l.starts_with("  "))
        .collect();
    assert_eq!(ranked_lines.len(), 1, "{top_text}");

    // --json --recomp --compile 1
    let json_out = run_shaders(&file, &["--json", "--recomp", "--compile", "1"]);
    assert!(json_out.status.success(), "{json_out:?}");
    let json: serde_json::Value = serde_json::from_slice(&json_out.stdout).unwrap();
    assert_eq!(json["source"], "single file");
    assert_eq!(json["shaders"], 1);
    assert_eq!(json["inflate_errors"], 0);
    assert_eq!(json["by_kind"]["xbox-native"], 1);
    assert_eq!(json["by_dxil_stage"]["PS"], 1);
    assert_eq!(json["by_hw_stage"]["PS"], 1);
    assert_eq!(json["fssm"], 1);
    assert_eq!(json["programs"], 1);
    assert_eq!(json["parse_problems"], 0);
    assert_eq!(json["instructions"], 3);
    assert_eq!(json["undecoded_dwords"], 0);
    assert_eq!(json["distinct_mnemonics"], 3);
    assert_eq!(json["mnemonics"].as_array().unwrap().len(), 3);
    let rc = &json["recomp"];
    assert_eq!(rc["programs"], 1, "{json}");
    assert_eq!(rc["complete"], 1, "{json}");
    assert_eq!(rc["translate_errors"], 0, "{json}");
    assert!(rc["hlsl_bytes"].as_u64().unwrap() > 0, "{json}");
    assert_eq!(rc["dxc_compiled"], 1, "{json}");
    assert_eq!(rc["dxc_failed"].as_array().unwrap().len(), 0, "{json}");
}

#[test]
fn shaders_folder_recurses_and_skips_side_files() {
    let dir = TempDir::new("folder");
    let nested = dir.path().join("nested");
    std::fs::create_dir_all(&nested).unwrap();

    std::fs::write(dir.path().join("ps.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();
    std::fs::write(
        nested.join("vs.bin"),
        xbox_shader(1, &[fssm(&CODE, &[(2, 0)]), fssm(&CODE, &[(4, 0)])]),
    )
    .unwrap();
    std::fs::write(dir.path().join("plain.bin"), b"not a shader").unwrap();
    // Side files written by XCL itself in <data>\shaders\ must be skipped by collect_files.
    std::fs::write(dir.path().join("listing.txt"), b"disassembly side file").unwrap();
    std::fs::write(nested.join("dump.fssm.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    let out = run_shaders(dir.path(), &["--json"]);
    assert!(out.status.success(), "{out:?}");
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["source"], "folder, 3 files", "{json}");
    assert_eq!(json["shaders"], 3);
    assert_eq!(json["by_kind"]["xbox-native"], 2);
    assert_eq!(json["by_kind"]["unknown"], 1);
    assert_eq!(json["by_dxil_stage"]["PS"], 1);
    assert_eq!(json["by_dxil_stage"]["VS"], 1);
    assert_eq!(json["by_hw_stage"]["ES"], 1);
    assert_eq!(json["by_hw_stage"]["VS"], 1);
    assert_eq!(json["by_hw_stage"]["PS"], 1);
    assert_eq!(json["fssm"], 3);
    assert_eq!(json["programs"], 3);
    assert_eq!(json["instructions"], 9);
    assert_eq!(json["parse_problems"], 0);
}

#[test]
fn shaders_rpf9_pack_scans_records_and_counts_inflate_errors() {
    let dir = TempDir::new("rpf9");
    let pack_path = dir.path().join("shaders.rpf");
    let ps = xbox_shader(0, &[fssm(&CODE, &[(5, 1)])]);
    let mut pack = b"9FPR".to_vec();
    pack.resize(0x40, 0xAA);
    pack.extend(rpf9_record("PS_Synthetic_1234abcd_Wrapped", &ps, None));
    pack.extend_from_slice(&[0x13, 0x37, 0, 1, 2, 3]);
    pack.extend(rpf9_record("VS_PlainBlob", b"DXBC-not-console", None));
    // Mismatched uncompressed size forces pack::inflate to return None -> inflate_errors == 1.
    pack.extend(rpf9_record("CS_BadInflate", b"synthetic payload", Some(999)));
    pack.extend_from_slice(&[0u8; 40]);
    std::fs::write(&pack_path, &pack).unwrap();

    let out = run_shaders(&pack_path, &["--json"]);
    assert!(out.status.success(), "{out:?}");
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        json["source"].as_str().unwrap().starts_with("RPF9 package "),
        "{json}"
    );
    assert_eq!(json["shaders"], 2, "{json}");
    assert_eq!(json["inflate_errors"], 1, "{json}");
    assert_eq!(json["by_kind"]["xbox-native"], 1);
    assert_eq!(json["by_kind"]["unknown"], 1);
    assert_eq!(json["by_dxil_stage"]["PS"], 1);
    assert_eq!(json["instructions"], 3);
    assert_eq!(json["undecoded_dwords"], 0);
}

#[test]
fn shaders_handles_non_console_file_and_missing_path() {
    let dir = TempDir::new("edge");
    let plain = dir.path().join("plain.bin");
    std::fs::write(&plain, b"not a shader").unwrap();

    let out = run_shaders(&plain, &[]);
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("shaders: 1"), "{text}");
    assert!(text.contains("no console containers (xbox-native)"), "{text}");

    let missing = dir.path().join("does-not-exist.bin");
    let err_out = run_shaders(&missing, &[]);
    assert_eq!(err_out.status.code(), Some(2), "{err_out:?}");
    let stderr = String::from_utf8(err_out.stderr).unwrap();
    assert!(stderr.contains("xcl shaders:"), "{stderr}");
}

#[test]
fn shaders_missing_arguments_prints_help_and_exits_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("shaders")
        .output()
        .expect("spawn xcl shaders");
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("xcl shaders <file | folder | package.rpf>"), "{stdout}");
}

#[test]
fn shaders_required_options_reject_missing_values() {
    let dir = TempDir::new("val");
    let file = dir.path().join("plain.bin");
    std::fs::write(&file, b"not a shader").unwrap();

    let cases: &[(&[&str], &str)] = &[
        (&["--export-hlsl"], "--export-hlsl needs a value"),
        (&["--export-hlsl", "--record-hash", "DEADBEEF"], "--export-hlsl needs a value"),
        (&["--record-hash"], "--record-hash needs a value"),
        (&["--record-hash", "--measure-pso", "CAFE"], "--record-hash needs a value"),
        (&["--measure-pso"], "--measure-pso needs a value"),
        (&["--measure-pso", "--json"], "--measure-pso needs a value"),
    ];

    for (args, expected_msg) in cases {
        let out = run_shaders(&file, args);
        assert_eq!(out.status.code(), Some(2), "args {args:?} unexpected status: {out:?}");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(stderr.contains(expected_msg), "args {args:?}: {stderr}");
    }
}
