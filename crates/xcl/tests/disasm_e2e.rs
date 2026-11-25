use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("xcl-disasm-{tag}-{}-{nonce}{ext}", std::process::id()))
}

/// s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm
const CODE: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];

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

#[test]
fn disasm_honors_byte_range_options() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let shader =
        std::env::temp_dir().join(format!("xcl-disasm-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&shader, [0, 0, 0x80, 0xbf, 0, 0, 0x81, 0xbf]).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("disasm")
            .arg(&shader)
            .args(args)
            .output()
    };
    let length = run(&["--length", "4"]);
    let offset = run(&["--offset", "0x4", "--length", "0x4"]);
    let malformed = run(&["--offset", "bogus"]);
    let missing = run(&["--length"]);
    std::fs::remove_file(&shader).unwrap();

    let length = length.unwrap();
    assert!(length.status.success());
    let listing = String::from_utf8(length.stdout).unwrap();
    assert!(listing.contains("00000000: s_nop"), "{listing}");
    assert!(!listing.contains("s_endpgm"), "{listing}");

    let offset = offset.unwrap();
    assert!(offset.status.success());
    let listing = String::from_utf8(offset.stdout).unwrap();
    assert!(listing.contains("00000004: s_endpgm"), "{listing}");
    assert!(!listing.contains("s_nop"), "{listing}");

    for (output, flag) in [(malformed, "--offset"), (missing, "--length")] {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8(output.stderr).unwrap().contains(flag));
    }
}

#[test]
fn disasm_all_flag_disassembles_past_endpgm() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let shader =
        std::env::temp_dir().join(format!("xcl-disasm-all-{}-{nonce}.bin", std::process::id()));
    // s_nop (0xbf800000), s_endpgm (0xbf810000), s_nop (0xbf800000)
    std::fs::write(&shader, [0, 0, 0x80, 0xbf, 0, 0, 0x81, 0xbf, 0, 0, 0x80, 0xbf]).unwrap();

    let without_all = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(&shader)
        .args(["--offset", "0"])
        .output()
        .unwrap();
    let with_all = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(&shader)
        .args(["--offset", "0", "--all"])
        .output()
        .unwrap();
    std::fs::remove_file(&shader).unwrap();

    assert!(without_all.status.success());
    let out_no_all = String::from_utf8(without_all.stdout).unwrap();
    assert!(out_no_all.contains("00000000: s_nop"));
    assert!(out_no_all.contains("00000004: s_endpgm"));
    assert!(!out_no_all.contains("00000008: s_nop"), "{out_no_all}");

    assert!(with_all.status.success());
    let out_all = String::from_utf8(with_all.stdout).unwrap();
    assert!(out_all.contains("00000000: s_nop"));
    assert!(out_all.contains("00000004: s_endpgm"));
    assert!(out_all.contains("00000008: s_nop"), "{out_all}");
}

#[test]
fn disasm_offset_past_eof_exits_code_two() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let shader =
        std::env::temp_dir().join(format!("xcl-disasm-eof-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&shader, [0, 0, 0x80, 0xbf]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(&shader)
        .args(["--offset", "0x100"])
        .output()
        .unwrap();
    std::fs::remove_file(&shader).unwrap();

    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("past the end of the file"), "{err}");
}

#[test]
fn disasm_missing_file_exits_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg("nonexistent_shader_file.bin")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("xcl disasm: nonexistent_shader_file.bin:"), "{err}");
}

#[test]
fn disasm_file_without_rdna2_programs_reports_diagnostic() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let file =
        std::env::temp_dir().join(format!("xcl-disasm-empty-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&file, [0u8; 16]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(&file)
        .output()
        .unwrap();
    std::fs::remove_file(&file).unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("no RDNA2 programs found (ending in s_endpgm)"), "{stdout}");
}

#[test]
fn disasm_missing_arguments_prints_help_and_exits_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .output()
        .expect("spawn xcl disasm");
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("xcl disasm <file>"), "{stdout}");
}

#[test]
fn disasm_xbox_container_unpacks_fssm_and_supports_hex_and_fssm_export() {
    let shader = temp_path("xbox", ".bin");
    let fssm_export = temp_path("exported", ".fssm");

    let fssm_blob = fssm(&CODE, &[(5, 1)]);
    let container = xbox_shader(0, &[fssm_blob.clone()]);
    std::fs::write(&shader, container).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(&shader)
        .args(["--hex", "--fssm", fssm_export.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();

    assert!(stdout.contains("format: xbox-native"), "{stdout}");
    assert!(stdout.contains("console container: stage PS (DXIL)"), "{stdout}");
    assert!(stdout.contains("XDAT 0x0d"), "{stdout}");
    assert!(stdout.contains("XBCH → FSSM (native code)"), "{stdout}");
    assert!(stdout.contains("FSSM #0:"), "{stdout}");
    assert!(stdout.contains("hardware stage: PS"), "{stdout}");
    assert!(stdout.contains("saved FSSM to"), "{stdout}");
    assert!(stdout.contains("s_mov_b32 s0, s1"), "{stdout}");
    assert!(stdout.contains("v_add_f32 v0, v1, v5"), "{stdout}");
    assert!(stdout.contains("s_endpgm"), "{stdout}");

    let exported = std::fs::read(&fssm_export).unwrap();
    assert_eq!(exported, fssm_blob);

    let _ = std::fs::remove_file(&shader);
    let _ = std::fs::remove_file(&fssm_export);
}
