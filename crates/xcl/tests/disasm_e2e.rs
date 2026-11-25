use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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
