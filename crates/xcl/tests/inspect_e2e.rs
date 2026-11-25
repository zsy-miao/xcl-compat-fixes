//! End-to-end integration tests for `xcl inspect`.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn inspect_missing_arguments_prints_help_and_exits_two() {
    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("inspect")
        .output()
        .expect("spawn xcl inspect");
    assert_eq!(out.status.code(), Some(2));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("xcl inspect <file.exe|file.dll> [--json]"));
}

#[test]
fn inspect_nonexistent_file_exits_two_with_error() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let missing = std::env::temp_dir().join(format!("xcl-inspect-missing-{nonce}.exe"));
    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("inspect")
        .arg(&missing)
        .output()
        .expect("spawn xcl inspect missing");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("xcl inspect: cannot read"));
}

#[test]
fn inspect_non_pe_file_exits_three_with_parse_error() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_file = std::env::temp_dir().join(format!("xcl-inspect-bad-{nonce}.bin"));
    std::fs::write(&temp_file, b"NOT_A_VALID_PE_IMAGE").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("inspect")
        .arg(&temp_file)
        .output()
        .expect("spawn xcl inspect bad");
    std::fs::remove_file(&temp_file).unwrap();

    assert_eq!(out.status.code(), Some(3));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("xcl inspect:"));
    assert!(stderr.contains("already unpacked files"));
}

#[test]
fn inspect_valid_pe_in_text_and_json_modes() {
    let xcl_exe = env!("CARGO_BIN_EXE_xcl");

    // Text output mode
    let out_text = Command::new(xcl_exe)
        .arg("inspect")
        .arg(xcl_exe)
        .output()
        .expect("spawn xcl inspect text");
    assert_eq!(out_text.status.code(), Some(0));
    let stdout_text = String::from_utf8(out_text.stdout).unwrap();
    assert!(stdout_text.contains("File:"));
    assert!(stdout_text.contains("Type:          EXE x64, subsystem"));
    assert!(stdout_text.contains("ImageBase:"));
    assert!(stdout_text.contains("Imports:"));

    // JSON output mode
    let out_json = Command::new(xcl_exe)
        .arg("inspect")
        .arg(xcl_exe)
        .arg("--json")
        .output()
        .expect("spawn xcl inspect json");
    assert_eq!(out_json.status.code(), Some(0));
    let json: serde_json::Value =
        serde_json::from_slice(&out_json.stdout).expect("parse inspect json output");
    assert!(json["file"].as_str().is_some());
    assert_eq!(json["machine"], "0x8664");
    assert_eq!(json["is_dll"], false);
    assert!(json["imports"].is_array());
    assert!(json["size_of_image"].as_str().unwrap().starts_with("0x"));
    assert_eq!(json["subsystem"], 3);
    assert!(json["exception_entries"].as_u64().unwrap_or(0) > 0);
}
