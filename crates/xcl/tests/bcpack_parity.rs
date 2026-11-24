//! Retail BCPack chunks decoded by XCL against a reference decoder's blocks.
//! Set `XCL_SAMPLE_BCPACK` to a directory of `<name>.bc<N>` + `<name>.vic<N>` files;
//! without it the test does nothing.

#[test]
fn retail_chunks_match_the_reference() {
    let Some(dir) = std::env::var_os("XCL_SAMPLE_BCPACK") else { return };
    let dir = std::path::Path::new(&dir);
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("XCL_SAMPLE_BCPACK {}: {e}", dir.display()));
    let mut cases = 0;
    let mut identical = 0;
    let mut bad_bytes = 0u64;
    let mut total_bytes = 0u64;
    for entry in entries {
        let path = entry.unwrap().path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        let Some(mode) = ext.strip_prefix("bc").and_then(|m| m.parse::<u8>().ok()) else { continue };
        let vic = path.with_extension(format!("vic{mode}"));
        let src = std::fs::read(&path).unwrap();
        let want = if vic.exists() {
            std::fs::read(&vic).unwrap()
        } else {
            std::fs::read(path.with_extension("out")).unwrap()
        };
        let got = match xcl::hle::bcpack::decode(mode, &src, want.len()) {
            Ok(got) => got,
            Err(e) => {
                println!("{}: mode {mode} failed: {e}", path.display());
                cases += 1;
                continue;
            }
        };
        cases += 1;
        total_bytes += want.len() as u64;
        if got == want {
            identical += 1;
        } else {
            let n = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            bad_bytes += n as u64;
            println!("{}: mode {mode}, {n} of {} bytes differ", path.file_name().unwrap().to_string_lossy(), want.len());
        }
    }
    println!(
        "cases {cases}, byte-identical {identical}, differing bytes {bad_bytes} of {total_bytes} ({:.4}%)",
        100.0 * bad_bytes as f64 / total_bytes.max(1) as f64
    );
    assert!(cases > 0);
    assert_eq!(identical, cases, "every case must match the reference");
}
