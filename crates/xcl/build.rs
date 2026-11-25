use std::path::{Path, PathBuf};

// `pso.rs` also contains cache storage and pipeline setup. Keep it out of the source hash so
// those changes do not discard translations; bump this only when its container parsing,
// translation options, profiles, or post-translation HLSL changes.
const TRANSLATION_GLUE_REVISION: &str = "pso-glue-v1";

const TRANSLATOR_SOURCES: &[&str] = &[
    "src/gpu/dxbc.rs",
    "src/gpu/rdna2.rs",
    "src/gpu/rdna2_spec.rs",
    "src/gpu/rdna2_xbox.rs",
    "src/gpu/recomp/dxc.rs",
    "src/gpu/recomp/emit.rs",
    "src/gpu/recomp/mesh_wave.rs",
    "src/gpu/recomp/mod.rs",
    "src/gpu/recomp/upper_lanes.rs",
    "src/gpu/recomp/x64_masks.rs",
    "src/gpu/shader.rs",
    "src/gpu/xbox.rs",
];

/// The per-lane form's sources (`XCL_PER_LANE`): part of only its own cache entries' identity,
/// so editing them does not invalidate the HLSL path's.
const PER_LANE_SOURCES: &[&str] = &[
    "src/gpu/recomp/lane_masks.rs",
    "src/gpu/recomp/lower.rs",
    "src/gpu/recomp/lower/wave.rs",
    "src/gpu/recomp/lower_dxil.rs",
    "src/gpu/recomp/ssa.rs",
];

/// The direct DXIL translator's sources (`XCL_DIRECT_DXIL`), part of only its own cache entries'
/// identity: every `.rs` file under `dir` but the test files (`*tests.rs`; tests inside the other
/// files are hashed with them), so a new one cannot be forgotten.
fn direct_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("list direct DXIL sources") {
        let path = entry.expect("list direct DXIL sources").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if path.is_dir() {
            direct_sources(&path, out);
        } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
            out.push(path);
        }
    }
}

// FNV-1a over the sources, each CRLF taken as LF (`source_hash.rs`, tested with the crate)
#[path = "src/source_hash.rs"]
mod source_hash;
use source_hash::{add, add_source};

fn main() {
    // xcl.exe without high-entropy ASLR: the title's memory then lies under 2^40, which a
    // retail title's BVH builder requires of its GPU addresses (#585, PR #631)
    match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
        Ok("msvc") => println!("cargo:rustc-link-arg-bin=xcl=/HIGHENTROPYVA:NO"),
        Ok("gnu") => println!("cargo:rustc-link-arg-bin=xcl=-Wl,--disable-high-entropy-va"),
        _ => {}
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    add(&mut h, b"xcl-translator-source-v1\0");
    add(&mut h, TRANSLATION_GLUE_REVISION.as_bytes());
    add(&mut h, &[0]);
    for relative in TRANSLATOR_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut h, relative.as_bytes());
        add(&mut h, &[0]);
        add_source(&mut h, &std::fs::read(root.join(relative)).expect("read shader translator source"));
        add(&mut h, &[0]);
    }
    println!("cargo:rustc-env=XCL_TRANSLATOR_SOURCE_ID={h:016x}");
    let mut p = 0xcbf2_9ce4_8422_2325u64;
    for relative in PER_LANE_SOURCES {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut p, relative.as_bytes());
        add(&mut p, &[0]);
        add_source(&mut p, &std::fs::read(root.join(relative)).expect("read per-lane source"));
        add(&mut p, &[0]);
    }
    println!("cargo:rustc-env=XCL_PER_LANE_SOURCE_ID={p:016x}");
    // a directory: cargo looks at every file under it
    println!("cargo:rerun-if-changed=src/gpu/dxil");
    let mut sources = Vec::new();
    direct_sources(&root.join("src/gpu/dxil"), &mut sources);
    sources.sort();
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    for path in &sources {
        add(&mut d, path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/").as_bytes());
        add(&mut d, &[0]);
        add_source(&mut d, &std::fs::read(path).expect("read direct DXIL source"));
        add(&mut d, &[0]);
    }
    println!("cargo:rustc-env=XCL_DIRECT_DXIL_SOURCE_ID={d:016x}");
}
