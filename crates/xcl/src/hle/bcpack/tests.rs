//! Stream tests: truncated input and the modes that have no decoder, and the lane stage against
//! the synthetic fixtures of `tests/data/bcpack` (built by its `generate.py`, an encoder written
//! from the format notes, not from this decoder).

use super::{can_decode, decode, Error};

fn data(name: &str) -> Option<Vec<u8>> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/bcpack").join(name)).ok()
}

/// `bc<mode>-<units>-<seed>`, units of 8 bytes of output: (mode, units)
const FIXTURES: [&str; 14] = [
    "bc1-1-8", "bc1-2-0", "bc1-16-1", "bc1-16-9", "bc1-700-2", "bc1-8192-3", "bc3-16-1", "bc4-16-1", "bc5-16-1", "bc6-16-1", "bc7-2-0", "bc7-16-1",
    "bc7-700-2", "bc7-8192-3",
];

/// The symbols of the two 64 KiB fixtures, which have no `.lanes` file (`generate.py`), lane-major.
fn large_fixture_symbols(mode: u8, lanes: usize) -> Vec<u8> {
    let mut s = vec![0u8; 8192 * lanes];
    if mode == 1 {
        for i in (0..8192).step_by(511) {
            s[2 * 8192 + i] = 30;
            s[8 * 8192 + i] = 5;
        }
        for i in (31..8192).step_by(31) {
            s[i] = 2;
        }
    }
    s
}

#[test]
fn fixtures_decode_to_their_lanes() {
    for stem in FIXTURES {
        let mut p = stem.strip_prefix("bc").unwrap().split('-');
        let (mode, units): (u8, usize) = (p.next().unwrap().parse().unwrap(), p.next().unwrap().parse().unwrap());
        let bin = data(&format!("{stem}.bin")).unwrap();
        let per = if matches!(mode, 1 | 4) { 1 } else { 2 };
        let got = super::victory::records(mode, &bin, units / per);
        // bc1-1-8 has states cut to fewer bytes than an encoder that starts at 2^24 can drop
        if stem == "bc1-1-8" {
            assert!(got.is_err(), "{stem}");
            continue;
        }
        let (records, lanes, per) = got.unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want = data(&format!("{stem}.lanes")).unwrap_or_else(|| large_fixture_symbols(mode, lanes));
        assert_eq!(want.len(), units * lanes, "{stem}");
        // `.lanes`: all of lane 0, then lane 1 …; a record: per lane, its symbols of the block
        for (u, unit) in (0..units).map(|u| (u, u / per * lanes * per + u % per)) {
            for j in 0..lanes {
                assert_eq!(records[unit + j * per], want[j * units + u], "{stem}: unit {u}, lane {j}");
            }
        }
    }
}

/// The 8-bit model 11 as US 11,234,023 prints it (its `base_table[8][12]`, counted from 1), and
/// every model's frequencies sum to 65536.
#[test]
fn model_matches_the_patent_excerpt() {
    use super::tables::{FREQ, MODELS};
    let base = [0u32, 7575, 14276, 25440, 41008, 56352, 64256, 65344, 65408];
    let freq = [7575u32, 6701, 5582, 3892, 1918, 494, 34, 1, 1];
    assert_eq!(FREQ[7 * 16 + 11], freq);
    assert_eq!(super::victory::Tables::builtin().start[7 * 16 + 11], base);
    for (m, f) in FREQ.iter().enumerate().take(MODELS) {
        let total: u32 = (0..9).map(|k| f[k] << k.saturating_sub(1)).sum();
        assert_eq!(total, 65536, "model {m}");
        let used = m / 16 + 2;
        assert!(f[..used].iter().all(|&v| v >= 1) && f[used..].iter().all(|&v| v == 0), "model {m}");
    }
}

#[test]
fn truncated_and_empty_streams_fail() {
    assert!(decode(1, &[0xA5; 64], 16 * 8).is_err());
    assert!(decode(1, &[], 16 * 8).is_err());
}

#[test]
fn only_published_modes_decode() {
    assert!(!can_decode(0) && !can_decode(2) && !can_decode(8));
    for mode in [1, 3, 4, 5, 6, 7] {
        assert!(can_decode(mode));
    }
    assert_eq!(decode(2, &[0; 16], 16), Err(Error::Unsupported));
    assert_eq!(decode(1, &[0; 16], 7), Err(Error::OutputSize));
}

#[test]
fn junk_streams_do_not_panic() {
    let mut x = 0x1234_5678u32;
    let junk: Vec<u8> = (0..4096)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        })
        .collect();
    for mode in [1, 3, 4, 5, 6, 7] {
        let block = if mode == 1 || mode == 4 { 8 } else { 16 };
        let _ = decode(mode, &junk, block * 64);
    }
}

/// The BC7 dictionary indices follow VIctory PR #658: a set with 0 bits holds no
/// indices, and pixel 0 is the anchor (top bit clear).
#[test]
fn bc7_dictionary_indices_line_up() {
    for (k, (_, ix)) in super::tables::BC7.iter().enumerate() {
        assert!(ix.bits[0] > 0, "entry {k}: the first index set has no bits");
        let hi = 1i16 << (ix.bits[0] - 1);
        let ok = (ix.bits[1] != 0 || ix.ix[1] == [0; 16]) && (0..hi).contains(&ix.ix[0][0]);
        assert!(ok, "entry {k}: {:?}", ix.ix);
    }
}
