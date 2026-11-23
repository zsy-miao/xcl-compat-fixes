//! Fixed synthetic BC7 controls; no game data, GPU, or external decoder required.
//! Format, provenance and MIT notices: tests/data/bcpack-synthetic-controls.md.

#[path = "../src/hle/bcpack_bc7_general.rs"]
mod candidate;

const CONTROLS: &[u8] = include_bytes!("../../../tests/data/bcpack-synthetic-controls.bin");
const RECORDS: usize = 2049;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> &'a [u8] {
        let end = self.at.checked_add(count).expect("control offset overflow");
        let bytes = self.bytes.get(self.at..end).expect("truncated synthetic control");
        self.at = end;
        bytes
    }

    fn word(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().expect("four-byte control word")) as usize
    }
}

#[test]
fn independent_synthetic_corpus() {
    assert_eq!(CONTROLS.len(), 115_564, "unexpected synthetic corpus size");
    let mut reader = Reader { bytes: CONTROLS, at: 0 };
    assert_eq!(reader.word(), RECORDS, "unexpected synthetic record count");
    let mut modes_seen = [false; 8];
    let mut blocks_checked = 0;

    for record in 0..RECORDS {
        let count = reader.word();
        // The first 2,032 records are one block; the final 17 are two blocks.
        // Validate before allocating or multiplying any lengths from the fixture.
        assert_eq!(count, if record < 2032 { 2 } else { 4 }, "control {record}: lane length");
        let lanes: [Vec<u8>; 16] = std::array::from_fn(|_| reader.take(count).to_vec());
        let expected_size = reader.word();
        assert_eq!(expected_size, count * 8, "control {record}: output length");
        let expected = reader.take(expected_size);

        // All controls start at the zero predictor. The history controls only
        // refer to their own first block, never to a fitted initial preset.
        assert_eq!(&lanes[0][..2], &[0, 0], "control {record}: initial index history");
        assert_eq!(&lanes[1][..2], &[0, 0], "control {record}: initial endpoint history");
        if count == 4 {
            assert_eq!(&lanes[0][2..], &[0, 0], "control {record}: index history");
            let endpoint_history = if record >= 2033 { [1, 0] } else { [0, 0] };
            assert_eq!(&lanes[1][2..], &endpoint_history, "control {record}: endpoint history");
        }

        let actual = candidate::reconstruct(&lanes)
            .unwrap_or_else(|error| panic!("control {record} refused: {error}"));
        assert_eq!(actual.as_slice(), expected, "control {record}: fixed reference bytes");
        for block in expected.chunks_exact(16) {
            let mode = block[0].trailing_zeros() as usize;
            assert!(mode < 8, "control {record}: invalid reference BC7 mode");
            modes_seen[mode] = true;
            blocks_checked += 1;
        }
    }

    assert_eq!(reader.at, CONTROLS.len(), "trailing synthetic control bytes");
    assert_eq!(modes_seen, [true; 8], "all eight standard BC7 modes must be covered");
    assert_eq!(blocks_checked, 2066);
    println!("2,049 independent synthetic controls / 2,066 BC7 blocks matched fixed expected bytes");
}
