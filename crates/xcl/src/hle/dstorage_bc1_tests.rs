//! Game-free controls adapted from Kalmarv's PR #273 at 5f126439.
//! The compressed packet and expected custom-table output are its synthetic
//! bc1-2-0 fixture; no game packet, asset, GPU or external helper is required.
use super::super::{
    execute, FileInner, Request, DEST_PHYS_PAGES_BIT, E_DECOMPRESSION, SOURCE_MEMORY_BIT,
};
use super::*;

fn packet() -> Vec<u8> {
    let hex = "1010101010101010050504040404040401010410040410040002000000000000c500f898b0404480d703c8f4689030480002000000000000101010100505050501010101000000009e25f1b532a933de00000000000000000000";
    hex.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}

fn made_up_tables() -> bc1::Tables {
    // Same independently encoded fixture tables as PR273's tests.
    let range = [32usize, 64, 32, 32, 64, 32];
    bc1::Tables {
        history: std::array::from_fn(|s| {
            Some(std::array::from_fn(|k| ((7 * s + 13 * k) % range[k]) as u8))
        }),
        quadrants: std::array::from_fn(|s| s as u8),
    }
}

// Fixed public symbols with the shipped fitted tables: history slots99/109,
// then signed endpoint residuals modulo565 and the published quadrant order.
const FITTED: [u8; 16] = [
    0x09, 0x08, 0x07, 0x9b, 0xf1, 0x3d, 0x38, 0xe1, 0x37, 0xf0, 0xb9, 0xa4, 0xba, 0x67, 0x37, 0x6a,
];

#[test]
fn bc1_twelve_lane_entropy_matches_independent_upstream_blocks() {
    let p = packet();
    assert_eq!(p.len(), 90);
    assert_eq!(
        decode_bc1_with_tables(&p, 16, &made_up_tables()).unwrap(),
        [
            0xba, 0x2a, 0x1e, 0x04, 0x5e, 0x29, 0x51, 0xbf, 0x2e, 0x75, 0xb7, 0x5e, 0x92, 0xa3,
            0xe3, 0xd3
        ]
    );
    assert_eq!(decode_bc1(&p, 16).unwrap(), FITTED);
    assert!(decode_bc1(&p[..45], 16).is_err());
    let mut tail = p;
    tail.push(0xff);
    assert!(decode_bc1(&tail, 16).is_err());
    assert!(supported_mode(1, 90, 8));
    assert!(!supported_mode(1, 90, 9));
    assert!(!supported_mode(7, 90, 8));
    assert!(entropy::decode_entropy(&[0], [], 1).is_err());
    assert!(entropy::decode_entropy(&[0], [7; 17], 1).is_err());
}

#[test]
fn bc1_fields_history_and_unknown_symbols_follow_pr273() {
    let tables = made_up_tables();
    let b0 = [
        0u8,
        0,
        6,
        1,
        0,
        3,
        10,
        2,
        0b11_10_01_00,
        1,
        0b10_10_10_10,
        0b11_00_11_00,
    ];
    let b1 = [2u8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let out = bc1::assemble(&[b0, b1].concat(), &tables).unwrap();
    assert_eq!(&out[..8], &[0xe0, 0x1f, 0xa1, 0xf0, 0x14, 0x0e, 0xca, 0xca]);
    assert_eq!(&out[8..], &[0xe0, 0xff, 0xa1, 0xf0, 0, 0, 0, 0]);
    let mut bad = b0;
    bad[1] = 1;
    assert!(bc1::assemble(&bad, &tables).unwrap_err().contains("lane 1"));
    bad = [0; 12];
    bad[0] = 7; // distance252 at block0 is unknown fitted slot4
    assert!(bc1::assemble(&bad, &bc1_fitted::TABLES)
        .unwrap_err()
        .contains("slot 4"));
    assert!(bc1::assemble(&[0; 13], &tables).is_err());
    let mut seen = [false; 256];
    for &q in &bc1_fitted::TABLES.quadrants {
        assert!(!std::mem::replace(&mut seen[q as usize], true));
    }
}

#[test]
fn bc1_virtual_request_succeeds_or_keeps_destination_untouched() {
    let mut source = packet();
    source.push(0xff);
    let mut destination = [0xa5; 16];
    let mut r = Request {
        options: SOURCE_MEMORY_BIT | 2,
        source: source.as_ptr() as u64,
        source_size: 90,
        destination: destination.as_mut_ptr() as u64,
        destination_size: 16,
        ..Default::default()
    };
    assert_eq!(r.check(true), Ok(()));
    assert_eq!(unsafe { execute(&r, None, None, None) }, Ok(()));
    assert_eq!(destination, FITTED);
    for n in [45, 91] {
        r.source_size = n;
        destination.fill(0xa5);
        assert_eq!(
            unsafe { execute(&r, None, None, None) },
            Err(E_DECOMPRESSION)
        );
        assert_eq!(destination, [0xa5; 16]);
    }
}

#[test]
fn bc1_file_request_reaches_scattered_physical_pages_transactionally() {
    use crate::hle::kernelx::{
        read_physical_pages, write_physical_pages, AllocateTitlePhysicalPages,
        FreeTitlePhysicalPages, PHYS_PAGE_SIZE,
    };
    use std::sync::{Arc, Mutex};
    assert!(crate::gpu::uma::get().is_some());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("xcl-bc1-{}-{nonce}.bin", std::process::id()));
    let mut data = b"bc1!".to_vec();
    data.extend(packet());
    data.push(0xff);
    std::fs::write(&path, data).unwrap();
    let file = FileInner {
        id: 1,
        path: crate::util::wide(&path.to_string_lossy()),
        file: Mutex::new(Some(Arc::new(std::fs::File::open(&path).unwrap()))),
    };
    let (mut checks, mut results, mut valid, mut outputs) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let freed;
    unsafe {
        let mut count = 2;
        let mut pages = [0usize; 2];
        assert_eq!(
            AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut count, pages.as_mut_ptr()),
            1
        );
        let scattered = [pages[1], pages[0]];
        let offset = PHYS_PAGE_SIZE - 8;
        let mut r = Request {
            options: DEST_PHYS_PAGES_BIT | 2,
            offset: 4,
            source_size: 90,
            destination: scattered.as_ptr() as u64,
            destination_page_offset: offset as u16,
            destination_size: 16,
            ..Default::default()
        };
        for n in [90, 45, 91] {
            r.source_size = n;
            valid.push(write_physical_pages(&scattered, offset, &[0xa5; 16]));
            checks.push(r.check(false));
            results.push(execute(&r, Some(&file), Some(&scattered), None));
            let mut b = [0; 16];
            valid.push(read_physical_pages(&scattered, offset, &mut b));
            outputs.push(b);
        }
        freed = FreeTitlePhysicalPages(std::ptr::null_mut(), count, pages.as_mut_ptr());
    }
    drop(file);
    std::fs::remove_file(path).unwrap();
    assert_eq!(freed, 1);
    assert!(valid.into_iter().all(|v| v));
    assert_eq!(checks, vec![Ok(()); 3]);
    assert_eq!(
        results,
        vec![Ok(()), Err(E_DECOMPRESSION), Err(E_DECOMPRESSION)]
    );
    assert_eq!(outputs, vec![FITTED, [0xa5; 16], [0xa5; 16]]);
}
