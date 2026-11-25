//! The hash `build.rs` gives the translators' sources (`XCL_TRANSLATOR_SOURCE_ID`,
//! `XCL_PER_LANE_SOURCE_ID`, `XCL_DIRECT_DXIL_SOURCE_ID`), part of the translated cache's identity
//! (`recomp::pso`). `build.rs` includes this file (`#[path]`), so its tests run with the crate's.

/// FNV-1a 64 of `bytes`, continuing `h`.
pub fn add(h: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

/// `add` of a source file's text with each CRLF taken as LF. rustc reads both the same, but a
/// checkout with `core.autocrlf` has CRLF and one without has LF: hashed as bytes, the same commit
/// got two identities, and a build from a tree where one file was rewritten with LF a third, so
/// none of them found the others' translations.
pub fn add_source(h: &mut u64, text: &[u8]) {
    let mut rest = text;
    while let Some(at) = rest.windows(2).position(|w| w == b"\r\n") {
        add(h, &rest[..at]);
        rest = &rest[at + 1..];
    }
    add(h, rest);
}

#[cfg(test)]
mod tests {
    use super::{add, add_source};

    fn hash(f: fn(&mut u64, &[u8]), text: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        f(&mut h, text);
        h
    }

    #[test]
    fn a_source_hashes_the_same_with_crlf_and_lf_line_endings() {
        let lf = b"fn a() {\n    b();\n}\n";
        let crlf = b"fn a() {\r\n    b();\r\n}\r\n";
        assert_eq!(hash(add_source, crlf), hash(add_source, lf));
        // an LF file keeps the identity it had before
        assert_eq!(hash(add_source, lf), hash(add, lf));
    }

    #[test]
    fn other_carriage_returns_and_other_text_still_count() {
        let lf = b"let s = \"a\";\n";
        assert_ne!(hash(add_source, b"let s = \"a\r\";\n"), hash(add_source, lf));
        assert_ne!(hash(add_source, b"let s = \"b\";\n"), hash(add_source, lf));
        assert_ne!(hash(add_source, b"a\r\r\nb"), hash(add_source, b"a\nb"));
        // only the CR right before an LF goes
        assert_eq!(hash(add_source, b"a\r\r\nb"), hash(add, b"a\r\nb"));
    }
}
