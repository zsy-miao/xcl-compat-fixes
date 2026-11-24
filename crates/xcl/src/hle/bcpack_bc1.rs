//! Adapted from Kalmarv, ivall/xcl PR #273, commit 5f12643999939d07a234081dbc835c0d8a0497c2.
//! Local adaptation (2026-11-23): reuse the existing bounded entropy decoder,
//! retain transactional assembly errors, and cap allocation before reconstruction.
//! Original XCL code is GPL-2.0-only; the format-note provenance below is retained.
//! https://github.com/ivall/xcl/pull/273
//!
//! BC1 blocks from the 12 lanes of a BCPack BC1 stream (one symbol per lane and block).
//!
//! For block `i`, in stream order:
//! * lane 0 is a distance `d` back to the block whose endpoints predict this one's: symbol 0 is
//!   "none" (predict zeros), an even symbol `v` is `v / 2`, an odd one `255 - v / 2`. Block
//!   `i - d` is meant, and before block 0 a built-in history of 255 endpoint sets, the same for
//!   every stream ([`Tables::history`]),
//! * lane 1 was 0 in every stream the format notes describe,
//! * lanes 2–7 are the differences of R0, G0, B0, R1, G1, B1 from the prediction, 0, -1, 1,
//!   -2, 2 … for the symbols 0, 1, 2, 3, 4 …, modulo the channel's range (5, 6, 5 bits),
//! * lanes 8–11 are the selectors of the top-left, top-right, bottom-left and bottom-right 2×2
//!   texels, each symbol the index of one of the 256 possible quadrants in a fixed order
//!   ([`Tables::quadrants`]).
//!
//! The blocks are in the order of the surface's memory, tiled as the console stores it.
//!
//! After the BC1 sections of the public format notes of bcpack-decoder (Tim0219800,
//! https://github.com/Tim0219800/bcpack-decoder, `docs/SPEC.md`) and bcpack-rs (Ahmet Dedeler,
//! https://github.com/Ahmet-Dedeler/bcpack-rs, `docs/BCPACK_SPEC.md`), both MIT; no code was
//! taken from either. Written for XCL on 2026-11-23.



/// The constants of the BC1 rebuild that are not in a stream.
pub struct Tables {
    /// The endpoints `[R0, G0, B0, R1, G1, B1]` of the blocks before a stream's first:
    /// `history[256 - k]` is the block `k` before block 0 (k = 1..=255; entry 0 is never read).
    /// `None`: not known.
    pub history: [Option<[u8; 6]>; 256],
    /// The selectors of each quadrant symbol, 2 bits a texel: bits 0–1 the top-left texel,
    /// 2–3 top-right, 4–5 bottom-left, 6–7 bottom-right.
    pub quadrants: [u8; 256],
}

/// The range of each endpoint channel.
const RANGE: [u8; 6] = [32, 64, 32, 32, 64, 32];

/// The endpoints `[R0, G0, B0, R1, G1, B1]` of a block.
fn endpoints(block: &[u8]) -> [u8; 6] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    [(c0 >> 11) as u8, (c0 >> 5 & 63) as u8, (c0 & 31) as u8, (c1 >> 11) as u8, (c1 >> 5 & 63) as u8, (c1 & 31) as u8]
}

/// The blocks of `symbols` (12 per block, in [`super::decode_lanes`]'s order).
pub fn assemble(symbols: &[u8], tables: &Tables) -> Result<Vec<u8>, String> {
    if symbols.is_empty() || symbols.len() % 12 != 0 || symbols.len() / 12 > (8 << 20) / 8 {
        return Err("BC1 symbol length is invalid or exceeds 8 MiB of blocks".into());
    }
    let blocks = symbols.len() / 12;
    let mut out = Vec::new();
    out.try_reserve_exact(blocks * 8).map_err(|_| "BC1 output allocation failed".to_string())?;
    out.resize(blocks * 8, 0u8);
    for (i, s) in symbols.chunks_exact(12).enumerate() {
        if s[1] != 0 {
            return Err(format!("BC1 block {i} has an unknown nonzero lane 1 symbol"));
        }
        let v = s[0] as usize;
        let d = if v % 2 == 0 { v / 2 } else { 255 - v / 2 };
        let predicted = if d == 0 {
            [0; 6]
        } else if d <= i {
            endpoints(&out[(i - d) * 8..])
        } else {
            let slot = 256 + i - d;
            tables.history[slot].ok_or_else(|| format!("BC1 block {i} refers to unknown fitted history slot {slot}"))?
        };
        let mut e = [0u16; 6];
        for c in 0..6 {
            let v = s[2 + c] as i32;
            let delta = if v % 2 == 0 { v / 2 } else { -(v / 2) - 1 };
            e[c] = (predicted[c] as i32 + delta).rem_euclid(RANGE[c] as i32) as u16;
        }
        let [tl, tr, bl, br] = [8, 9, 10, 11].map(|l| tables.quadrants[s[l] as usize]);
        let b = &mut out[i * 8..i * 8 + 8];
        b[..2].copy_from_slice(&(e[0] << 11 | e[1] << 5 | e[2]).to_le_bytes());
        b[2..4].copy_from_slice(&(e[3] << 11 | e[4] << 5 | e[5]).to_le_bytes());
        // a row of the block is four texels, 2 bits each, the left one lowest
        b[4] = (tl & 15) | (tr & 15) << 4;
        b[5] = (tl >> 4) | (tr >> 4) << 4;
        b[6] = (bl & 15) | (br & 15) << 4;
        b[7] = (bl >> 4) | (br >> 4) << 4;
    }
    Ok(out)
}
