//! Strict reconstruction of the documented BCPack BC7 mode-6 subset.
//!
//! The lane/predictor format is documented in Xbox2Windows issue 442:
//! https://github.com/R0Y00/Xbox2Windows/issues/442
//! The synthetic fixture bytes originate from that MIT-licensed repository.
//! This is an independent implementation of those documented field operations.
//! Other inner BC7 modes and fitted initial history are deliberately rejected.

#[derive(Clone, Default)]
struct Block {
    // Red includes its endpoint p-bit; the other channels are seven-bit fields.
    endpoints: [[u8; 2]; 4],
    indices: [u8; 16],
}

fn unzig(value: u8) -> i16 {
    (value as i16 >> 1) ^ -((value & 1) as i16)
}

fn inter(a: u8, b: u8) -> u8 {
    let mut value = 0;
    for bit in 0..4 {
        value |= ((a >> bit) & 1) << (bit * 2);
        value |= ((b >> bit) & 1) << (bit * 2 + 1);
    }
    value
}

fn nibble(value: u8, parity: u8) -> u8 {
    let mut result = 0;
    for bit in 0..4 {
        result |= ((value >> (bit * 2 + parity)) & 1) << bit;
    }
    result
}

fn predictor(history: &[Block], distance: u8) -> Result<Block, String> {
    if distance == 0 {
        Ok(Block::default())
    } else if distance as usize <= history.len() {
        Ok(history[history.len() - distance as usize].clone())
    } else {
        Err(format!("BC7 initial history is unavailable (block {}, distance {distance})", history.len()))
    }
}

fn put(bits: &mut u128, position: &mut u32, value: u8, width: u32) {
    *bits |= u128::from(value) << *position;
    *position += width;
}

/// Returns complete BC7 blocks in their original stored order, never RGBA pixels.
/// No partial output is returned if any block cannot be reconstructed exactly.
pub(super) fn reconstruct_mode6(lanes: &[Vec<u8>; 16]) -> Result<Vec<u8>, String> {
    let target = lanes[0].len();
    if target == 0 || target % 2 != 0 || target > (16 << 20) / 8
        || lanes.iter().any(|lane| lane.len() != target)
    {
        return Err("BC7 lane lengths are invalid or exceed the 16 MiB output limit".into());
    }
    if lanes[..2].iter().flatten().any(|&symbol| symbol > 15) {
        return Err("BC7 distance lane symbol exceeds its four-bit alphabet".into());
    }
    if lanes[6..11].iter().flatten().any(|&symbol| symbol != 0) {
        return Err("BC7 mode-6 packet uses undocumented auxiliary fields".into());
    }
    let blocks = target / 2;
    let mut history = Vec::<Block>::new();
    history.try_reserve_exact(blocks).map_err(|_| "BC7 history allocation failed".to_string())?;
    let mut output = Vec::new();
    output.try_reserve_exact(blocks * 16).map_err(|_| "BC7 output allocation failed".to_string())?;
    for i in 0..blocks {
        let pair = i * 2;
        // The published fixed packets identify an unrotated mode-6 block by
        // (20, 0). Unknown configurations must not be guessed from entropy alone.
        if lanes[15][pair..pair + 2] != [20, 0] {
            return Err(format!("BC7 block {i} has an unsupported configuration ({}, {})", lanes[15][pair], lanes[15][pair + 1]));
        }
        let di = inter(lanes[1][pair], lanes[1][pair + 1]);
        let de = (i16::from(di) + unzig(inter(lanes[0][pair], lanes[0][pair + 1]))) as u8;
        let ep = predictor(&history, de)?;
        let ix = predictor(&history, di)?;
        let mut block = Block::default();
        for channel in 0..4 {
            let mask = if channel == 0 { 255 } else { 127 };
            for endpoint in 0..2 {
                block.endpoints[channel][endpoint] = ((i16::from(ep.endpoints[channel][endpoint])
                    + unzig(lanes[2 + channel][pair + endpoint])) & mask) as u8;
            }
        }
        for row in 0..4 {
            for half in 0..2 {
                let residual = lanes[11 + row][pair + half];
                for pixel in 0..2 {
                    let index = row * 4 + half * 2 + pixel;
                    let parity = if row == 3 { pixel } else { 1 - pixel };
                    block.indices[index] = ((i16::from(ix.indices[index])
                        + unzig(nibble(residual, parity as u8))) & 15) as u8;
                }
            }
        }
        if block.indices[0] >= 8 {
            return Err(format!("BC7 block {i} has an invalid mode-6 anchor selector"));
        }
        let mut bits = 0x40u128;
        let mut position = 7;
        for channel in 0..4 {
            for endpoint in 0..2 {
                let value = block.endpoints[channel][endpoint];
                put(&mut bits, &mut position, if channel == 0 { value >> 1 } else { value }, 7);
            }
        }
        for endpoint in 0..2 {
            put(&mut bits, &mut position, block.endpoints[0][endpoint] & 1, 1);
        }
        for pixel in 0..16 {
            put(&mut bits, &mut position, block.indices[pixel], if pixel == 0 { 3 } else { 4 });
        }
        debug_assert_eq!(position, 128);
        output.extend_from_slice(&bits.to_le_bytes());
        history.push(block);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // MIT synthetic control from R0Y00/Xbox2Windows issue 442. Endpoint and
    // selector symbols come from the independent bounded-rANS decoding test.
    fn one() -> [Vec<u8>; 16] {
        [vec![0,0], vec![0,0], vec![117,223], vec![70,95], vec![40,75], vec![24,124],
            vec![0,0], vec![0,0], vec![0,0], vec![0,0], vec![0,0], vec![8,148],
            vec![228,254], vec![219,107], vec![39,2], vec![20,0]]
    }

    #[test]
    fn published_mode6_symbols_reconstruct_exactly() {
        assert_eq!(reconstruct_mode6(&one()).unwrap(), [0x40,0x31,0x72,0x04,0xa5,0x68,0x19,0xbe,0x02,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0]);
    }

    #[test]
    fn two_distinct_published_blocks_keep_stored_order() {
        // stan495/Xbox2Windows 5e1af169, MIT two-block fixture.
        let second = [[0,0], [0,0], [80,189], [120,25], [35,80], [55,15],
            [0,0], [0,0], [0,0], [0,0], [0,0], [4,52], [196,244], [251,203], [55,7], [20,0]];
        let mut lanes = one();
        for (lane, symbols) in lanes.iter_mut().zip(second) { lane.extend(symbols); }
        let output = reconstruct_mode6(&lanes).unwrap();
        assert_eq!(&output[..16], &reconstruct_mode6(&one()).unwrap());
        assert_eq!(&output[16..], &[0x40,0x0a,0x94,0x37,0x77,0xa3,0xc8,0x78,0x11,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]);
    }

    #[test]
    fn previous_block_prediction_and_refusals_are_transactional() {
        let mut lanes = one();
        for lane in &mut lanes { lane.extend([0,0]); }
        lanes[1][2] = 1; // inter(1,0) = one block back for both predictors.
        lanes[15][2] = 20;
        let output = reconstruct_mode6(&lanes).unwrap();
        assert_eq!(&output[..16], &output[16..]);
        lanes[15][2] = 0;
        assert!(reconstruct_mode6(&lanes).unwrap_err().contains("configuration"));
        let mut missing = one();
        missing[1][0] = 1;
        assert!(reconstruct_mode6(&missing).unwrap_err().contains("initial history"));
        let mut invalid = one();
        invalid[11][0] = 2; // odd nibble 1 -> selector 15, invalid anchor.
        assert!(reconstruct_mode6(&invalid).unwrap_err().contains("anchor"));
        invalid = one();
        invalid[0][0] = 16;
        assert!(reconstruct_mode6(&invalid).unwrap_err().contains("alphabet"));
        invalid = one();
        invalid[6][0] = 1;
        assert!(reconstruct_mode6(&invalid).unwrap_err().contains("auxiliary"));
    }
}
