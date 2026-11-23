//! Independent reconstruction of all eight ordinary BC7 modes from BCPack symbol lanes.
//!
//! BCPack field assignments and configuration numbers were measured with synthetic inputs
//! through a separate public reference API; no reference implementation or preset history is
//! included here. Standard BC7 packing and partition/anchor facts follow Microsoft's BC7
//! specification and the MIT-licensed Xbox2Windows `tools/bc7_blocks.py`, revision
//! addd95378dd4c62d1cad4efb2f2b48c30239e166 (https://github.com/R0Y00/Xbox2Windows).
//! Initial preset references and unverified cross-mode prediction are rejected atomically.

/*
Standard BC7 tables adapted from Xbox2Windows:
MIT License

Copyright (c) 2026 Xbox2Windows contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

const MAX_OUTPUT: usize = 16 << 20;
const SYMBOL_BITS: [u8; 16] = [4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Config {
    mode: u8,
    partition: u8,
    rotation: u8,
    selection: u8,
}

#[derive(Clone, Copy)]
struct Mode {
    subsets: u8,
    partition_bits: u8,
    rotation_bits: u8,
    selection_bits: u8,
    colour_bits: u8,
    alpha_bits: u8,
    endpoint_p: bool,
    subset_p: bool,
    index_bits: u8,
    second_bits: u8,
}

const MODES: [Mode; 8] = [
    Mode { subsets: 3, partition_bits: 4, rotation_bits: 0, selection_bits: 0, colour_bits: 4, alpha_bits: 0, endpoint_p: true, subset_p: false, index_bits: 3, second_bits: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, selection_bits: 0, colour_bits: 6, alpha_bits: 0, endpoint_p: false, subset_p: true, index_bits: 3, second_bits: 0 },
    Mode { subsets: 3, partition_bits: 6, rotation_bits: 0, selection_bits: 0, colour_bits: 5, alpha_bits: 0, endpoint_p: false, subset_p: false, index_bits: 2, second_bits: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, selection_bits: 0, colour_bits: 7, alpha_bits: 0, endpoint_p: true, subset_p: false, index_bits: 2, second_bits: 0 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, selection_bits: 1, colour_bits: 5, alpha_bits: 6, endpoint_p: false, subset_p: false, index_bits: 2, second_bits: 3 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, selection_bits: 0, colour_bits: 7, alpha_bits: 8, endpoint_p: false, subset_p: false, index_bits: 2, second_bits: 2 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 0, selection_bits: 0, colour_bits: 7, alpha_bits: 7, endpoint_p: true, subset_p: false, index_bits: 4, second_bits: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, selection_bits: 0, colour_bits: 5, alpha_bits: 5, endpoint_p: true, subset_p: false, index_bits: 2, second_bits: 0 },
];

#[derive(Clone, Copy, Default)]
struct State {
    config: Config,
    // Raw predictor-domain fields, before anchor normalization. Red carries the P bit.
    ep: [[[u8; 2]; 4]; 3],
    // Physical first and second selector planes, before anchor normalization.
    ix: [[u8; 16]; 2],
}

fn config(a: u8, b: u8) -> Option<Config> {
    let key = (u16::from(a) << 5) | u16::from(b);
    let i = CONFIGS.binary_search_by_key(&key, |entry| entry.0).ok()?;
    let value = CONFIGS[i].1;
    Some(Config {
        mode: (value & 7) as u8,
        partition: ((value >> 3) & 63) as u8,
        rotation: ((value >> 9) & 3) as u8,
        selection: ((value >> 11) & 1) as u8,
    })
}

fn unzig(v: u8) -> i16 { (i16::from(v) >> 1) ^ -i16::from(v & 1) }

fn inter(a: u8, b: u8) -> u8 {
    let mut v = 0;
    for bit in 0..4 {
        v |= ((a >> bit) & 1) << (2 * bit);
        v |= ((b >> bit) & 1) << (2 * bit + 1);
    }
    v
}

fn index_residual(v: u8, parity: usize, width: u8) -> u8 {
    let mut n = 0;
    for bit in 0..usize::from(width) { n |= ((v >> (2 * bit + parity)) & 1) << bit; }
    n
}

fn predictor(history: &[State; 256], block: usize, distance: u8, cfg: Config, indices: bool) -> Result<State, String> {
    if distance == 0 { return Ok(State::default()); }
    if usize::from(distance) > block {
        return Err(format!("BC7 block {block} references unavailable initial history at distance {distance}"));
    }
    let previous = history[(block - usize::from(distance)) & 255];
    if previous.config.mode != cfg.mode || previous.config.rotation != cfg.rotation
        || (indices && previous.config.selection != cfg.selection) {
        return Err(format!("BC7 block {block} uses an unverified cross-mode/index-plane predictor"));
    }
    Ok(previous)
}

fn decode_plane(lanes: &[Vec<u8>; 16], row_lanes: [usize; 4], at: usize, bits: u8, previous: &[u8; 16]) -> [u8; 16] {
    let mut out = [0; 16];
    let mask = (1i16 << bits) - 1;
    for row in 0..4 {
        for pair in 0..2 {
            let symbol = lanes[row_lanes[row]][at + pair];
            for within in 0..2 {
                let pixel = row * 4 + pair * 2 + within;
                let parity = if row == 3 { within } else { 1 - within };
                out[pixel] = ((i16::from(previous[pixel]) + unzig(index_residual(symbol, parity, bits))) & mask) as u8;
            }
        }
    }
    out
}

fn subset(cfg: Config, pixel: usize) -> usize {
    match MODES[cfg.mode as usize].subsets {
        1 => 0,
        2 => PARTITIONS_2[cfg.partition as usize][pixel] as usize,
        _ => PARTITIONS_3[cfg.partition as usize][pixel] as usize,
    }
}

fn anchors(cfg: Config) -> [usize; 3] {
    let partition = cfg.partition as usize;
    match MODES[cfg.mode as usize].subsets {
        1 => [0, usize::MAX, usize::MAX],
        2 => [0, ANCHORS_2[partition] as usize, usize::MAX],
        _ => [0, ANCHORS_3[0][partition] as usize, ANCHORS_3[1][partition] as usize],
    }
}

/// BC7 permits reversing a subset's endpoint order and complementing its selectors. This
/// canonicalizes shortened anchor selectors without altering the predictor-domain state.
fn normalized(state: State) -> State {
    let mut out = state;
    let mode = MODES[state.config.mode as usize];
    if mode.subset_p {
        // The second endpoint's redundant P residual is retained in predictor history but
        // has no influence on the emitted shared-P block, even when endpoints are reversed.
        for s in 0..mode.subsets as usize {
            let shared = out.ep[s][0][0] & 1;
            out.ep[s][0][1] = (out.ep[s][0][1] & !1) | shared;
        }
    }
    if mode.second_bits == 0 {
        for (s, anchor) in anchors(state.config).into_iter().enumerate().take(mode.subsets as usize) {
            if out.ix[0][anchor] & (1 << (mode.index_bits - 1)) == 0 { continue; }
            for channel in 0..4 { out.ep[s][channel].swap(0, 1); }
            let max = (1 << mode.index_bits) - 1;
            for pixel in 0..16 {
                if subset(state.config, pixel) == s { out.ix[0][pixel] = max - out.ix[0][pixel]; }
            }
        }
    } else {
        for plane in 0..2 {
            let bits = if plane == 0 { mode.index_bits } else { mode.second_bits };
            if out.ix[plane][0] & (1 << (bits - 1)) == 0 { continue; }
            let colour = if state.config.mode == 4 { plane == usize::from(state.config.selection) } else { plane == 0 };
            if colour {
                for channel in 0..3 { out.ep[0][channel].swap(0, 1); }
            } else { out.ep[0][3].swap(0, 1); }
            let max = (1 << bits) - 1;
            for selector in &mut out.ix[plane] { *selector = max - *selector; }
        }
    }
    out
}

fn put(bits: &mut u128, position: &mut u32, value: u8, width: u8) -> Result<(), String> {
    if *position + u32::from(width) > 128 || (width < 8 && value >= 1 << width) {
        return Err("BC7 packed field exceeds its standard bit allocation".into());
    }
    *bits |= u128::from(value) << *position;
    *position += u32::from(width);
    Ok(())
}

fn pack(state: State) -> Result<[u8; 16], String> {
    let state = normalized(state);
    let cfg = state.config;
    let mode = MODES[cfg.mode as usize];
    let mut bits = 0;
    let mut position = 0;
    put(&mut bits, &mut position, 1 << cfg.mode, cfg.mode + 1)?;
    put(&mut bits, &mut position, cfg.partition, mode.partition_bits)?;
    put(&mut bits, &mut position, cfg.rotation, mode.rotation_bits)?;
    put(&mut bits, &mut position, cfg.selection, mode.selection_bits)?;
    for channel in 0..4 {
        let width = if channel == 3 { mode.alpha_bits } else { mode.colour_bits };
        if width == 0 { continue; }
        for s in 0..mode.subsets as usize {
            for e in 0..2 {
                let field = state.ep[s][channel][e];
                let value = if channel == 0 && (mode.endpoint_p || mode.subset_p) { field >> 1 } else { field };
                put(&mut bits, &mut position, value, width)?;
            }
        }
    }
    if mode.endpoint_p {
        for s in 0..mode.subsets as usize {
            for e in 0..2 { put(&mut bits, &mut position, state.ep[s][0][e] & 1, 1)?; }
        }
    } else if mode.subset_p {
        for s in 0..mode.subsets as usize { put(&mut bits, &mut position, state.ep[s][0][0] & 1, 1)?; }
    }
    let anchors = anchors(cfg);
    for pixel in 0..16 {
        let width = mode.index_bits - u8::from(anchors.contains(&pixel));
        put(&mut bits, &mut position, state.ix[0][pixel], width)?;
    }
    if mode.second_bits != 0 {
        for pixel in 0..16 { put(&mut bits, &mut position, state.ix[1][pixel], mode.second_bits - u8::from(pixel == 0))?; }
    }
    if position != 128 { return Err("BC7 field layout did not fill one 128-bit block".into()); }
    Ok(bits.to_le_bytes())
}

/// Reconstruct complete 16-byte BC7 blocks. Any unsupported prediction or malformed lane
/// discards the entire result; no output is returned until every block has been checked.
pub(super) fn reconstruct(lanes: &[Vec<u8>; 16]) -> Result<Vec<u8>, String> {
    let target = lanes[0].len();
    if target == 0 || target % 2 != 0 || target > MAX_OUTPUT / 8 || lanes.iter().any(|lane| lane.len() != target) {
        return Err("BC7 lane lengths are invalid or exceed the 16 MiB output limit".into());
    }
    for (i, lane) in lanes.iter().enumerate() {
        let max = (1u16 << SYMBOL_BITS[i]) - 1;
        if lane.iter().any(|&symbol| u16::from(symbol) > max) {
            return Err(format!("BC7 lane {i} contains a symbol beyond its alphabet"));
        }
    }
    let count = target / 2;
    let mut output = Vec::new();
    output.try_reserve_exact(count * 16).map_err(|_| "BC7 output allocation failed")?;
    let mut history = [State::default(); 256];
    const RGB_LANES: [[usize; 3]; 3] = [[2, 3, 4], [6, 7, 8], [10, 5, 9]];
    for block in 0..count {
        let at = block * 2;
        let cfg = config(lanes[15][at], lanes[15][at + 1]).ok_or_else(|| format!("BC7 block {block} has an unknown configuration"))?;
        let mode = MODES[cfg.mode as usize];
        let di = inter(lanes[1][at], lanes[1][at + 1]);
        let de = (i16::from(di) + unzig(inter(lanes[0][at], lanes[0][at + 1]))) as u8;
        let ep = predictor(&history, block, de, cfg, false)?;
        let ix = predictor(&history, block, di, cfg, true)?;
        let mut state = State { config: cfg, ..State::default() };
        for s in 0..mode.subsets as usize {
            for channel in 0..4 {
                let width = if channel == 3 { mode.alpha_bits } else { mode.colour_bits + u8::from(channel == 0 && (mode.endpoint_p || mode.subset_p)) };
                if width == 0 { continue; }
                // Rotation exchanges the alpha endpoint lane with one colour lane before
                // quantizing the resulting BC7 fields, not only when the GPU reads pixels.
                let source_channel = if cfg.rotation != 0 && channel == 3 { usize::from(cfg.rotation - 1) }
                    else if cfg.rotation != 0 && channel == usize::from(cfg.rotation - 1) { 3 }
                    else { channel };
                let lane = if source_channel == 3 { if s == 0 { 5 } else { 9 } } else { RGB_LANES[s][source_channel] };
                let mask = (1i16 << width) - 1;
                for endpoint in 0..2 {
                    state.ep[s][channel][endpoint] = ((i16::from(ep.ep[s][channel][endpoint]) + unzig(lanes[lane][at + endpoint])) & mask) as u8;
                }
            }
        }
        if cfg.mode == 4 {
            state.ix[0] = decode_plane(lanes, [6, 7, 8, 10], at, mode.index_bits, &ix.ix[0]);
            state.ix[1] = decode_plane(lanes, [11, 12, 13, 14], at, mode.second_bits, &ix.ix[1]);
        } else {
            state.ix[0] = decode_plane(lanes, [11, 12, 13, 14], at, mode.index_bits, &ix.ix[0]);
            if mode.second_bits != 0 { state.ix[1] = decode_plane(lanes, [6, 7, 8, 10], at, mode.second_bits, &ix.ix[1]); }
        }
        output.extend_from_slice(&pack(state).map_err(|e| format!("BC7 block {block}: {e}"))?);
        history[block & 255] = state;
    }
    Ok(output)
}

// Standard BC7 anchor/partition facts (MIT Xbox2Windows source cited above).
// Configuration records below are independently measured format identifiers, not presets.

const PARTITIONS_2: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
];

const PARTITIONS_3: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2],
    [0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0],
    [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
    [0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2],
    [0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    [0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0],
    [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1],
    [0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1],
    [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2],
    [0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2],
    [0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0],
];

const ANCHORS_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15
];

const ANCHORS_3: [[u8; 64]; 2] = [
    [3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3],
    [15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8],
];

// Measured with public API synthetic controls at external reference 8183ea3c.
// Encoded standard facts: mode[0:2], partition[3:8], rotation[9:10], selector[11].
const CONFIGS: [(u16, u16); 285] = [
    (0, 0x1ff), (1, 0x1f7), (2, 0x1e7), (3, 0x1d7), (4, 0x1bf), (5, 0x1af),
    (6, 0x18f), (7, 0x17f), (8, 0x15f), (9, 0x147), (10, 0x127), (11, 0x117),
    (12, 0x0e7), (13, 0x0d7), (14, 0x0b7), (15, 0x097), (32, 0x1ef), (33, 0x1cf),
    (34, 0x1a7), (35, 0x177), (36, 0x13f), (37, 0x10f), (38, 0x0cf), (39, 0x08f),
    (40, 0x067), (41, 0x037), (42, 0x605), (43, 0xe04), (44, 0x1f3), (45, 0x1c3),
    (46, 0x193), (47, 0x153), (64, 0x1df), (65, 0x19f), (66, 0x157), (67, 0x107),
    (68, 0x0af), (69, 0x05f), (70, 0x00f), (71, 0x604), (72, 0x1db), (73, 0x18b),
    (74, 0x12b), (75, 0x103), (76, 0x0cb), (77, 0x093), (78, 0x053), (79, 0x02b),
    (96, 0x1c7), (97, 0x16f), (98, 0x0ff), (99, 0x087), (100, 0x02f), (101, 0xc04),
    (102, 0x1bb), (103, 0x14b), (104, 0x0fb), (105, 0x0bb), (106, 0x06b), (107, 0x023),
    (108, 0x1a2), (109, 0x102), (110, 0x1f9), (111, 0x199), (128, 0x1b7), (129, 0x137),
    (130, 0x0a7), (131, 0x027), (132, 0x004), (133, 0x183), (134, 0x113), (135, 0x0b3),
    (136, 0x04b), (137, 0x1d2), (138, 0x11a), (139, 0x1f1), (140, 0x179), (141, 0x141),
    (142, 0x0f1), (143, 0x0a9), (160, 0x197), (161, 0x0f7), (162, 0x057), (163, 0x404),
    (164, 0x17b), (165, 0x0f3), (166, 0x08b), (167, 0x01b), (168, 0x152), (169, 0x1e9),
    (170, 0x169), (171, 0x121), (172, 0x0c1), (173, 0x071), (174, 0x019), (175, 0x040),
    (192, 0x187), (193, 0x0c7), (194, 0x007), (195, 0x1b3), (196, 0x10b), (197, 0x083),
    (198, 0x1da), (199, 0x0ba), (200, 0x1b1), (201, 0x139), (202, 0x0d1), (203, 0x069),
    (204, 0x068), (205, 0x000), (206, 0x16a), (207, 0x0da), (224, 0x167), (225, 0x07f),
    (226, 0xa04), (227, 0x143), (228, 0x0ab), (229, 0x013), (230, 0x0aa), (231, 0x191),
    (232, 0x119), (233, 0x0a1), (234, 0x031), (235, 0x028), (236, 0x18a), (237, 0x0d2),
    (238, 0x052), (256, 0x14f), (257, 0x04f), (258, 0x1d3), (259, 0x0eb), (260, 0x043),
    (261, 0x142), (262, 0x1a9), (263, 0x111), (264, 0x081), (265, 0x011), (266, 0x1ea),
    (267, 0x13a), (268, 0x072), (269, 0x002), (288, 0x12f), (289, 0x01f), (290, 0x173),
    (291, 0x0a3), (292, 0x1ca), (293, 0x1e1), (294, 0x131), (295, 0x099), (296, 0x009),
    (297, 0x1c2), (298, 0x0f2), (299, 0x04a), (320, 0x11f), (321, 0x405), (322, 0x123),
    (323, 0x063), (324, 0x112), (325, 0x161), (326, 0x0c9), (327, 0x029), (328, 0x1e2),
    (329, 0x0ea), (330, 0x01a), (352, 0x0ef), (353, 0x204), (354, 0x0e3), (355, 0x00b),
    (356, 0x1d9), (357, 0x109), (358, 0x061), (359, 0x020), (360, 0x122), (361, 0x042),
    (384, 0x0df), (385, 0x1eb), (386, 0x0c3), (387, 0x17a), (388, 0x171), (389, 0x0b9),
    (390, 0x060), (391, 0x182), (392, 0x06a), (416, 0x0bf), (417, 0x1ab), (418, 0x07b),
    (419, 0x092), (420, 0x129), (421, 0x059), (422, 0x1f2), (423, 0x0ca), (424, 0x038),
    (448, 0x09f), (449, 0x16b), (450, 0x03b), (451, 0x1d1), (452, 0x0e9), (453, 0x001),
    (454, 0x162), (455, 0x03a), (480, 0x077), (481, 0x13b), (482, 0x003), (483, 0x189),
    (484, 0x091), (485, 0x018), (486, 0x0c2), (512, 0x06f), (513, 0x11b), (514, 0x1aa),
    (515, 0x151), (516, 0x041), (517, 0x192), (518, 0x062), (544, 0x047), (545, 0x0db),
    (546, 0x132), (547, 0x101), (548, 0x078), (549, 0x10a), (550, 0x030), (576, 0x03f),
    (577, 0x0d3), (578, 0x082), (579, 0x0d9), (580, 0x048), (581, 0x0a2), (608, 0x017),
    (609, 0x09b), (610, 0x1c9), (611, 0x089), (612, 0x1ba), (613, 0x032), (640, 0x006),
    (641, 0x073), (642, 0x1a1), (643, 0x051), (644, 0x14a), (672, 0x205), (673, 0x05b),
    (674, 0x159), (675, 0x021), (676, 0x0e2), (704, 0x005), (705, 0x033), (706, 0x149),
    (707, 0x058), (708, 0x09a), (736, 0x804), (737, 0x1fa), (738, 0x0f9), (739, 0x010),
    (740, 0x02a), (768, 0x1fb), (769, 0x1b2), (770, 0x0e1), (771, 0x19a), (772, 0x00a),
    (800, 0x1e3), (801, 0x15a), (802, 0x0b1), (803, 0x172), (832, 0x1cb), (833, 0x12a),
    (834, 0x079), (835, 0x0fa), (864, 0x1a3), (865, 0x08a), (866, 0x049), (867, 0x0b2),
    (896, 0x19b), (897, 0x05a), (898, 0x039), (899, 0x07a), (928, 0x163), (929, 0x1c1),
    (930, 0x070), (931, 0x022), (960, 0x15b), (961, 0x1b9), (962, 0x050), (963, 0x012),
    (992, 0x133), (993, 0x181), (994, 0x008),
];

#[cfg(test)]
mod tests {
    use super::*;
    // Independently generated synthetic controls; output observed through a separate API.
    const GOLDENS: [([[u8; 2]; 16], [u8; 16]); 8] = [
        ([[0,0], [0,0], [16,12], [177,55], [235,8], [33,158], [220,220], [46,17], [44,71], [11,15], [36,36], [0,0], [12,4], [8,4], [4,4], [17,4]],
         [0x9f,0xe6,0x2e,0xf3,0xe8,0xee,0x5f,0xc9,0x58,0x11,0x00,0x20,0x41,0x02,0x12,0x04]),
        ([[0,0], [0,0], [54,137], [114,160], [56,128], [0,0], [154,161], [87,43], [60,73], [0,0], [0,0], [8,12], [4,12], [8,4], [12,4], [3,14]],
         [0xfe,0x4d,0x67,0x5e,0x39,0x44,0xa9,0x1c,0xe0,0x6d,0x87,0x04,0x49,0x02,0x24,0x09]),
        ([[0,0], [0,0], [20,71], [13,92], [138,97], [203,141], [33,179], [63,63], [26,12], [30,0], [32,54], [4,0], [0,8], [12,12], [4,8], [23,1]],
         [0xfc,0x15,0x7f,0x06,0xee,0xec,0x00,0xd0,0xb9,0xbc,0x66,0x1e,0x10,0x20,0x56,0x41]),
        ([[0,0], [0,0], [207,61], [196,26], [66,22], [0,0], [8,251], [33,115], [37,11], [0,0], [0,0], [0,12], [4,12], [8,12], [4,4], [24,0]],
         [0xf8,0x33,0xe1,0x82,0xa0,0xb8,0xf1,0x36,0x86,0x16,0x6d,0xbd,0xa0,0xa8,0xa2,0x22]),
        ([[0,0], [0,0], [172,235], [45,137], [12,119], [18,32], [8,0], [12,12], [4,12], [0,0], [8,4], [4,12], [0,8], [8,0], [8,8], [1,11]],
         [0xf0,0x56,0xa5,0x9d,0xa0,0x41,0x04,0xaa,0xa8,0x28,0x48,0x02,0x04,0x01,0x80,0x20]),
        ([[0,0], [0,0], [10,197], [205,148], [18,112], [85,15], [4,8], [4,8], [12,8], [0,0], [4,0], [12,8], [4,0], [8,8], [4,4], [1,10]],
         [0xe0,0x85,0x4e,0x46,0x59,0xc5,0x27,0xe0,0x2c,0x08,0x22,0x22,0x14,0x14,0x15,0x01]),
        ([[0,0], [0,0], [202,19], [166,22], [64,42], [5,49], [0,0], [0,0], [0,0], [0,0], [0,0], [4,0], [0,8], [4,8], [12,0], [20,0]],
         [0x40,0xd9,0x7e,0xba,0x00,0x55,0xfa,0xe7,0x10,0x00,0x00,0x01,0x10,0x01,0x11,0x00]),
        ([[0,0], [0,0], [142,84], [83,107], [41,31], [101,226], [194,193], [0,99], [70,86], [3,24], [0,0], [4,8], [8,4], [8,0], [8,0], [0,0]],
         [0x80,0xff,0xa8,0xf0,0x59,0x05,0xdc,0x82,0x63,0xb5,0xe8,0x59,0x2b,0x82,0x02,0x08]),
    ];
    fn lanes(pairs: [[u8; 2]; 16]) -> [Vec<u8>; 16] { pairs.map(|pair| pair.to_vec()) }
    #[test]
    fn independently_observed_controls_cover_all_eight_modes() {
        for (mode, (pairs, expected)) in GOLDENS.iter().enumerate() {
            assert_eq!(reconstruct(&lanes(*pairs)).unwrap(), *expected, "mode {mode}");
        }
    }

    #[test]
    fn previous_block_prediction_preserves_every_mode() {
        for (mode, (pairs, expected)) in GOLDENS.iter().enumerate() {
            let mut input = lanes(*pairs);
            for lane in &mut input { lane.extend([0, 0]); }
            input[1][2] = 1;
            input[15][2..4].copy_from_slice(&pairs[15]);
            let out = reconstruct(&input).unwrap();
            assert_eq!(&out[..16], expected, "first mode {mode}");
            assert_eq!(&out[16..], expected, "predicted mode {mode}");
        }
    }

    #[test]
    fn anchor_normalization_keeps_raw_predictor_history() {
        for (mode, (pairs, _)) in GOLDENS.iter().enumerate() {
            let mut input = lanes(*pairs);
            input[if mode == 4 { 6 } else { 11 }][0] = 2;
            for lane in &mut input { lane.extend([0, 0]); }
            input[1][2] = 1;
            input[15][2..4].copy_from_slice(&pairs[15]);
            let out = reconstruct(&input).unwrap();
            assert_eq!(&out[..16], &out[16..], "normalized history mode {mode}");
        }
    }

    #[test]
    fn malformed_and_unavailable_predictors_return_no_partial_output() {
        let mut input = lanes(GOLDENS[6].0);
        for lane in &mut input { lane.extend([0, 0]); }
        input[15][2..4].copy_from_slice(&[31, 31]);
        assert!(reconstruct(&input).unwrap_err().contains("configuration"));
        input = lanes(GOLDENS[6].0);
        input[1][0] = 1;
        assert!(reconstruct(&input).unwrap_err().contains("initial history"));
        input = lanes(GOLDENS[6].0);
        input[0][0] = 16;
        assert!(reconstruct(&input).unwrap_err().contains("alphabet"));
        input = lanes(GOLDENS[6].0);
        input[3].pop();
        assert!(reconstruct(&input).unwrap_err().contains("length"));
        input = lanes(GOLDENS[6].0);
        for lane in &mut input { lane.extend([0, 0]); }
        input[1][2] = 1;
        input[15][2..4].copy_from_slice(&GOLDENS[5].0[15]);
        assert!(reconstruct(&input).unwrap_err().contains("cross-mode"));
    }

    #[test]
    fn shared_p_bit_output_does_not_erase_redundant_predictor_residual() {
        let pairs = GOLDENS[1].0;
        let mut input: [Vec<u8>; 16] = std::array::from_fn(|_| vec![0; 4]);
        input[15][..2].copy_from_slice(&pairs[15]);
        input[15][2..].copy_from_slice(&pairs[15]);
        input[2][..2].copy_from_slice(&[4, 2]);
        input[1][2] = 1; // both predictors use the previous block
        input[2][3] = 1; // odd endpoint raw field 1 -> 0, same emitted colour field
        let out = reconstruct(&input).unwrap();
        assert_eq!(&out[..16], &out[16..]);
    }

    #[test]
    fn bounded_history_ring_survives_wraparound() {
        let (pairs, expected) = GOLDENS[6];
        let mut input: [Vec<u8>; 16] = std::array::from_fn(|_| vec![0; 600]);
        for lane in 0..16 { input[lane][..2].copy_from_slice(&pairs[lane]); }
        for block in 1..300 {
            input[1][block * 2] = 1;
            input[15][block * 2..block * 2 + 2].copy_from_slice(&pairs[15]);
        }
        input[1][598] = 15;
        input[1][599] = 15; // 255 blocks back
        let out = reconstruct(&input).unwrap();
        assert_eq!(out.len(), 4800);
        assert!(out.chunks_exact(16).all(|block| block == expected));
    }
}
