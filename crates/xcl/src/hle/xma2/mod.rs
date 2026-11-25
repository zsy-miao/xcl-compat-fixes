//! Pure-Rust XMA1 (`0x0165`) and XMA2 (`0x0166`) audio decoder.
//!
//! XMA1/XMA2 wrap a WMA Pro subframe codec inside fixed 2048-byte packets with up to 8
//! interleaved 1- or 2-channel streams (up to 16 channels total). Each decoded frame yields
//! 512 samples per channel across variable subframe sizes (512, 256, 128, or 64 samples).
//!
//! Ported from FFmpeg `libavcodec/wmaprodec.c`, `libavcodec/wma.c`,
//! `libavcodec/sinewin_tablegen.h`, `libavutil/float_dsp.c`, and `libavutil/tx_template.c`
//! (LGPL-2.1+, Copyright (c) 2005-2011 Balatoni Denes, Loren Merritt, Baptiste Coudurier,
//! Benjamin Larsson, Ulion, Sascha Sommer, Reimar Döffinger, Lynne), adapted to safe Rust for
//! XCL (GPL-2.0-only) on 2026-11-21.

pub mod tables;

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::OnceLock;

use tables::*;

pub const PACKET_SIZE: usize = 2048;
pub const SAMPLES_PER_FRAME: usize = 512;
pub const MAX_STREAMS: usize = 8;
pub const MAX_CHANNELS_PER_STREAM: usize = 2;
pub const MAX_CHANNELS: usize = MAX_STREAMS * MAX_CHANNELS_PER_STREAM;

const BLOCK_MIN_BITS: usize = 6; // 64 samples
const BLOCK_MAX_BITS: usize = 9; // 512 samples (XMA fixed frame size)
const BLOCK_SIZES: usize = BLOCK_MAX_BITS - BLOCK_MIN_BITS + 1; // 4 sizes: 64, 128, 256, 512
const MAX_SUBFRAMES: usize = 32;
const MAX_BANDS: usize = 29;
const MAX_FRAMESIZE: usize = 32768;
const BIT_PADDING: usize = 64;
const XMA_DECODE_FLAGS: u32 = 0x10d6;
const XMA_LOG2_FRAME_SIZE: usize = 15; // log2(2048) + 4
const INITIAL_SKIP_SAMPLES: usize = 64;

const VLC_BITS: usize = 9;
const SCALE_VLC_BITS: usize = 8;

// The endpoints round to zero and positive infinity as f32. Exponents outside
// this range round to the same values as the nearest endpoint.
const QUANT_MIN_EXP: i32 = -904;
const QUANT_MAX_EXP: i32 = 771;
const QUANT_GAIN_COUNT: usize = (QUANT_MAX_EXP - QUANT_MIN_EXP + 1) as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    Xma1,
    Xma2,
}

/// Configuration for an XMA1 or XMA2 stream parsed from a WAVEFORMAT structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamConfig {
    pub version: Version,
    pub sample_rate: u32,
    pub channels: u16,
    /// Channel count (1 or 2) for each interleaved XMA stream.
    pub stream_channels: Vec<u8>,
}

impl StreamConfig {
    /// Parse an XMA2 configuration (`WAVE_FORMAT_XMA2` = `0x0166`) from `wfx.channels`,
    /// `wfx.rate`, and the `cbSize` extradata bytes following the 18-byte `WAVEFORMATEX`.
    ///
    /// Supports standard `XMA2WAVEFORMATEX` (`cbSize == 34`), legacy `XMA2WAVEFORMAT`
    /// (rev 3/4), and empty extradata (synthesizing a default 2+2+...+1/2 layout).
    pub fn from_xma2_wfx(channels: u16, sample_rate: u32, extradata: &[u8]) -> Result<Self, String> {
        if channels == 0 || channels as usize > MAX_CHANNELS {
            return Err(format!("invalid XMA2 channel count {channels}"));
        }
        let rate = if sample_rate == 0 { 48_000 } else { sample_rate };

        let stream_channels = if extradata.len() == 34 {
            // Standard XMA2WAVEFORMATEX (cbSize == 34): NumStreams u16 LE @ 0.
            let raw_streams = u16::from_le_bytes([extradata[0], extradata[1]]) as usize;
            let num_streams = if raw_streams == 0 {
                (channels as usize).div_ceil(2)
            } else {
                raw_streams
            };
            if num_streams == 0 || num_streams > MAX_STREAMS {
                return Err(format!("invalid XMA2 stream count {num_streams}"));
            }
            if num_streams == channels as usize {
                // Explicit 1-channel-per-stream configuration.
                vec![1u8; num_streams]
            } else {
                let mut sc = Vec::with_capacity(num_streams);
                for i in 0..num_streams {
                    let ch = if (i + 1) * MAX_CHANNELS_PER_STREAM > channels as usize { 1 } else { 2 };
                    sc.push(ch);
                }
                let total: usize = sc.iter().map(|&c| c as usize).sum();
                if total != channels as usize {
                    return Err(format!("XMA2 stream count {num_streams} does not match {channels} channels"));
                }
                sc
            }
        } else if extradata.len() >= 2 {
            // Legacy XMA2WAVEFORMAT: rev @ 0 (3 or 4), num_streams @ 1.
            let rev = extradata[0];
            let num_streams = extradata[1] as usize;
            let header_base = 32 + if rev == 3 { 0 } else { 8 };
            let expected = header_base + 4 * num_streams;
            if num_streams == 0 || num_streams > MAX_STREAMS || extradata.len() != expected {
                return Err(format!(
                    "invalid legacy XMA2 extradata (len={}, rev={rev}, streams={num_streams})",
                    extradata.len()
                ));
            }
            let mut sc = Vec::with_capacity(num_streams);
            for i in 0..num_streams {
                let ch = extradata[header_base + 4 * i];
                if ch == 0 || ch as usize > MAX_CHANNELS_PER_STREAM {
                    return Err(format!("invalid XMA2 stream {i} channel count {ch}"));
                }
                sc.push(ch);
            }
            let total: usize = sc.iter().map(|&c| c as usize).sum();
            if total != channels as usize {
                return Err(format!("XMA2 stream channel sum {total} != {channels}"));
            }
            sc
        } else if extradata.is_empty() {
            default_stream_channels(channels)
        } else {
            return Err(format!("unsupported XMA2 extradata length {}", extradata.len()));
        };

        Ok(Self { version: Version::Xma2, sample_rate: rate, channels, stream_channels })
    }

    /// Parse an XMA1 configuration (`WAVE_FORMAT_XMA` = `0x0165`) from `wfx.channels`,
    /// `wfx.rate`, and the `cbSize` extradata bytes (`XMAWAVEFORMAT`).
    pub fn from_xma1_wfx(channels: u16, sample_rate: u32, extradata: &[u8]) -> Result<Self, String> {
        if channels == 0 || channels as usize > MAX_CHANNELS {
            return Err(format!("invalid XMA1 channel count {channels}"));
        }
        let rate = if sample_rate == 0 { 48_000 } else { sample_rate };

        let stream_channels = if extradata.len() >= 8 {
            let num_streams = extradata[4] as usize;
            let expected = 8 + 20 * num_streams;
            if num_streams == 0 || num_streams > MAX_STREAMS || extradata.len() != expected {
                return Err(format!(
                    "invalid XMA1 extradata (len={}, streams={num_streams})",
                    extradata.len()
                ));
            }
            let mut sc = Vec::with_capacity(num_streams);
            for i in 0..num_streams {
                let ch = extradata[8 + 20 * i + 17];
                if ch == 0 || ch as usize > MAX_CHANNELS_PER_STREAM {
                    return Err(format!("invalid XMA1 stream {i} channel count {ch}"));
                }
                sc.push(ch);
            }
            let total: usize = sc.iter().map(|&c| c as usize).sum();
            if total != channels as usize {
                return Err(format!("XMA1 stream channel sum {total} != {channels}"));
            }
            sc
        } else if extradata.is_empty() {
            default_stream_channels(channels)
        } else {
            return Err(format!("unsupported XMA1 extradata length {}", extradata.len()));
        };

        Ok(Self { version: Version::Xma1, sample_rate: rate, channels, stream_channels })
    }
}

fn default_stream_channels(channels: u16) -> Vec<u8> {
    let n = (channels as usize).div_ceil(2);
    (0..n)
        .map(|i| if (i + 1) * 2 > channels as usize { 1 } else { 2 })
        .collect()
}

// ---------------------------------------------------------------------------------------
// Bitstream reader & writer (MSB-first, safe bounds)
// ---------------------------------------------------------------------------------------

#[derive(Clone)]
struct BitReader<'a> {
    buf: &'a [u8],
    bit_len: usize,
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8], bit_len: usize) -> Self {
        Self { buf, bit_len: bit_len.min(buf.len() * 8), pos: 0 }
    }

    #[inline]
    fn bits_count(&self) -> usize {
        self.pos
    }

    #[inline]
    fn bits_left(&self) -> isize {
        self.bit_len as isize - self.pos as isize
    }

    #[inline]
    fn show_bits(&self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        let byte_idx = self.pos >> 3;
        let bit_off = self.pos & 7;
        // Gather up to 5 bytes safely (zero-padded past end of slice).
        let mut acc: u64 = 0;
        for i in 0..5 {
            let b = self.buf.get(byte_idx + i).copied().unwrap_or(0) as u64;
            acc = (acc << 8) | b;
        }
        let shifted = acc << (24 + bit_off);
        (shifted >> (64 - n)) as u32
    }

    #[inline]
    fn skip_bits(&mut self, n: usize) {
        self.pos = self.pos.saturating_add(n);
    }

    #[inline]
    fn get_bits(&mut self, n: usize) -> u32 {
        let v = self.show_bits(n);
        self.pos = self.pos.saturating_add(n);
        v
    }

    #[inline]
    fn get_bits1(&mut self) -> u32 {
        let v = if self.pos < self.buf.len() * 8 {
            ((self.buf[self.pos >> 3] >> (7 - (self.pos & 7))) & 1) as u32
        } else {
            0
        };
        self.pos = self.pos.saturating_add(1);
        v
    }

    #[inline]
    fn get_sbits(&mut self, n: usize) -> i32 {
        debug_assert!(n > 0 && n <= 32);
        let v = self.get_bits(n);
        ((v as i32) << (32 - n)) >> (32 - n)
    }
}

/// MSB-first bit reservoir writer.
struct BitWriter {
    buf: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self { buf: vec![0u8; MAX_FRAMESIZE + BIT_PADDING], bit_pos: 0 }
    }

    fn reset(&mut self) {
        let bytes = (self.bit_pos + 7) / 8 + 8;
        let clear = bytes.min(self.buf.len());
        self.buf[..clear].fill(0);
        self.bit_pos = 0;
    }

    fn put_bits(&mut self, n: usize, val: u32) {
        for i in (0..n).rev() {
            let byte = self.bit_pos >> 3;
            let bit = 7 - (self.bit_pos & 7);
            if byte < MAX_FRAMESIZE {
                if ((val >> i) & 1) != 0 {
                    self.buf[byte] |= 1 << bit;
                } else {
                    self.buf[byte] &= !(1 << bit);
                }
            }
            self.bit_pos += 1;
        }
    }

    /// Copy `bit_count` bits starting at bit 0 of `src`.
    fn copy_aligned_bits(&mut self, src: &[u8], bit_count: usize) {
        if bit_count == 0 {
            return;
        }
        let dst_off = self.bit_pos & 7;
        if dst_off == 0 {
            let full_bytes = bit_count >> 3;
            let dst_byte = self.bit_pos >> 3;
            let n = full_bytes.min(src.len()).min(MAX_FRAMESIZE.saturating_sub(dst_byte));
            self.buf[dst_byte..dst_byte + n].copy_from_slice(&src[..n]);
            self.bit_pos += n * 8;
            let rem = bit_count & 7;
            if rem > 0 {
                let b = src.get(full_bytes).copied().unwrap_or(0);
                self.put_bits(rem, (b >> (8 - rem)) as u32);
            }
        } else {
            let full_bytes = bit_count >> 3;
            for i in 0..full_bytes {
                let b = src.get(i).copied().unwrap_or(0);
                self.put_bits(8, b as u32);
            }
            let rem = bit_count & 7;
            if rem > 0 {
                let b = src.get(full_bytes).copied().unwrap_or(0);
                self.put_bits(rem, (b >> (8 - rem)) as u32);
            }
        }
        // Ensure zero padding after bit_pos for safe over-read in BitReader.
        let end_byte = (self.bit_pos + 7) >> 3;
        let pad_end = (end_byte + 8).min(self.buf.len());
        if end_byte < pad_end {
            let rem = self.bit_pos & 7;
            if rem != 0 && end_byte > 0 {
                let mask = !((1u8 << (8 - rem)) - 1);
                self.buf[end_byte - 1] &= mask;
            }
            self.buf[end_byte..pad_end].fill(0);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Multi-level canonical VLC decoder (matching FFmpeg's ff_vlc_init_from_lengths)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct VlcEntry {
    /// Positive: decoded symbol (`sym`). Negative: `-subtable_bits`, and `sym` is subtable offset.
    sym: i32,
    len: i16,
}

struct VlcTable {
    root_bits: usize,
    table: Vec<VlcEntry>,
}

struct VlcCode {
    code: u32,
    bits: u8,
    sym: i32,
}

impl VlcTable {
    fn from_lengths(root_bits: usize, lens: &[u8], syms: &[u16], offset: i32) -> Self {
        let mut codes = Vec::with_capacity(lens.len());
        let mut code: u64 = 0;
        for (i, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let sym = syms.get(i).copied().unwrap_or(i as u16) as i32 + offset;
            codes.push(VlcCode { code: code as u32, bits: len, sym });
            code += 1u64 << (32 - len as u32);
        }
        debug_assert_eq!(code, 1u64 << 32, "incomplete Huffman table");
        Self::build(root_bits, &codes)
    }

    fn from_pairs(root_bits: usize, pairs: &[(u8, u8)], offset: i32) -> Self {
        let mut codes = Vec::with_capacity(pairs.len());
        let mut code: u64 = 0;
        for &(sym, len) in pairs {
            if len == 0 {
                continue;
            }
            codes.push(VlcCode { code: code as u32, bits: len, sym: sym as i32 + offset });
            code += 1u64 << (32 - len as u32);
        }
        debug_assert_eq!(code, 1u64 << 32, "incomplete Huffman table");
        Self::build(root_bits, &codes)
    }

    fn build(root_bits: usize, codes: &[VlcCode]) -> Self {
        let mut table = Vec::new();
        Self::build_subtable(&mut table, root_bits, 0, 0, codes);
        Self { root_bits, table }
    }

    fn build_subtable(table: &mut Vec<VlcEntry>, nb_bits: usize, prefix: u32, prefix_bits: usize, codes: &[VlcCode]) -> usize {
        let base = table.len();
        let size = 1usize << nb_bits;
        table.resize(base + size, VlcEntry::default());

        // Partition codes matching this prefix into short (fits in nb_bits) and long (needs subtable).
        let mask: u32 = if prefix_bits == 0 { 0 } else { !0u32 << (32 - prefix_bits) };
        let mut long_groups: Vec<(usize, usize)> = Vec::new(); // (slot, max_rem_bits)

        for c in codes {
            if (c.code & mask) != prefix {
                continue;
            }
            let rem_bits = c.bits as usize - prefix_bits;
            let shifted = (c.code << prefix_bits) >> (32 - nb_bits);
            if rem_bits <= nb_bits {
                let start = shifted as usize;
                let count = 1usize << (nb_bits - rem_bits);
                for slot in start..start + count {
                    table[base + slot] = VlcEntry { sym: c.sym, len: rem_bits as i16 };
                }
            } else {
                let slot = shifted as usize;
                let extra = rem_bits - nb_bits;
                if let Some(g) = long_groups.iter_mut().find(|g| g.0 == slot) {
                    if extra > g.1 {
                        g.1 = extra;
                    }
                } else {
                    long_groups.push((slot, extra));
                }
            }
        }

        for (slot, max_extra) in long_groups {
            let sub_bits = max_extra.min(nb_bits);
            let sub_prefix = prefix | ((slot as u32) << (32 - prefix_bits - nb_bits));
            let sub_base = Self::build_subtable(table, sub_bits, sub_prefix, prefix_bits + nb_bits, codes);
            table[base + slot] = VlcEntry { sym: sub_base as i32, len: -(sub_bits as i16) };
        }

        base
    }

    #[inline]
    fn decode(&self, br: &mut BitReader<'_>) -> Result<i32, &'static str> {
        let mut base = 0usize;
        let mut nb_bits = self.root_bits;
        for _ in 0..4 {
            let idx = br.show_bits(nb_bits) as usize;
            let e = self.table[base + idx];
            if e.len > 0 {
                br.skip_bits(e.len as usize);
                return Ok(e.sym);
            } else if e.len < 0 {
                br.skip_bits(nb_bits);
                base = e.sym as usize;
                nb_bits = (-e.len) as usize;
            } else {
                return Err("invalid VLC code");
            }
        }
        Err("VLC depth exceeded")
    }
}

// ---------------------------------------------------------------------------------------
// Fast O(N log N) IMDCT-half (matching FFmpeg AV_TX_FLOAT_MDCT inverse) & sine windows
// ---------------------------------------------------------------------------------------

/// Precomputed twiddle tables for one power-of-two IMDCT-half size (`N = 64, 128, 256, 512`).
struct ImdctPlan {
    n: usize,
    exp: Vec<(f32, f32)>,
    fft_twiddles: Vec<(f32, f32)>, // size N/4: exp(+i * 2*PI * k / (N/2)) for inverse FFT
    bitrev: Vec<usize>,
}

impl ImdctPlan {
    fn new(n: usize, bits_per_sample: u8) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 8);
        let len2 = n / 2;
        let scale = 1.0_f64 / (len2 as f64) / ((1u64 << (bits_per_sample - 1)) as f64);
        let s_sqrt = scale.sqrt();

        // Matches ff_tx_mdct_gen_exp (scale > 0, theta = 1/8).
        let mut exp = Vec::with_capacity(len2);
        for i in 0..len2 {
            let alpha = (PI * 0.5) * (i as f64 + 0.125) / (len2 as f64);
            exp.push(((alpha.cos() * s_sqrt) as f32, (alpha.sin() * s_sqrt) as f32));
        }

        // Inverse FFT twiddles for size `len2`: W_M^{+k} = cos(2*PI*k/M) + i*sin(2*PI*k/M).
        let mut fft_twiddles = Vec::with_capacity(len2 / 2);
        for k in 0..len2 / 2 {
            let angle = (2.0 * PI * k as f64) / (len2 as f64);
            fft_twiddles.push((angle.cos() as f32, angle.sin() as f32));
        }

        let log_m = len2.trailing_zeros();
        let bitrev = (0..len2).map(|i| i.reverse_bits() >> (usize::BITS - log_m)).collect();

        Self { n, exp, fft_twiddles, bitrev }
    }

    /// Computes FFmpeg's `AV_TX_FLOAT_MDCT` inverse (`imdct_half`): `src[0..N] -> dst[0..N]`.
    fn imdct_half(&self, src: &[f32], dst: &mut [f32]) {
        let n = self.n;
        let len2 = n / 2;
        let len4 = n / 4;
        debug_assert!(src.len() >= n && dst.len() >= n);

        // Pre-rotation with bit-reversed store directly into `z`.
        let mut z = [(0.0f32, 0.0f32); 256]; // max len2 = 512 / 2 = 256
        for i in 0..len2 {
            let re_in = src[n - 1 - 2 * i];
            let im_in = src[2 * i];
            let (er, ei) = self.exp[i];
            // CMUL3(z, tmp, exp)
            z[self.bitrev[i]] = (re_in * er - im_in * ei, re_in * ei + im_in * er);
        }

        // Iterative radix-2 inverse FFT of size `len2`.
        let mut half = 1usize;
        while half < len2 {
            let step = half * 2;
            let tw_stride = len2 / step;
            for base in (0..len2).step_by(step) {
                for j in 0..half {
                    let (wr, wi) = self.fft_twiddles[j * tw_stride];
                    let (ur, ui) = z[base + j];
                    let (vr, vi) = z[base + j + half];
                    let tr = vr * wr - vi * wi;
                    let ti = vr * wi + vi * wr;
                    z[base + j] = (ur + tr, ui + ti);
                    z[base + j + half] = (ur - tr, ui - ti);
                }
            }
            half = step;
        }

        // Post-rotation & interleave (matches ff_tx_mdct_inv lines 1338-1345).
        for i in 0..len4 {
            let i0 = len4 + i;
            let i1 = len4 - i - 1;
            let (s1r, s1i) = (z[i1].1, z[i1].0);
            let (s0r, s0i) = (z[i0].1, z[i0].0);
            let (e1r, e1i) = self.exp[i1];
            let (e0r, e0i) = self.exp[i0];

            dst[2 * i1] = s1r * e1i - s1i * e1r;
            dst[2 * i0 + 1] = s1r * e1r + s1i * e1i;
            dst[2 * i0] = s0r * e0i - s0i * e0r;
            dst[2 * i1 + 1] = s0r * e0r + s0i * e0i;
        }
    }
}

struct StaticTables {
    sf_vlc: VlcTable,
    sf_rl_vlc: VlcTable,
    coef_vlc: [VlcTable; 2],
    vec4_vlc: VlcTable,
    vec2_vlc: VlcTable,
    vec1_vlc: VlcTable,
    windows: [Vec<f32>; BLOCK_SIZES],
    imdct: [ImdctPlan; BLOCK_SIZES],
    quant_gains: [f32; QUANT_GAIN_COUNT],
}

impl StaticTables {
    fn quant_gain(&self, exp: i32) -> f32 {
        self.quant_gains[(exp.clamp(QUANT_MIN_EXP, QUANT_MAX_EXP) - QUANT_MIN_EXP) as usize]
    }
}

fn static_tables() -> &'static StaticTables {
    static TABLES: OnceLock<StaticTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let windows = std::array::from_fn(|i| {
            let n = 1usize << (BLOCK_MIN_BITS + i);
            (0..n).map(|k| (((k as f64) + 0.5) * (PI / (2.0 * n as f64))).sin() as f32).collect()
        });
        let imdct = std::array::from_fn(|i| ImdctPlan::new(1usize << (BLOCK_MIN_BITS + i), 16));
        StaticTables {
            sf_vlc: VlcTable::from_pairs(SCALE_VLC_BITS, &SCALE_TABLE, -60),
            sf_rl_vlc: VlcTable::from_pairs(VLC_BITS, &SCALE_RL_TABLE, 0),
            coef_vlc: [
                VlcTable::from_lengths(VLC_BITS, &COEF0_LENS, &COEF0_SYMS, 0),
                VlcTable::from_pairs(VLC_BITS, &COEF1_TABLE, 0),
            ],
            vec4_vlc: VlcTable::from_lengths(VLC_BITS, &VEC4_LENS, &VEC4_SYMS, -1),
            vec2_vlc: VlcTable::from_pairs(VLC_BITS, &VEC2_TABLE, -1),
            vec1_vlc: VlcTable::from_pairs(VLC_BITS, &VEC1_TABLE, 0),
            windows,
            imdct,
            quant_gains: std::array::from_fn(|i| {
                let exp = QUANT_MIN_EXP + i as i32;
                10.0_f64.powf(exp as f64 / 20.0) as f32
            }),
        }
    })
}

/// Sine windowing & overlap-add (`vector_fmul_window_c` from FFmpeg `float_dsp.c`).
fn vector_fmul_window(buf: &mut [f32], win: &[f32], len: usize) {
    debug_assert!(buf.len() >= 2 * len && win.len() >= 2 * len);
    for m in 0..len {
        let j = len - 1 - m;
        let s0 = buf[m];
        let s1 = buf[len + j];
        let wi = win[m];
        let wj = win[len + j];
        buf[m] = s0 * wj - s1 * wi;
        buf[len + j] = s0 * wi + s1 * wj;
    }
}

/// XMA effective rate for scale-factor band calculation (`get_rate` in `wmaprodec.c`).
fn xma_sfb_rate(sample_rate: u32) -> usize {
    if sample_rate > 44_100 {
        48_000
    } else if sample_rate > 32_000 {
        44_100
    } else if sample_rate > 24_000 {
        32_000
    } else {
        24_000
    }
}

/// Decode an uncompressed large coefficient (`ff_wma_get_large_val` in `wma.c`).
fn get_large_val(br: &mut BitReader<'_>) -> u32 {
    let mut n_bits = 8usize;
    if br.get_bits1() != 0 {
        n_bits += 8;
        if br.get_bits1() != 0 {
            n_bits += 8;
            if br.get_bits1() != 0 {
                n_bits += 7;
            }
        }
    }
    br.get_bits(n_bits)
}

/// Decode run-level coded coefficients (`ff_wma_run_level_decode` in `wma.c`, version=1).
fn run_level_decode(
    br: &mut BitReader<'_>,
    vlc: &VlcTable,
    level_table: &[f32],
    run_table: &[u16],
    coeffs: &mut [f32],
    mut offset: usize,
    subframe_len: usize,
    esc_len: usize,
) -> Result<(), &'static str> {
    let coef_mask = subframe_len - 1;
    while offset < subframe_len {
        let code = vlc.decode(br)?;
        if code > 1 {
            let idx = code as usize;
            offset += run_table[idx] as usize;
            let sign_bit = br.get_bits1();
            let val = if sign_bit != 0 { level_table[idx] } else { -level_table[idx] };
            coeffs[offset & coef_mask] = val;
        } else if code == 1 {
            // End-of-block.
            break;
        } else {
            // Escape code.
            let level = get_large_val(br) as i32;
            if br.get_bits1() != 0 {
                if br.get_bits1() != 0 {
                    if br.get_bits1() != 0 {
                        return Err("broken RLE escape sequence");
                    } else {
                        offset += br.get_bits(esc_len) as usize + 4;
                    }
                } else {
                    offset += br.get_bits(2) as usize + 1;
                }
            }
            let sign_bit = br.get_bits1();
            let signed = if sign_bit != 0 { level } else { -level };
            coeffs[offset & coef_mask] = signed as f32;
        }
        offset += 1;
    }
    if offset > subframe_len {
        return Err("overflow in spectral RLE");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Per-channel & per-stream decoder state (1 or 2 channels per XMA stream)
// ---------------------------------------------------------------------------------------

struct ChannelState {
    prev_block_len: usize,
    transmit_coefs: bool,
    num_subframes: usize,
    subframe_len: [usize; MAX_SUBFRAMES],
    subframe_offset: [usize; MAX_SUBFRAMES],
    cur_subframe: usize,
    decoded_samples: usize,
    grouped: bool,
    quant_step: i32,
    reuse_sf: bool,
    scale_factor_step: i32,
    max_scale_factor: i32,
    saved_scale_factors: [[i32; MAX_BANDS]; 2],
    scale_factor_idx: usize,
    cur_sf_idx: usize,
    table_idx: usize,
    coeffs_offset: usize,
    num_vec_coeffs: usize,
    out: [f32; SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2],
}

impl ChannelState {
    fn new() -> Self {
        Self {
            prev_block_len: SAMPLES_PER_FRAME,
            transmit_coefs: false,
            num_subframes: 0,
            subframe_len: [0; MAX_SUBFRAMES],
            subframe_offset: [0; MAX_SUBFRAMES],
            cur_subframe: 0,
            decoded_samples: 0,
            grouped: false,
            quant_step: 0,
            reuse_sf: false,
            scale_factor_step: 1,
            max_scale_factor: 0,
            saved_scale_factors: [[0; MAX_BANDS]; 2],
            scale_factor_idx: 0,
            cur_sf_idx: 1,
            table_idx: 0,
            coeffs_offset: 0,
            num_vec_coeffs: 0,
            out: [0.0; SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2],
        }
    }
}

#[derive(Default)]
struct ChannelGroup {
    num_channels: usize,
    transform: bool,
    transform_band: [bool; MAX_BANDS],
    decorrelation_matrix: [f32; MAX_CHANNELS_PER_STREAM * MAX_CHANNELS_PER_STREAM],
    channel_indices: [usize; MAX_CHANNELS_PER_STREAM],
}

struct StreamDecoder {
    version: Version,
    nb_channels: usize,
    bits_per_sample: u8,
    len_prefix: bool,
    dynamic_range_compression: bool,
    max_num_subframes: usize,
    subframe_len_bits: usize,
    max_subframe_len_bit: bool,
    min_samples_per_subframe: usize,
    num_sfb: [usize; BLOCK_SIZES],
    sfb_offsets: [[usize; MAX_BANDS]; BLOCK_SIZES],
    sf_offsets: [[[usize; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES],

    // Packet / bit-reservoir state
    reservoir: BitWriter,
    num_saved_bits: usize,
    frame_offset: usize,
    packet_offset: usize,
    next_packet_start: usize,
    packet_loss: bool,
    packet_done: bool,
    eof_done: bool,
    has_decoded: bool,
    skip_frame: bool,
    skip_packets: u8,
    trim_start: u16,
    trim_end: u16,

    // Per-subframe working state
    subframe_len: usize,
    channels_for_cur_subframe: usize,
    channel_indexes_for_cur_subframe: [usize; MAX_CHANNELS_PER_STREAM],
    num_bands: usize,
    transmit_num_vec_coeffs: bool,
    table_idx: usize,
    esc_len: usize,
    parsed_all_subframes: bool,
    num_chgroups: usize,
    chgroup: [ChannelGroup; MAX_CHANNELS_PER_STREAM],
    channel: [ChannelState; MAX_CHANNELS_PER_STREAM],
    tmp: [f32; SAMPLES_PER_FRAME],
}

impl StreamDecoder {
    fn new(version: Version, nb_channels: usize, sample_rate: u32) -> Self {
        let decode_flags = XMA_DECODE_FLAGS;
        let log2_max_num_subframes = ((decode_flags & 0x38) >> 3) as usize; // 2
        let max_num_subframes = 1usize << log2_max_num_subframes; // 4
        let max_subframe_len_bit = max_num_subframes == 16 || max_num_subframes == 4; // true
        let subframe_len_bits = (usize::BITS - 1 - log2_max_num_subframes.leading_zeros()) as usize + 1; // 2
        let num_possible_block_sizes = log2_max_num_subframes + 1; // 3 (512, 256, 128)
        let min_samples_per_subframe = SAMPLES_PER_FRAME / max_num_subframes; // 128

        let rate = xma_sfb_rate(sample_rate);
        let mut num_sfb = [0usize; BLOCK_SIZES];
        let mut sfb_offsets = [[0usize; MAX_BANDS]; BLOCK_SIZES];
        let mut sf_offsets = [[[0usize; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES];

        for i in 0..num_possible_block_sizes {
            let subframe_len = SAMPLES_PER_FRAME >> i;
            let mut band = 1usize;
            sfb_offsets[i][0] = 0;
            for &cf in &CRITICAL_FREQ {
                if sfb_offsets[i][band - 1] >= subframe_len || band >= MAX_BANDS {
                    break;
                }
                let offset = ((subframe_len * 2 * cf as usize) / rate + 2) & !3;
                if offset > sfb_offsets[i][band - 1] {
                    sfb_offsets[i][band] = offset;
                    band += 1;
                }
                if offset >= subframe_len {
                    break;
                }
            }
            sfb_offsets[i][band - 1] = subframe_len;
            num_sfb[i] = band - 1;
        }

        for i in 0..num_possible_block_sizes {
            for b in 0..num_sfb[i] {
                let offset = ((sfb_offsets[i][b] + sfb_offsets[i][b + 1] - 1) << i) >> 1;
                for x in 0..num_possible_block_sizes {
                    let mut v = 0usize;
                    while v + 1 < MAX_BANDS && (sfb_offsets[x][v + 1] << x) < offset {
                        v += 1;
                    }
                    sf_offsets[i][x][b] = v;
                }
            }
        }

        Self {
            version,
            nb_channels,
            bits_per_sample: 16,
            len_prefix: (decode_flags & 0x40) != 0,
            dynamic_range_compression: (decode_flags & 0x80) != 0,
            max_num_subframes,
            subframe_len_bits,
            max_subframe_len_bit,
            min_samples_per_subframe,
            num_sfb,
            sfb_offsets,
            sf_offsets,
            reservoir: BitWriter::new(),
            num_saved_bits: 0,
            frame_offset: 0,
            packet_offset: 0,
            next_packet_start: 0,
            packet_loss: true,
            packet_done: false,
            eof_done: false,
            has_decoded: false,
            skip_frame: true,
            skip_packets: 0,
            trim_start: 0,
            trim_end: 0,
            subframe_len: SAMPLES_PER_FRAME,
            channels_for_cur_subframe: 0,
            channel_indexes_for_cur_subframe: [0; MAX_CHANNELS_PER_STREAM],
            num_bands: 0,
            transmit_num_vec_coeffs: false,
            table_idx: 0,
            esc_len: 0,
            parsed_all_subframes: false,
            num_chgroups: 0,
            chgroup: [ChannelGroup::default(), ChannelGroup::default()],
            channel: [ChannelState::new(), ChannelState::new()],
            tmp: [0.0; SAMPLES_PER_FRAME],
        }
    }

    fn flush(&mut self) {
        for c in 0..self.nb_channels {
            self.channel[c].out.fill(0.0);
            self.channel[c].prev_block_len = SAMPLES_PER_FRAME;
        }
        self.reservoir.reset();
        self.num_saved_bits = 0;
        self.frame_offset = 0;
        self.packet_offset = 0;
        self.next_packet_start = 0;
        self.packet_loss = true;
        self.packet_done = false;
        self.eof_done = false;
        self.has_decoded = false;
        self.skip_frame = true;
        self.skip_packets = 0;
        self.trim_start = 0;
        self.trim_end = 0;
    }

    fn save_bits(&mut self, pgb: &mut BitReader<'_>, mut len: usize, append: bool) {
        let buflen = if !append {
            self.frame_offset = pgb.bits_count() & 7;
            self.num_saved_bits = self.frame_offset;
            self.reservoir.reset();
            (self.num_saved_bits + len + 7) >> 3
        } else {
            (self.reservoir.bit_pos + len + 7) >> 3
        };

        if len == 0 || buflen > MAX_FRAMESIZE {
            self.packet_loss = true;
            return;
        }

        self.num_saved_bits += len;
        if !append {
            let start_byte = pgb.bits_count() >> 3;
            let slice = pgb.buf.get(start_byte..).unwrap_or(&[]);
            self.reservoir.copy_aligned_bits(slice, self.num_saved_bits);
            pgb.skip_bits(len);
        } else {
            let align = (8 - (pgb.bits_count() & 7)).min(len);
            let bits = pgb.get_bits(align);
            self.reservoir.put_bits(align, bits);
            len -= align;
            if len > 0 {
                let start_byte = pgb.bits_count() >> 3;
                let slice = pgb.buf.get(start_byte..).unwrap_or(&[]);
                self.reservoir.copy_aligned_bits(slice, len);
                pgb.skip_bits(len);
            }
        }
    }

    fn decode_subframe_length(&self, br: &mut BitReader<'_>, offset: usize) -> Result<usize, &'static str> {
        if offset == SAMPLES_PER_FRAME - self.min_samples_per_subframe {
            return Ok(self.min_samples_per_subframe);
        }
        if br.bits_left() < 1 {
            return Err("truncated subframe length");
        }
        let shift = if self.max_subframe_len_bit {
            if br.get_bits1() != 0 {
                1 + br.get_bits(self.subframe_len_bits - 1) as usize
            } else {
                0
            }
        } else {
            br.get_bits(self.subframe_len_bits) as usize
        };
        let subframe_len = SAMPLES_PER_FRAME >> shift;
        if subframe_len < self.min_samples_per_subframe || subframe_len > SAMPLES_PER_FRAME {
            return Err("invalid subframe length");
        }
        Ok(subframe_len)
    }

    fn decode_tilehdr(&mut self, br: &mut BitReader<'_>) -> Result<(), &'static str> {
        let mut num_samples = [0usize; MAX_CHANNELS_PER_STREAM];
        let mut contains_subframe = [false; MAX_CHANNELS_PER_STREAM];
        let mut channels_for_cur = self.nb_channels;
        let mut min_channel_len = 0usize;

        for c in 0..self.nb_channels {
            self.channel[c].num_subframes = 0;
        }
        let fixed_channel_layout = self.max_num_subframes == 1 || br.get_bits1() != 0;

        loop {
            for c in 0..self.nb_channels {
                if num_samples[c] == min_channel_len {
                    contains_subframe[c] = if fixed_channel_layout
                        || channels_for_cur == 1
                        || min_channel_len == SAMPLES_PER_FRAME - self.min_samples_per_subframe
                    {
                        true
                    } else {
                        br.get_bits1() != 0
                    };
                } else {
                    contains_subframe[c] = false;
                }
            }

            let subframe_len = self.decode_subframe_length(br, min_channel_len)?;
            min_channel_len += subframe_len;

            for c in 0..self.nb_channels {
                if contains_subframe[c] {
                    let chan = &mut self.channel[c];
                    if chan.num_subframes >= MAX_SUBFRAMES {
                        return Err("too many subframes");
                    }
                    chan.subframe_len[chan.num_subframes] = subframe_len;
                    num_samples[c] += subframe_len;
                    chan.num_subframes += 1;
                    if num_samples[c] > SAMPLES_PER_FRAME {
                        return Err("channel subframe length overflow");
                    }
                } else if num_samples[c] <= min_channel_len {
                    if num_samples[c] < min_channel_len {
                        channels_for_cur = 0;
                        min_channel_len = num_samples[c];
                    }
                    channels_for_cur += 1;
                }
            }

            if min_channel_len >= SAMPLES_PER_FRAME {
                break;
            }
        }

        for c in 0..self.nb_channels {
            let mut offset = 0usize;
            for i in 0..self.channel[c].num_subframes {
                self.channel[c].subframe_offset[i] = offset;
                offset += self.channel[c].subframe_len[i];
            }
        }
        Ok(())
    }

    fn decode_channel_transform(&mut self, br: &mut BitReader<'_>) -> Result<(), &'static str> {
        self.num_chgroups = 0;
        if self.nb_channels <= 1 {
            return Ok(());
        }
        let mut remaining_channels = self.channels_for_cur_subframe;
        if br.get_bits1() != 0 {
            return Err("unsupported channel transform bit");
        }

        let mut g_idx = 0usize;
        while remaining_channels > 0 && g_idx < self.channels_for_cur_subframe {
            let mut grp = ChannelGroup::default();

            // XMA streams have at most 2 channels, so remaining_channels <= 2 always.
            grp.num_channels = remaining_channels;
            let mut n = 0usize;
            for i in 0..self.channels_for_cur_subframe {
                let c = self.channel_indexes_for_cur_subframe[i];
                if !self.channel[c].grouped {
                    grp.channel_indices[n] = c;
                    n += 1;
                }
                self.channel[c].grouped = true;
            }

            if grp.num_channels == 2 {
                if br.get_bits1() != 0 {
                    if br.get_bits1() != 0 {
                        return Err("unknown 2ch transform type");
                    }
                } else {
                    grp.transform = true;
                    // For 2-channel streams (all XMA stereo streams):
                    grp.decorrelation_matrix = [1.0, -1.0, 1.0, 1.0];
                }
            }

            if grp.transform {
                if br.get_bits1() == 0 {
                    for b in 0..self.num_bands {
                        grp.transform_band[b] = br.get_bits1() != 0;
                    }
                } else {
                    for b in 0..self.num_bands {
                        grp.transform_band[b] = true;
                    }
                }
            }

            remaining_channels -= grp.num_channels;
            self.chgroup[g_idx] = grp;
            g_idx += 1;
        }
        self.num_chgroups = g_idx;
        Ok(())
    }

    fn decode_scale_factors(&mut self, br: &mut BitReader<'_>, st: &StaticTables) -> Result<(), &'static str> {
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let dst_idx = 1 - self.channel[c].scale_factor_idx;
            self.channel[c].cur_sf_idx = dst_idx;

            if self.channel[c].reuse_sf {
                let src_idx = self.channel[c].scale_factor_idx;
                let ref_tbl = self.channel[c].table_idx;
                let map = &self.sf_offsets[self.table_idx][ref_tbl];
                for b in 0..self.num_bands {
                    self.channel[c].saved_scale_factors[dst_idx][b] = self.channel[c].saved_scale_factors[src_idx][map[b]];
                }
            }

            if self.channel[c].cur_subframe == 0 || br.get_bits1() != 0 {
                if !self.channel[c].reuse_sf {
                    let step = br.get_bits(2) as i32 + 1;
                    self.channel[c].scale_factor_step = step;
                    let mut val = 45 / step;
                    for b in 0..self.num_bands {
                        val += st.sf_vlc.decode(br)?;
                        self.channel[c].saved_scale_factors[dst_idx][b] = val;
                    }
                } else {
                    let mut b = 0usize;
                    while b < self.num_bands {
                        let idx = st.sf_rl_vlc.decode(br)? as usize;
                        let (skip, val, sign_bit) = if idx == 0 {
                            let code = br.get_bits(14);
                            let val = (code >> 6) as i32;
                            let sign_bit = code & 1;
                            let skip = ((code & 0x3f) >> 1) as usize;
                            (skip, val, sign_bit)
                        } else if idx == 1 {
                            break;
                        } else {
                            let skip = SCALE_RL_RUN[idx] as usize;
                            let val = SCALE_RL_LEVEL[idx] as i32;
                            let sign_bit = br.get_bits1();
                            (skip, val, sign_bit)
                        };
                        b += skip;
                        if b >= self.num_bands {
                            return Err("invalid scale factor RL index");
                        }
                        let delta = if sign_bit != 0 { val } else { -val };
                        self.channel[c].saved_scale_factors[dst_idx][b] += delta;
                        b += 1;
                    }
                }
                self.channel[c].scale_factor_idx = dst_idx;
                self.channel[c].table_idx = self.table_idx;
                self.channel[c].reuse_sf = true;
            }

            let sfs = &self.channel[c].saved_scale_factors[dst_idx][..self.num_bands];
            self.channel[c].max_scale_factor = sfs.iter().copied().max().unwrap_or(0);
        }
        Ok(())
    }

    fn decode_coeffs(&mut self, br: &mut BitReader<'_>, c: usize, st: &StaticTables) -> Result<(), &'static str> {
        let vlctable = br.get_bits1() as usize;
        let vlc = &st.coef_vlc[vlctable];
        let (run, level): (&[u16], &[f32]) = if vlctable == 1 {
            (&COEF1_RUN, &COEF1_LEVEL)
        } else {
            (&COEF0_RUN, &COEF0_LEVEL)
        };

        let ci = &mut self.channel[c];
        let base = ci.coeffs_offset;
        let coeffs = &mut ci.out[base..base + self.subframe_len];

        let mut rl_mode = false;
        let mut cur_coeff = 0usize;
        let mut num_zeros = 0usize;

        while (self.transmit_num_vec_coeffs || !rl_mode) && cur_coeff + 3 < ci.num_vec_coeffs {
            let mut vals = [0.0f32; 4];
            let idx = st.vec4_vlc.decode(br)?;
            if idx < 0 {
                for i in (0..4).step_by(2) {
                    let idx2 = st.vec2_vlc.decode(br)?;
                    if idx2 < 0 {
                        let mut v0 = st.vec1_vlc.decode(br)? as u32;
                        if v0 as usize == HUFF_VEC1_SIZE - 1 {
                            v0 = v0.wrapping_add(get_large_val(br));
                        }
                        let mut v1 = st.vec1_vlc.decode(br)? as u32;
                        if v1 as usize == HUFF_VEC1_SIZE - 1 {
                            v1 = v1.wrapping_add(get_large_val(br));
                        }
                        vals[i] = v0 as f32;
                        vals[i + 1] = v1 as f32;
                    } else {
                        let u = idx2 as u32;
                        vals[i] = (u >> 4) as f32;
                        vals[i + 1] = (u & 0xF) as f32;
                    }
                }
            } else {
                let u = idx as u32;
                vals[0] = (u >> 12) as f32;
                vals[1] = ((u >> 8) & 0xF) as f32;
                vals[2] = ((u >> 4) & 0xF) as f32;
                vals[3] = (u & 0xF) as f32;
            }

            for v in vals {
                if v != 0.0 {
                    let sign_bit = br.get_bits1();
                    coeffs[cur_coeff] = if sign_bit != 0 { v } else { -v };
                    num_zeros = 0;
                } else {
                    coeffs[cur_coeff] = 0.0;
                    num_zeros += 1;
                    if num_zeros > (self.subframe_len >> 8) {
                        rl_mode = true;
                    }
                }
                cur_coeff += 1;
            }
        }

        if cur_coeff < self.subframe_len {
            coeffs[cur_coeff..self.subframe_len].fill(0.0);
            run_level_decode(br, vlc, level, run, coeffs, cur_coeff, self.subframe_len, self.esc_len)?;
        }
        Ok(())
    }

    fn inverse_channel_transform(&mut self) {
        let sfb = &self.sfb_offsets[self.table_idx];
        for g in 0..self.num_chgroups {
            if !self.chgroup[g].transform {
                continue;
            }
            let num_ch = self.chgroup[g].num_channels;
            debug_assert_eq!(num_ch, 2);
            let c0 = self.chgroup[g].channel_indices[0];
            let c1 = self.chgroup[g].channel_indices[1];
            let off0 = self.channel[c0].coeffs_offset;
            let off1 = self.channel[c1].coeffs_offset;
            let mat = self.chgroup[g].decorrelation_matrix;

            for b in 0..self.num_bands {
                let start = sfb[b];
                let end = sfb[b + 1].min(self.subframe_len);
                if self.chgroup[g].transform_band[b] {
                    for y in start..end {
                        let d0 = self.channel[c0].out[off0 + y];
                        let d1 = self.channel[c1].out[off1 + y];
                        self.channel[c0].out[off0 + y] = d0 * mat[0] + d1 * mat[1];
                        self.channel[c1].out[off1 + y] = d0 * mat[2] + d1 * mat[3];
                    }
                } else if self.nb_channels == 2 {
                    const SCALE_UNCOUPLED: f32 = 181.0 / 128.0;
                    for y in start..end {
                        self.channel[c0].out[off0 + y] *= SCALE_UNCOUPLED;
                        self.channel[c1].out[off1 + y] *= SCALE_UNCOUPLED;
                    }
                }
            }
        }
    }

    fn wmapro_window(&mut self, st: &StaticTables) {
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let mut winlen = self.channel[c].prev_block_len;
            let mut start = self.channel[c].coeffs_offset - (winlen >> 1);
            if self.subframe_len < winlen {
                start += (winlen - self.subframe_len) >> 1;
                winlen = self.subframe_len;
            }
            let win_idx = winlen.trailing_zeros() as usize - BLOCK_MIN_BITS;
            let win = &st.windows[win_idx];
            let half = winlen >> 1;
            vector_fmul_window(&mut self.channel[c].out[start..start + winlen], win, half);
            self.channel[c].prev_block_len = self.subframe_len;
        }
    }

    fn decode_subframe(&mut self, br: &mut BitReader<'_>, st: &StaticTables) -> Result<(), &'static str> {
        let mut offset = SAMPLES_PER_FRAME;
        let mut subframe_len = SAMPLES_PER_FRAME;
        let mut total_samples = SAMPLES_PER_FRAME * self.nb_channels;
        let mut transmit_coeffs = false;

        for c in 0..self.nb_channels {
            self.channel[c].grouped = false;
            if offset > self.channel[c].decoded_samples {
                offset = self.channel[c].decoded_samples;
                subframe_len = self.channel[c].subframe_len[self.channel[c].cur_subframe];
            }
        }

        self.channels_for_cur_subframe = 0;
        for c in 0..self.nb_channels {
            let cur_sf = self.channel[c].cur_subframe;
            total_samples -= self.channel[c].decoded_samples;
            if offset == self.channel[c].decoded_samples && subframe_len == self.channel[c].subframe_len[cur_sf] {
                total_samples -= self.channel[c].subframe_len[cur_sf];
                self.channel[c].decoded_samples += self.channel[c].subframe_len[cur_sf];
                self.channel_indexes_for_cur_subframe[self.channels_for_cur_subframe] = c;
                self.channels_for_cur_subframe += 1;
            }
        }

        if total_samples == 0 {
            self.parsed_all_subframes = true;
        }

        self.table_idx = (SAMPLES_PER_FRAME / subframe_len).trailing_zeros() as usize;
        self.num_bands = self.num_sfb[self.table_idx];

        let out_offset = offset + (SAMPLES_PER_FRAME >> 1);
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            self.channel[c].coeffs_offset = out_offset;
        }

        self.subframe_len = subframe_len;
        self.esc_len = (usize::BITS - 1 - (subframe_len - 1).leading_zeros()) as usize + 1;

        // Extended header (fill bits).
        if br.get_bits1() != 0 {
            let mut num_fill_bits = br.get_bits(2) as usize;
            if num_fill_bits == 0 {
                let len = br.get_bits(4) as usize;
                num_fill_bits = br.get_bits(len) as usize + 1;
            }
            if br.bits_count() + num_fill_bits > self.num_saved_bits {
                return Err("invalid fill bits");
            }
            br.skip_bits(num_fill_bits);
        }

        // Reserved bit (must be 0).
        if br.get_bits1() != 0 {
            return Err("reserved subframe bit set");
        }

        self.decode_channel_transform(br)?;

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let tc = br.get_bits1() != 0;
            self.channel[c].transmit_coefs = tc;
            if tc {
                transmit_coeffs = true;
            }
        }

        if transmit_coeffs {
            let mut quant_step = (90 * self.bits_per_sample as i32) >> 4;

            self.transmit_num_vec_coeffs = br.get_bits1() != 0;
            if self.transmit_num_vec_coeffs {
                let num_bits = (usize::BITS - 1 - ((self.subframe_len + 3) / 4).leading_zeros()) as usize + 1;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    let nvc = (br.get_bits(num_bits) as usize) << 2;
                    if nvc > self.subframe_len {
                        return Err("num_vec_coeffs exceeds subframe length");
                    }
                    self.channel[c].num_vec_coeffs = nvc;
                }
            } else {
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].num_vec_coeffs = self.subframe_len;
                }
            }

            let step = br.get_sbits(6);
            quant_step += step;
            if step == -32 || step == 31 {
                let positive = step == 31;
                let mut quant = 0i32;
                let mut last_step = 0i32;
                while br.bits_count() + 5 < self.num_saved_bits {
                    last_step = br.get_bits(5) as i32;
                    if last_step == 31 {
                        quant += 31;
                    } else {
                        break;
                    }
                }
                let total = quant + last_step;
                quant_step += if positive { total } else { -total };
            }

            if self.channels_for_cur_subframe == 1 {
                let c0 = self.channel_indexes_for_cur_subframe[0];
                self.channel[c0].quant_step = quant_step;
            } else {
                let modifier_len = br.get_bits(3) as usize;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].quant_step = quant_step;
                    if br.get_bits1() != 0 {
                        if modifier_len > 0 {
                            self.channel[c].quant_step += br.get_bits(modifier_len) as i32 + 1;
                        } else {
                            self.channel[c].quant_step += 1;
                        }
                    }
                }
            }

            self.decode_scale_factors(br, st)?;
        }

        // Parse spectral coefficients for each channel in this subframe.
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].transmit_coefs && br.bits_count() < self.num_saved_bits {
                let _ = self.decode_coeffs(br, c, st);
            } else {
                let base = self.channel[c].coeffs_offset;
                self.channel[c].out[base..base + subframe_len].fill(0.0);
            }
        }

        if transmit_coeffs {
            let imdct_idx = subframe_len.trailing_zeros() as usize - BLOCK_MIN_BITS;
            let imdct = &st.imdct[imdct_idx];
            self.inverse_channel_transform();

            for i in 0..self.channels_for_cur_subframe {
                let c = self.channel_indexes_for_cur_subframe[i];
                let sf_buf = self.channel[c].cur_sf_idx;
                let base = self.channel[c].coeffs_offset;
                let max_sf = self.channel[c].max_scale_factor;
                let sf_step = self.channel[c].scale_factor_step;
                let q_step = self.channel[c].quant_step;

                for b in 0..self.num_bands {
                    let start = self.sfb_offsets[self.table_idx][b];
                    let end = self.sfb_offsets[self.table_idx][b + 1].min(self.subframe_len);
                    let sf = self.channel[c].saved_scale_factors[sf_buf][b];
                    let exp = q_step - (max_sf - sf) * sf_step;
                    let quant = st.quant_gain(exp);
                    for k in start..end {
                        self.tmp[k] = self.channel[c].out[base + k] * quant;
                    }
                }

                imdct.imdct_half(&self.tmp[..subframe_len], &mut self.channel[c].out[base..base + subframe_len]);
            }
        }

        self.wmapro_window(st);

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].cur_subframe >= self.channel[c].num_subframes {
                return Err("subframe index overflow");
            }
            self.channel[c].cur_subframe += 1;
        }
        Ok(())
    }

    /// Decode one WMA/XMA frame from `self.reservoir`.
    /// Returns `(more_frames, got_frame)`.
    fn decode_frame(&mut self, out_frame: &mut [[f32; SAMPLES_PER_FRAME]; MAX_CHANNELS_PER_STREAM]) -> (bool, bool) {
        let st = static_tables();
        let data = self.reservoir.buf.clone();
        let mut br = BitReader::new(&data, self.num_saved_bits);
        br.skip_bits(self.frame_offset);

        let len = if self.len_prefix {
            br.get_bits(XMA_LOG2_FRAME_SIZE) as usize
        } else {
            0
        };

        if self.decode_tilehdr(&mut br).is_err() {
            self.packet_loss = true;
            return (false, false);
        }

        // Optional postproc transform (skipped in WMA Pro / XMA).
        if self.nb_channels > 1 && br.get_bits1() != 0 && br.get_bits1() != 0 {
            br.skip_bits(4 * self.nb_channels * self.nb_channels);
        }

        // Dynamic range compression gain (present when decode_flags & 0x80).
        if self.dynamic_range_compression {
            let _drc_gain = br.get_bits(8);
        }

        // Trim start / trim end (10 bits each for 512-sample frames: log2(512 * 2) = 10).
        if br.get_bits1() != 0 {
            let trim_bits = (SAMPLES_PER_FRAME * 2).trailing_zeros() as usize;
            if br.get_bits1() != 0 {
                self.trim_start = br.get_bits(trim_bits) as u16;
            }
            if br.get_bits1() != 0 {
                self.trim_end = br.get_bits(trim_bits) as u16;
            }
        } else {
            self.trim_start = 0;
            self.trim_end = 0;
        }

        self.parsed_all_subframes = false;
        for c in 0..self.nb_channels {
            self.channel[c].decoded_samples = 0;
            self.channel[c].cur_subframe = 0;
            self.channel[c].reuse_sf = false;
        }

        while !self.parsed_all_subframes {
            if self.decode_subframe(&mut br, st).is_err() {
                self.packet_loss = true;
                return (false, false);
            }
        }

        for c in 0..self.nb_channels {
            out_frame[c].copy_from_slice(&self.channel[c].out[..SAMPLES_PER_FRAME]);
            self.channel[c].out.copy_within(SAMPLES_PER_FRAME..SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2, 0);
        }

        let got_frame = if self.skip_frame {
            self.skip_frame = false;
            false
        } else {
            true
        };
        self.has_decoded = true;

        if self.len_prefix {
            let consumed = br.bits_count() - self.frame_offset;
            if len != consumed + 2 {
                self.packet_loss = true;
                return (false, got_frame);
            }
            br.skip_bits(1);
        } else {
            while br.bits_count() < self.num_saved_bits && br.get_bits1() == 0 {}
        }

        let more_frames = br.get_bits1() != 0;
        (more_frames, got_frame)
    }

    /// Run one step of FFmpeg's `decode_packet` on `pkt_slice`, returning
    /// `Ok((bytes_consumed, got_frame))` or `Err(())` on packet loss.
    fn decode_packet_step(
        &mut self,
        pkt_slice: &[u8],
        out_frame: &mut [[f32; SAMPLES_PER_FRAME]; MAX_CHANNELS_PER_STREAM],
    ) -> Result<(usize, bool), ()> {
        let mut got_frame = false;

        if pkt_slice.is_empty() {
            // EOF flush: emit final overlap-add half-frame padded with zeros.
            self.packet_done = false;
            if self.eof_done {
                return Ok((0, false));
            }
            for c in 0..self.nb_channels {
                out_frame[c].fill(0.0);
                out_frame[c][..SAMPLES_PER_FRAME / 2].copy_from_slice(&self.channel[c].out[..SAMPLES_PER_FRAME / 2]);
            }
            self.eof_done = true;
            self.packet_done = true;
            return Ok((0, true));
        }

        let mut pgb;
        let buf_bit_size: usize;

        if self.packet_done || self.packet_loss {
            self.packet_done = false;

            let use_size = pkt_slice.len().min(PACKET_SIZE);
            self.next_packet_start = pkt_slice.len() - use_size;
            buf_bit_size = use_size * 8;
            pgb = BitReader::new(&pkt_slice[..use_size], buf_bit_size);

            match self.version {
                Version::Xma1 => {
                    let _seq = pgb.get_bits(4);
                    pgb.skip_bits(2);
                }
                Version::Xma2 => {
                    let _num_frames = pgb.get_bits(6);
                }
            }
            let mut num_bits_prev_frame = pgb.get_bits(XMA_LOG2_FRAME_SIZE) as usize;
            pgb.skip_bits(3);
            self.skip_packets = pgb.get_bits(8) as u8;

            if num_bits_prev_frame > 0 {
                let remaining_packet_bits = buf_bit_size.saturating_sub(pgb.bits_count());
                if num_bits_prev_frame >= remaining_packet_bits {
                    num_bits_prev_frame = remaining_packet_bits;
                    self.packet_done = true;
                }
                self.save_bits(&mut pgb, num_bits_prev_frame, true);
                if !self.packet_loss {
                    let (_, gf) = self.decode_frame(out_frame);
                    got_frame = gf;
                }
            }

            if self.packet_loss {
                self.num_saved_bits = 0;
                self.packet_loss = false;
            }
        } else {
            if pkt_slice.len() < self.next_packet_start {
                self.packet_loss = true;
                return Err(());
            }
            let use_size = pkt_slice.len() - self.next_packet_start;
            buf_bit_size = use_size * 8;
            pgb = BitReader::new(&pkt_slice[..use_size], buf_bit_size);
            pgb.skip_bits(self.packet_offset);

            let rem = buf_bit_size as isize - pgb.bits_count() as isize;
            if self.len_prefix && rem > XMA_LOG2_FRAME_SIZE as isize {
                let frame_size = pgb.show_bits(XMA_LOG2_FRAME_SIZE) as usize;
                if frame_size > 0 && frame_size as isize <= rem {
                    self.save_bits(&mut pgb, frame_size, false);
                    if !self.packet_loss {
                        let (more, gf) = self.decode_frame(out_frame);
                        got_frame = gf;
                        self.packet_done = !more;
                    }
                } else {
                    self.packet_done = true;
                }
            } else {
                self.packet_done = true;
            }
        }

        let rem = buf_bit_size as isize - pgb.bits_count() as isize;
        if rem < 0 {
            self.packet_loss = true;
        }
        if self.packet_done && !self.packet_loss && rem > 0 {
            self.save_bits(&mut pgb, rem as usize, false);
        }

        self.packet_offset = pgb.bits_count() & 7;
        if self.packet_loss {
            return Err(());
        }
        Ok((pgb.bits_count() >> 3, got_frame))
    }
}

// ---------------------------------------------------------------------------------------
// Multi-stream XMA1 / XMA2 Decoder
// ---------------------------------------------------------------------------------------

/// Stateful multi-stream XMA1/XMA2 decoder (1–8 streams, 1–16 channels).
pub struct Decoder {
    cfg: StreamConfig,
    streams: Vec<StreamDecoder>,
    fifos: Vec<[VecDeque<f32>; MAX_CHANNELS_PER_STREAM]>,
    current_stream: usize,
    initial_skip: usize,
    trim_start: u16,
    trim_end: u16,
    flushed: bool,
}

impl Decoder {
    pub fn new(cfg: StreamConfig) -> Result<Self, String> {
        if cfg.stream_channels.is_empty() || cfg.stream_channels.len() > MAX_STREAMS {
            return Err(format!("invalid stream count {}", cfg.stream_channels.len()));
        }
        let sum: usize = cfg.stream_channels.iter().map(|&c| c as usize).sum();
        if sum == 0 || sum != cfg.channels as usize {
            return Err(format!("channel sum {sum} != {}", cfg.channels));
        }
        // Ensure static tables are initialized once.
        let _ = static_tables();

        let streams = cfg
            .stream_channels
            .iter()
            .map(|&ch| StreamDecoder::new(cfg.version, ch as usize, cfg.sample_rate))
            .collect();
        let fifos = (0..cfg.stream_channels.len())
            .map(|_| [VecDeque::with_capacity(4096), VecDeque::with_capacity(4096)])
            .collect();

        Ok(Self {
            cfg,
            streams,
            fifos,
            current_stream: 0,
            initial_skip: INITIAL_SKIP_SAMPLES,
            trim_start: 0,
            trim_end: 0,
            flushed: false,
        })
    }

    pub fn channels(&self) -> u16 {
        self.cfg.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    /// Reset all stream states and FIFOs (called on `FlushSourceBuffers` or after `END_OF_STREAM`).
    pub fn flush(&mut self) {
        for s in &mut self.streams {
            s.flush();
        }
        for f in &mut self.fifos {
            f[0].clear();
            f[1].clear();
        }
        self.current_stream = 0;
        self.initial_skip = INITIAL_SKIP_SAMPLES;
        self.trim_start = 0;
        self.trim_end = 0;
        self.flushed = false;
    }

    /// Feed a single 2048-byte XMA packet to the active stream and advance stream selection.
    fn feed_packet(&mut self, pkt: &[u8]) -> Result<(), String> {
        // Explicit XMA2 empty/padding packet sentinel (unused tail of a multi-packet block).
        if pkt.len() >= 4 && pkt[0..4] == [0xFF, 0xFF, 0xFF, 0xFF] {
            return Ok(());
        }
        let mut frame_buf = [[0.0f32; SAMPLES_PER_FRAME]; MAX_CHANNELS_PER_STREAM];
        let mut pos = 0usize;

        // Mirror FFmpeg's decode_simple_internal loop: keep calling decode_packet_step on the
        // remaining slice of this 2048-byte packet until packet_done or packet_loss.
        loop {
            let cur = self.current_stream;
            let step = if !self.streams[cur].eof_done {
                self.streams[cur].decode_packet_step(&pkt[pos..], &mut frame_buf)
            } else {
                Ok((pkt.len() - pos, false))
            };

            if self.streams[0].trim_start != 0 {
                self.trim_start = self.streams[0].trim_start;
            }
            if self.streams[0].trim_end != 0 {
                self.trim_end = self.streams[0].trim_end;
            }

            match step {
                Ok((consumed, got_frame)) => {
                    if got_frame {
                        let nb_ch = self.streams[cur].nb_channels;
                        self.fifos[cur][0].extend(&frame_buf[0]);
                        if nb_ch > 1 {
                            self.fifos[cur][1].extend(&frame_buf[1]);
                        }
                    }

                    let done = self.streams[cur].packet_done || self.streams[cur].packet_loss;
                    if done {
                        self.advance_stream();
                        break;
                    }
                    if consumed == 0 && !got_frame {
                        // Guard against zero-progress loops on malformed input.
                        self.streams[cur].packet_done = true;
                        self.advance_stream();
                        break;
                    }
                    pos = (pos + consumed).min(pkt.len());
                    if pos >= pkt.len() {
                        self.streams[cur].packet_done = true;
                        self.advance_stream();
                        break;
                    }
                }
                Err(()) => {
                    self.advance_stream();
                    self.current_stream = 0;
                    return Err("XMA packet decode error".into());
                }
            }
        }
        Ok(())
    }

    fn advance_stream(&mut self) {
        let cur = self.current_stream;
        if self.streams[cur].skip_packets != 0 {
            let mut min_skip = self.streams[0].skip_packets;
            let mut best = 0usize;
            for i in 1..self.streams.len() {
                if self.streams[i].skip_packets < min_skip {
                    min_skip = self.streams[i].skip_packets;
                    best = i;
                }
            }
            self.current_stream = best;
        }
        for s in &mut self.streams {
            s.skip_packets = s.skip_packets.saturating_sub(1);
        }
    }

    /// Drain synchronized samples from all stream FIFOs into `out` (interleaved f32).
    fn drain_synchronized(&mut self, out: &mut Vec<f32>, trim_tail: usize) {
        let mut nb_samples = usize::MAX;
        for f in &self.fifos {
            nb_samples = nb_samples.min(f[0].len());
        }
        if nb_samples == usize::MAX || nb_samples == 0 {
            return;
        }
        let emit_total = nb_samples.saturating_sub(trim_tail);
        let skip = self.initial_skip.min(emit_total);
        self.initial_skip -= skip;
        let to_emit = emit_total - skip;

        // Discard initial encoder-delay samples from every stream.
        if skip > 0 {
            for i in 0..self.streams.len() {
                self.fifos[i][0].drain(..skip);
                if self.streams[i].nb_channels > 1 {
                    self.fifos[i][1].drain(..skip);
                }
            }
        }

        // Interleave `to_emit` frames across all streams/channels.
        out.reserve(to_emit * self.cfg.channels as usize);
        for _ in 0..to_emit {
            for i in 0..self.streams.len() {
                out.push(self.fifos[i][0].pop_front().unwrap_or(0.0));
                if self.streams[i].nb_channels > 1 {
                    out.push(self.fifos[i][1].pop_front().unwrap_or(0.0));
                }
            }
        }

        // Discard trimmed tail samples (only at EOF).
        if trim_tail > 0 {
            for i in 0..self.streams.len() {
                let n = trim_tail.min(self.fifos[i][0].len());
                self.fifos[i][0].drain(..n);
                if self.streams[i].nb_channels > 1 {
                    let n1 = trim_tail.min(self.fifos[i][1].len());
                    self.fifos[i][1].drain(..n1);
                }
            }
        }
    }

    /// Decode an XMA1/XMA2 buffer containing one or more 2048-byte packets into interleaved `f32`
    /// samples in `[-1.0, 1.0]`. When `end_of_stream` is true, flushes the final IMDCT overlap
    /// tail, applies bitstream `trim_end`, and resets the decoder for any subsequent stream.
    pub fn decode_packets_f32(&mut self, data: &[u8], end_of_stream: bool) -> Result<Vec<f32>, String> {
        if data.is_empty() && !end_of_stream {
            return Ok(Vec::new());
        }

        let mut packets_ok = 0usize;
        let mut last_err: Option<String> = None;

        for chunk in data.chunks(PACKET_SIZE) {
            if chunk.len() < 4 {
                break;
            }
            match self.feed_packet(chunk) {
                Ok(()) => packets_ok += 1,
                Err(e) => last_err = Some(e),
            }
        }

        let mut out = Vec::new();
        if end_of_stream {
            if !self.flushed {
                let mut frame_buf = [[0.0f32; SAMPLES_PER_FRAME]; MAX_CHANNELS_PER_STREAM];
                for i in 0..self.streams.len() {
                    if !self.streams[i].eof_done && self.streams[i].has_decoded {
                        if let Ok((_, true)) = self.streams[i].decode_packet_step(&[], &mut frame_buf) {
                            let nb_ch = self.streams[i].nb_channels;
                            self.fifos[i][0].extend(&frame_buf[0]);
                            if nb_ch > 1 {
                                self.fifos[i][1].extend(&frame_buf[1]);
                            }
                        }
                    }
                }
                let trim_raw = self.trim_end as i32 + self.trim_start as i32 - 128 - 64;
                let trim_tail = trim_raw.max(0) as usize;
                self.drain_synchronized(&mut out, trim_tail);
                self.flushed = true;
            }
            self.flush();
        } else {
            self.drain_synchronized(&mut out, 0);
        }

        if out.is_empty() && packets_ok == 0 {
            if let Some(e) = last_err {
                return Err(e);
            }
        }
        Ok(out)
    }

    /// Decode an XMA1/XMA2 buffer into interleaved PCM16 (`i16`) samples for XAudio2.
    pub fn decode_buffer_i16(&mut self, data: &[u8], end_of_stream: bool) -> Result<Vec<i16>, String> {
        let floats = self.decode_packets_f32(data, end_of_stream)?;
        let mut pcm = Vec::with_capacity(floats.len());
        for s in floats {
            let scaled = (s * 32768.0_f32).round().clamp(-32768.0, 32767.0) as i16;
            pcm.push(scaled);
        }
        Ok(pcm)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn quant_gains_match_power_at_and_beyond_both_limits() {
        let st = static_tables();
        for exp in (-2000..=2000).chain([i32::MIN, i32::MAX]) {
            let expected = 10.0_f64.powf(exp as f64 / 20.0) as f32;
            assert_eq!(st.quant_gain(exp).to_bits(), expected.to_bits(), "exponent {exp}");
        }
    }

    // FFmpeg-verified synthetic XMA2/XMA1 test packets (copyright-free spectral tones;
    // zero-padded to 2048-byte packets). Every vector's output was cross-checked against
    // FFmpeg's `xma2`/`xma1` decoders to < 1.5e-7 f32 (0-1 LSB i16).

    pub(crate) const MONO_48K_PREFIX: &[u8] = &[
        0x30, 0x00, 0x00, 0x00, 0x00, 0xe7, 0x00, 0x0c, 0x13, 0xe3, 0xff, 0xff, 0xfb, 0x39, 0x46, 0x44,
        0xf6, 0x55, 0xa0, 0x1c, 0xe0, 0x01, 0x82, 0x7c, 0x7f, 0xff, 0xff, 0x67, 0x28, 0xc8, 0x9d, 0xca,
        0xb4, 0x03, 0x9c, 0x00, 0x30, 0x4f, 0x8f, 0xff, 0xff, 0xec, 0xe5, 0x19, 0x13, 0xd9, 0x56, 0x80,
        0x73, 0x80, 0x06, 0x09, 0xf1, 0xff, 0xff, 0xfd, 0x9c, 0xa3, 0x22, 0x77, 0x2a, 0xd0, 0x0e, 0x70,
        0x00, 0xc1, 0x3e, 0x3f, 0xff, 0xff, 0xb3, 0x94, 0x64, 0x4f, 0x65, 0x5a, 0x01, 0xce, 0x00, 0x18,
        0x27, 0xc7, 0xff, 0xff, 0xf6, 0x72, 0x8c, 0x89, 0xdc, 0xab, 0x40, 0x39, 0xc0, 0x03, 0x04, 0xf8,
        0xff, 0xff, 0xfe, 0xce, 0x51, 0x91, 0x3d, 0x95, 0x68, 0x07, 0x38, 0x00, 0x60, 0x9f, 0x1f, 0xff,
        0xff, 0xd9, 0xca, 0x32, 0x27, 0x72, 0xad, 0x00, 0xe7, 0x00, 0x0c, 0x13, 0xe3, 0xff, 0xff, 0xfb,
        0x39, 0x46, 0x44, 0xf6, 0x55, 0xa0, 0x1c, 0xe0, 0x01, 0x82, 0x7c, 0x7f, 0xff, 0xff, 0x67, 0x28,
        0xc8, 0x9d, 0xca, 0xb4, 0x03, 0x9c, 0x00, 0x30, 0x4f, 0x8f, 0xff, 0xff, 0xec, 0xe5, 0x19, 0x13,
        0xd9, 0x56, 0x80, 0x73, 0x80, 0x06, 0x09, 0xf1, 0xff, 0xff, 0xfd, 0x9c, 0xa3, 0x22, 0x77, 0x2a,
        0xc0,
    ];

    const STEREO_MS_PREFIX: &[u8] = &[
        0x20, 0x00, 0x00, 0x00, 0x01, 0xf5, 0x00, 0x01, 0xe0, 0x60, 0x7e, 0x16, 0xff, 0xff, 0xff, 0x7f,
        0xff, 0xff, 0xb3, 0x95, 0x4b, 0x56, 0x94, 0x06, 0xc0, 0x4a, 0xbe, 0x72, 0xa9, 0x6a, 0xfc, 0x40,
        0x6c, 0x04, 0x7f, 0x40, 0x7d, 0x40, 0x00, 0x78, 0x18, 0x1f, 0x85, 0xbf, 0xff, 0xff, 0xdf, 0xff,
        0xff, 0xec, 0xe5, 0x52, 0xd5, 0xa5, 0x01, 0xb0, 0x12, 0xaf, 0x9c, 0xaa, 0x5a, 0xbf, 0x10, 0x1b,
        0x01, 0x1f, 0xd0, 0x1f, 0x50, 0x00, 0x1e, 0x06, 0x07, 0xe1, 0x6f, 0xff, 0xff, 0xf7, 0xff, 0xff,
        0xfb, 0x39, 0x54, 0xb5, 0x69, 0x40, 0x6c, 0x04, 0xab, 0xe7, 0x2a, 0x96, 0xaf, 0xc4, 0x06, 0xc0,
        0x47, 0xf4, 0x07, 0xd4, 0x00, 0x07, 0x81, 0x81, 0xf8, 0x5b, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xfe,
        0xce, 0x55, 0x2d, 0x5a, 0x50, 0x1b, 0x01, 0x2a, 0xf9, 0xca, 0xa5, 0xab, 0xf1, 0x01, 0xb0, 0x11,
        0xfd, 0x02, 0x0f, 0x00, 0x00, 0xaa, 0xaa, 0xab, 0xc0, 0xc0, 0xfc, 0x2d, 0xff, 0xff, 0xfe, 0xff,
        0xff, 0xff, 0x67, 0x2b, 0x85, 0x5a, 0x50, 0x17, 0x06, 0x2a, 0xf9, 0xcc, 0x7b, 0xe2, 0x01, 0xe0,
        0xc3, 0xfa, 0x04, 0x1e, 0x00, 0x01, 0x55, 0x55, 0x57, 0x81, 0x81, 0xf8, 0x5b, 0xff, 0xff, 0xfd,
        0xff, 0xff, 0xfe, 0xce, 0x57, 0x0a, 0xb4, 0xa0, 0x2e, 0x0c, 0x55, 0xf3, 0x98, 0xf7, 0xc4, 0x03,
        0xc1, 0x87, 0xf4, 0x08, 0x3c, 0x00, 0x02, 0xaa, 0xaa, 0xaf, 0x03, 0x03, 0xf0, 0xb7, 0xff, 0xff,
        0xfb, 0xff, 0xff, 0xfd, 0x9c, 0xae, 0x15, 0x69, 0x40, 0x5c, 0x18, 0xab, 0xe7, 0x31, 0xef, 0x88,
        0x07, 0x83, 0x0f, 0xe8, 0x10, 0x78, 0x00, 0x05, 0x55, 0x55, 0x5e, 0x06, 0x07, 0xe1, 0x6f, 0xff,
        0xff, 0xf7, 0xff, 0xff, 0xfb, 0x39, 0x5c, 0x2a, 0xd2, 0x80, 0xb8, 0x31, 0x57, 0xce, 0x63, 0xdf,
        0x10, 0x0f, 0x06, 0x1f, 0xc0,
    ];

    const VAR_SUBFRAMES_PREFIX: &[u8] = &[
        0x18, 0x00, 0x00, 0x00, 0x04, 0x46, 0xaf, 0xb8, 0x00, 0x18, 0x2f, 0x0a, 0x89, 0x44, 0xa2, 0x51,
        0x28, 0x94, 0x4c, 0x87, 0x35, 0x69, 0x40, 0x5c, 0x0c, 0x56, 0x30, 0xbc, 0x12, 0x94, 0xa5, 0x29,
        0x4d, 0x55, 0x0d, 0xe7, 0x45, 0x3f, 0xc6, 0x17, 0x9e, 0x04, 0x10, 0xc0, 0x14, 0xd9, 0x45, 0x55,
        0x2b, 0x18, 0x5e, 0x61, 0x81, 0x81, 0x3c, 0xaa, 0xcb, 0x7f, 0xc6, 0x0f, 0xc3, 0x39, 0x55, 0x1c,
        0x2e, 0x26, 0xab, 0x18, 0x3e, 0x12, 0x66, 0xd5, 0xa0, 0x88, 0xd5, 0xf7, 0x00, 0x03, 0x05, 0xe1,
        0x51, 0x28, 0x94, 0x4a, 0x25, 0x12, 0x89, 0x90, 0xe6, 0xad, 0x28, 0x0b, 0x81, 0x8a, 0xc6, 0x17,
        0x82, 0x52, 0x94, 0xa5, 0x29, 0xaa, 0xa1, 0xbc, 0xe8, 0xa7, 0xf8, 0xc2, 0xf3, 0xc0, 0x82, 0x18,
        0x02, 0x9b, 0x28, 0xaa, 0xa5, 0x63, 0x0b, 0xcc, 0x30, 0x30, 0x27, 0x95, 0x59, 0x6f, 0xf8, 0xc1,
        0xf8, 0x67, 0x2a, 0xa3, 0x85, 0xc4, 0xd5, 0x63, 0x07, 0xc2, 0x4c, 0xda, 0xb4, 0x11, 0x1a, 0xbe,
        0xe0, 0x00, 0x60, 0xbc, 0x2a, 0x25, 0x12, 0x89, 0x44, 0xa2, 0x51, 0x32, 0x1c, 0xd5, 0xa5, 0x01,
        0x70, 0x31, 0x58, 0xc2, 0xf0, 0x4a, 0x52, 0x94, 0xa5, 0x35, 0x54, 0x37, 0x9d, 0x14, 0xff, 0x18,
        0x5e, 0x78, 0x10, 0x43, 0x00, 0x53, 0x65, 0x15, 0x54, 0xac, 0x61, 0x79, 0x86, 0x06, 0x04, 0xf2,
        0xab, 0x2d, 0xff, 0x18, 0x3f, 0x0c, 0xe5, 0x54, 0x70, 0xb8, 0x9a, 0xac, 0x60, 0xf8, 0x49, 0x9b,
        0x56, 0x82, 0x23, 0x57, 0xdc, 0x00, 0x0c, 0x17, 0x85, 0x44, 0xa2, 0x51, 0x28, 0x94, 0x4a, 0x26,
        0x43, 0x9a, 0xb4, 0xa0, 0x2e, 0x06, 0x2b, 0x18, 0x5e, 0x09, 0x4a, 0x52, 0x94, 0xa6, 0xaa, 0x86,
        0xf3, 0xa2, 0x9f, 0xe3, 0x0b, 0xcf, 0x02, 0x08, 0x60, 0x0a, 0x6c, 0xa2, 0xaa, 0x95, 0x8c, 0x2f,
        0x30, 0xc0, 0xc0, 0x9e, 0x55, 0x65, 0xbf, 0xe3, 0x07, 0xe1, 0x9c, 0xaa, 0x8e, 0x17, 0x13, 0x55,
        0x8c, 0x1f, 0x09, 0x33, 0x6a, 0xd0, 0x44, 0x6a, 0xfb, 0x80, 0x01, 0x82, 0xf0, 0xa8, 0x94, 0x4a,
        0x25, 0x12, 0x89, 0x44, 0xc8, 0x73, 0x56, 0x94, 0x05, 0xc0, 0xc5, 0x63, 0x0b, 0xc1, 0x29, 0x4a,
        0x52, 0x94, 0xd5, 0x50, 0xde, 0x74, 0x53, 0xfc, 0x61, 0x79, 0xe0, 0x41, 0x0c, 0x01, 0x4d, 0x94,
        0x55, 0x52, 0xb1, 0x85, 0xe6, 0x18, 0x18, 0x13, 0xca, 0xac, 0xb7, 0xfc, 0x60, 0xfc, 0x33, 0x95,
        0x51, 0xc2, 0xe2, 0x6a, 0xb1, 0x83, 0xe1, 0x26, 0x6d, 0x5a, 0x08, 0x8d, 0x5f, 0x70, 0x00, 0x30,
        0x5e, 0x15, 0x12, 0x89, 0x44, 0xa2, 0x51, 0x28, 0x99, 0x0e, 0x6a, 0xd2, 0x80, 0xb8, 0x18, 0xac,
        0x61, 0x78, 0x25, 0x29, 0x4a, 0x52, 0x9a, 0xaa, 0x1b, 0xce, 0x8a, 0x7f, 0x8c, 0x2f, 0x3c, 0x08,
        0x21, 0x80, 0x29, 0xb2, 0x8a, 0xaa, 0x56, 0x30, 0xbc, 0xc3, 0x03, 0x02, 0x79, 0x55, 0x96, 0xff,
        0x8c, 0x1f, 0x86, 0x72, 0xaa, 0x38, 0x5c, 0x4d, 0x56, 0x30, 0x7c, 0x24, 0xcd, 0xab,
    ];

    const ALL_COEFF_PATHS_PREFIX: &[u8] = &[
        0x18, 0x00, 0x00, 0x00, 0x02, 0xb3, 0x00, 0x0c, 0x0e, 0x01, 0xdf, 0xff, 0xff, 0xe1, 0xb8, 0x64,
        0xd4, 0x28, 0xd8, 0x29, 0x66, 0x91, 0x0c, 0xd5, 0x13, 0x46, 0x89, 0xa3, 0x49, 0x50, 0x0e, 0x15,
        0xd2, 0x84, 0x22, 0x54, 0x02, 0x31, 0x34, 0xa5, 0xf6, 0x17, 0x92, 0x80, 0xb8, 0xf7, 0x56, 0x81,
        0x20, 0x80, 0x10, 0xf0, 0x05, 0x8d, 0xff, 0xff, 0xff, 0x7d, 0x74, 0x6c, 0x14, 0xb3, 0x49, 0xbb,
        0x9f, 0x14, 0x03, 0x85, 0x7c, 0x50, 0xbc, 0x50, 0x08, 0xc4, 0xfc, 0x4b, 0xec, 0x2f, 0x78, 0x80,
        0xb8, 0xf7, 0xfe, 0x81, 0x22, 0x80, 0x18, 0x60, 0x70, 0x0e, 0xff, 0xff, 0xff, 0x3e, 0xba, 0x36,
        0x0a, 0x59, 0xa4, 0xdd, 0xca, 0x54, 0x03, 0x85, 0x74, 0xa1, 0x08, 0x95, 0x00, 0x8c, 0x4d, 0x29,
        0x7d, 0x85, 0xe4, 0xa0, 0x2e, 0x3d, 0xd5, 0xa0, 0x44, 0x20, 0x01, 0x63, 0x7f, 0xff, 0xff, 0xdf,
        0x5d, 0x1b, 0x05, 0x2c, 0xd2, 0x6e, 0xe7, 0xc5, 0x00, 0xe1, 0x5f, 0x14, 0x2f, 0x14, 0x02, 0x31,
        0x3f, 0x12, 0xfb, 0x0b, 0xde, 0x20, 0x2e, 0x3d, 0xff, 0xa0, 0x47, 0xa0, 0x01, 0x81, 0xc0, 0x3b,
        0xff, 0xff, 0xfc, 0xfa, 0xe8, 0xd8, 0x29, 0x66, 0x93, 0x77, 0x29, 0x50, 0x0e, 0x15, 0xd2, 0x84,
        0x22, 0x54, 0x02, 0x31, 0x34, 0xa5, 0xf6, 0x17, 0x92, 0x80, 0xb8, 0xf7, 0x56, 0x81, 0x10, 0x80,
        0x05, 0x8d, 0xff, 0xff, 0xff, 0x7d, 0x74, 0x6c, 0x14, 0xb3, 0x49, 0xbb, 0x9f, 0x14, 0x03, 0x85,
        0x7c, 0x50, 0xbc, 0x50, 0x08, 0xc4, 0xfc, 0x4b, 0xec, 0x2f, 0x78, 0x80, 0xb8, 0xf7, 0xfe,
    ];

    const XMA1_STEREO_PREFIX: &[u8] = &[
        0x00, 0x00, 0x00, 0x00, 0x01, 0xdf, 0x00, 0x02, 0xe0, 0xc0, 0xde, 0x00, 0xff, 0xff, 0xfe, 0x7f,
        0xff, 0xff, 0x67, 0x20, 0x8d, 0x56, 0xab, 0x16, 0xce, 0x55, 0xb3, 0x9c, 0x90, 0xfa, 0xac, 0xfb,
        0x3d, 0x5a, 0x03, 0xbe, 0x00, 0x05, 0xc1, 0x81, 0xbc, 0x01, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xfe,
        0xce, 0x41, 0x1a, 0xad, 0x56, 0x2d, 0x9c, 0xab, 0x67, 0x39, 0x21, 0xf5, 0x59, 0xf6, 0x7a, 0xb4,
        0x07, 0x7c, 0x00, 0x0b, 0x83, 0x03, 0x78, 0x03, 0xff, 0xff, 0xf9, 0xff, 0xff, 0xfd, 0x9c, 0x82,
        0x35, 0x5a, 0xac, 0x5b, 0x39, 0x56, 0xce, 0x72, 0x43, 0xea, 0xb3, 0xec, 0xf5, 0x68, 0x0e, 0xf8,
        0x00, 0x17, 0x06, 0x06, 0xf0, 0x07, 0xff, 0xff, 0xf3, 0xff, 0xff, 0xfb, 0x39, 0x04, 0x6a, 0xb5,
        0x58, 0xb6, 0x72, 0xad, 0x9c, 0xe4, 0x87, 0xd5, 0x67, 0xd9, 0xea, 0xd0, 0x1d, 0xf0, 0x00, 0x2e,
        0x0c, 0x0d, 0xe0, 0x0f, 0xff, 0xff, 0xe7, 0xff, 0xff, 0xf6, 0x72, 0x08, 0xd5, 0x6a, 0xb1, 0x6c,
        0xe5, 0x5b, 0x39, 0xc9, 0x0f, 0xaa, 0xcf, 0xb3, 0xd5, 0xa0, 0x3b, 0xe0, 0x00, 0x5c, 0x18, 0x1b,
        0xc0, 0x1f, 0xff, 0xff, 0xcf, 0xff, 0xff, 0xec, 0xe4, 0x11, 0xaa, 0xd5, 0x62, 0xd9, 0xca, 0xb6,
        0x73, 0x92, 0x1f, 0x55, 0x9f, 0x67, 0xab, 0x40, 0x77, 0xc0, 0x00, 0xb8, 0x30, 0x37, 0x80, 0x3f,
        0xff, 0xff, 0x9f, 0xff, 0xff, 0xd9, 0xc8, 0x23, 0x55, 0xaa, 0xc5, 0xb3, 0x95, 0x6c, 0xe7, 0x24,
        0x3e, 0xab, 0x3e, 0xcf, 0x56, 0x80, 0xef, 0x80, 0x01, 0x70, 0x60, 0x6f, 0x00, 0x7f, 0xff, 0xff,
        0x3f, 0xff, 0xff, 0xb3, 0x90, 0x46, 0xab, 0x55, 0x8b, 0x67, 0x2a, 0xd9, 0xce, 0x48, 0x7d, 0x56,
        0x7d, 0x9e, 0xac,
    ];

    const CROSS_PACKET_SPARSE: &[&[(usize, &[u8])]] = &[
        &[
            (0, &[0x08, 0x00, 0x00, 0x00, 0x7f, 0x9d, 0x00, 0x65, 0x02, 0x77, 0xe6, 0x60]),
            (2034, &[0xc1, 0x3d, 0x3f, 0xff, 0xff, 0xb3, 0x94, 0xb3, 0x57, 0x26, 0x72, 0xb4, 0x03, 0xd4]),
        ],
        &[(0, &[
            0x10, 0x03, 0x40, 0x00, 0x00, 0x30, 0x4f, 0x4f, 0xff, 0xff, 0xec, 0xe5, 0x2c, 0xd5, 0x49, 0x9e,
            0xad, 0x00, 0xf5, 0x00, 0x0c, 0x13, 0xd3, 0xff, 0xff, 0xfb, 0x39, 0x4b, 0x35, 0x72, 0x67, 0x2b,
            0x40, 0x3d, 0x40, 0x03, 0x04, 0xf4, 0xff, 0xff, 0xfe, 0xce, 0x52, 0xcd, 0x54, 0x99, 0xea, 0xd0,
            0x0f, 0x50, 0x00, 0xc1, 0x3d, 0x3f, 0xff, 0xff, 0xb3, 0x94, 0xb3, 0x57, 0x26, 0x72, 0xb4, 0x04,
            0x34, 0x01, 0x4b, 0x43, 0x04, 0xf4, 0xff, 0xff, 0xfe, 0xce, 0x52, 0xcd, 0x54, 0x99, 0xea, 0xc0,
        ])],
    ];

    const SURROUND_5_1_SPARSE: &[&[(usize, &[u8])]] = &[
        &[
            (0, &[0x04, 0x00, 0x00, 0x02, 0x80, 0x3f, 0x00, 0x13, 0xbf, 0x15]),
            (2029, &[
                0xb8, 0x48, 0x4f, 0x80, 0x3f, 0xff, 0xff, 0x9f, 0xff, 0xff, 0xd9, 0x55, 0x96, 0xf3, 0x9c, 0xe7,
                0x39, 0xca, 0xb6,
            ]),
        ],
        &[
            (0, &[0x04, 0x00, 0x00, 0x02, 0x80, 0x51, 0x00, 0x13, 0xbf, 0x15]),
            (2029, &[
                0xb8, 0x48, 0x4f, 0x80, 0x3f, 0xff, 0xff, 0x9f, 0xff, 0xff, 0xd9, 0xce, 0x72, 0x2b, 0x35, 0x57,
                0x9c, 0xe7, 0x2a,
            ]),
        ],
        &[
            (0, &[0x04, 0x00, 0x00, 0x02, 0x80, 0x5b, 0x00, 0x13, 0xbf, 0x15]),
            (2029, &[
                0xb8, 0x48, 0x4f, 0x80, 0x3f, 0xff, 0xff, 0x9f, 0xff, 0xff, 0xd9, 0xce, 0x73, 0x9c, 0xaa, 0x9e,
                0xcd, 0x73, 0x95,
            ]),
        ],
        &[(0, &[
            0x0c, 0x01, 0xf8, 0x02, 0x72, 0xab, 0x14, 0xce, 0x73, 0x9c, 0xe5, 0x5a, 0x03, 0xd6, 0x00, 0x05,
            0xc2, 0x42, 0x7c, 0x01, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xfe, 0xca, 0xac, 0xb7, 0x9c, 0xe7, 0x39,
            0xce, 0x55, 0xb3, 0x95, 0x58, 0xa6, 0x73, 0x9c, 0xe7, 0x2a, 0xd0, 0x1e, 0xb0, 0x00, 0x2e, 0x12,
            0x13, 0xe0, 0x0f, 0xff, 0xff, 0xe7, 0xff, 0xff, 0xf6, 0x55, 0x65, 0xbc, 0xe7, 0x39, 0xce, 0x72,
            0xad, 0x9c, 0xaa, 0xc5, 0x33, 0x9c, 0xe7, 0x39, 0x56, 0x80, 0xf5, 0x80, 0x01, 0x70, 0x90, 0x9f,
            0x00, 0x7f, 0xff, 0xff, 0x3f, 0xff, 0xff, 0xb2, 0xab, 0x2d, 0xe7, 0x39, 0xce, 0x73, 0x95, 0x6c,
            0xe5, 0x56, 0x29, 0x9c, 0xe7, 0x39, 0xca, 0xb0,
        ])],
        &[(0, &[
            0x0c, 0x02, 0x40, 0x02, 0xd9, 0xce, 0x73, 0x92, 0xd7, 0xaa, 0xce, 0x72, 0xad, 0x01, 0xfd, 0x00,
            0x02, 0xe1, 0x21, 0x3e, 0x00, 0xff, 0xff, 0xfe, 0x7f, 0xff, 0xff, 0x67, 0x39, 0xc8, 0xac, 0xd5,
            0x5e, 0x73, 0x9c, 0xab, 0x67, 0x39, 0xce, 0x4b, 0x5e, 0xab, 0x39, 0xca, 0xb4, 0x07, 0xf4, 0x00,
            0x0b, 0x84, 0x84, 0xf8, 0x03, 0xff, 0xff, 0xf9, 0xff, 0xff, 0xfd, 0x9c, 0xe7, 0x22, 0xb3, 0x55,
            0x79, 0xce, 0x72, 0xad, 0x9c, 0xe7, 0x39, 0x2d, 0x7a, 0xac, 0xe7, 0x2a, 0xd0, 0x1f, 0xd0, 0x00,
            0x2e, 0x12, 0x13, 0xe0, 0x0f, 0xff, 0xff, 0xe7, 0xff, 0xff, 0xf6, 0x73, 0x9c, 0x8a, 0xcd, 0x55,
            0xe7, 0x39, 0xca, 0xb6, 0x73, 0x9c, 0xe4, 0xb5, 0xea, 0xb3, 0x9c, 0xab,
        ])],
        &[(0, &[
            0x0c, 0x02, 0x68, 0x02, 0x6c, 0xe7, 0x39, 0xce, 0x72, 0xaa, 0x78, 0x69, 0x95, 0x68, 0x10, 0x38,
            0x00, 0x17, 0x09, 0x09, 0xf0, 0x07, 0xff, 0xff, 0xf3, 0xff, 0xff, 0xfb, 0x39, 0xce, 0x73, 0x95,
            0x53, 0xd9, 0xae, 0x72, 0xad, 0x9c, 0xe7, 0x39, 0xce, 0x55, 0x4f, 0x0d, 0x32, 0xad, 0x02, 0x07,
            0x00, 0x02, 0xe1, 0x21, 0x3e, 0x00, 0xff, 0xff, 0xfe, 0x7f, 0xff, 0xff, 0x67, 0x39, 0xce, 0x72,
            0xaa, 0x7b, 0x35, 0xce, 0x55, 0xb3, 0x9c, 0xe7, 0x39, 0xca, 0xa9, 0xe1, 0xa6, 0x55, 0xa0, 0x40,
            0xe0, 0x00, 0x5c, 0x24, 0x27, 0xc0, 0x1f, 0xff, 0xff, 0xcf, 0xff, 0xff, 0xec, 0xe7, 0x39, 0xce,
            0x55, 0x4f, 0x66, 0xb9, 0xca, 0xb6, 0x73, 0x9c, 0xe7, 0x39, 0x55, 0x3c, 0x34, 0xca, 0xb0,
        ])],
    ];

    pub(crate) fn pad_packet(prefix: &[u8]) -> Vec<u8> {
        let mut p = vec![0u8; PACKET_SIZE];
        p[..prefix.len()].copy_from_slice(prefix);
        p
    }

    fn expand_sparse(pkts: &[&[(usize, &[u8])]]) -> Vec<u8> {
        let mut out = vec![0u8; pkts.len() * PACKET_SIZE];
        for (pi, runs) in pkts.iter().enumerate() {
            let base = pi * PACKET_SIZE;
            for &(off, data) in *runs {
                out[base + off..base + off + data.len()].copy_from_slice(data);
            }
        }
        out
    }

    /// Direct O(N^2) reference of FFmpeg `ff_tx_mdct_naive_inv` (tx_template.c:1169-1197).
    fn naive_imdct_half(src: &[f32], bits_per_sample: u8) -> Vec<f32> {
        let len2 = src.len();
        let len = len2 / 2;
        let scale = 1.0_f64 / (len as f64) / ((1u64 << (bits_per_sample - 1)) as f64);
        let phase = PI / (4.0 * len2 as f64);
        let mut dst = vec![0.0f32; len2];
        for i in 0..len {
            let mut sum_d = 0.0_f64;
            let mut sum_u = 0.0_f64;
            let i_d = phase * (4 * len - 2 * i - 1) as f64;
            let i_u = phase * (3 * len2 + 2 * i + 1) as f64;
            for (j, &s) in src.iter().enumerate() {
                let a = (2 * j + 1) as f64;
                sum_d += (a * i_d).cos() * s as f64;
                sum_u += (a * i_u).cos() * s as f64;
            }
            dst[i] = (sum_d * scale) as f32;
            dst[i + len] = (-sum_u * scale) as f32;
        }
        dst
    }

    fn assert_anchors(name: &str, got: &[f32], anchors: &[(usize, f32)]) {
        for &(idx, exp) in anchors {
            let diff = (got[idx] - exp).abs();
            assert!(diff < 1e-5, "{name}[{idx}]: got {}, expected {exp} (diff {diff:.2e})", got[idx]);
        }
    }

    #[test]
    fn fast_imdct_matches_naive_reference_for_all_xma_sizes() {
        for bits in BLOCK_MIN_BITS..=BLOCK_MAX_BITS {
            let n = 1usize << bits;
            let plan = ImdctPlan::new(n, 16);
            let src: Vec<f32> = (0..n).map(|k| ((k as f32 * 0.73).sin() * 12345.0) - 500.0).collect();
            let expected = naive_imdct_half(&src, 16);
            let mut actual = vec![0.0f32; n];
            plan.imdct_half(&src, &mut actual);

            let max_err = expected
                .iter()
                .zip(actual.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-5, "IMDCT size {n}: max_err={max_err}");
        }
    }

    #[test]
    fn all_huffman_tables_build_and_roundtrip_every_symbol() {
        let st = static_tables();

        fn check_pairs(tbl: &VlcTable, pairs: &[(u8, u8)], offset: i32, name: &str) {
            let mut code: u64 = 0;
            for &(sym, len) in pairs {
                let bytes = (code as u32).to_be_bytes();
                let mut br = BitReader::new(&bytes, 32);
                let got = tbl.decode(&mut br).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(got, sym as i32 + offset, "{name}: symbol mismatch");
                assert_eq!(br.bits_count(), len as usize, "{name}: bit length mismatch");
                code += 1u64 << (32 - len as u32);
            }
            assert_eq!(code, 1u64 << 32, "{name}: incomplete Kraft sum");
        }

        fn check_split(tbl: &VlcTable, lens: &[u8], syms: &[u16], offset: i32, name: &str) {
            let mut code: u64 = 0;
            for (&len, &sym) in lens.iter().zip(syms.iter()) {
                let bytes = (code as u32).to_be_bytes();
                let mut br = BitReader::new(&bytes, 32);
                let got = tbl.decode(&mut br).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(got, sym as i32 + offset, "{name}: symbol mismatch");
                assert_eq!(br.bits_count(), len as usize, "{name}: bit length mismatch");
                code += 1u64 << (32 - len as u32);
            }
            assert_eq!(code, 1u64 << 32, "{name}: incomplete Kraft sum");
        }

        check_pairs(&st.sf_vlc, &SCALE_TABLE, -60, "sf_vlc");
        check_pairs(&st.sf_rl_vlc, &SCALE_RL_TABLE, 0, "sf_rl_vlc");
        check_split(&st.coef_vlc[0], &COEF0_LENS, &COEF0_SYMS, 0, "coef_vlc[0]");
        check_pairs(&st.coef_vlc[1], &COEF1_TABLE, 0, "coef_vlc[1]");
        check_split(&st.vec4_vlc, &VEC4_LENS, &VEC4_SYMS, -1, "vec4_vlc");
        check_pairs(&st.vec2_vlc, &VEC2_TABLE, -1, "vec2_vlc");
        check_pairs(&st.vec1_vlc, &VEC1_TABLE, 0, "vec1_vlc");
    }

    #[test]
    fn stream_config_parses_xma2_and_xma1_layouts() {
        let mut extra = [0u8; 34];
        extra[0..2].copy_from_slice(&3u16.to_le_bytes());
        let cfg = StreamConfig::from_xma2_wfx(6, 48_000, &extra).unwrap();
        assert_eq!(cfg.stream_channels, vec![2, 2, 2]);

        let cfg5 = StreamConfig::from_xma2_wfx(5, 44_100, &extra).unwrap();
        assert_eq!(cfg5.stream_channels, vec![2, 2, 1]);

        let mut legacy = vec![0u8; 48];
        legacy[0] = 4;
        legacy[1] = 2;
        legacy[40] = 2;
        legacy[44] = 1;
        let cfgl = StreamConfig::from_xma2_wfx(3, 48_000, &legacy).unwrap();
        assert_eq!(cfgl.stream_channels, vec![2, 1]);

        let mut xma1 = vec![0u8; 48];
        xma1[4] = 2;
        xma1[8 + 17] = 2;
        xma1[8 + 20 + 17] = 2;
        let cfg1 = StreamConfig::from_xma1_wfx(4, 48_000, &xma1).unwrap();
        assert_eq!(cfg1.version, Version::Xma1);
        assert_eq!(cfg1.stream_channels, vec![2, 2]);

        assert!(StreamConfig::from_xma2_wfx(0, 48_000, &extra).is_err());
        assert!(StreamConfig::from_xma2_wfx(17, 48_000, &extra).is_err());
    }

    #[test]
    fn decodes_xma2_and_xma1_bitstreams_matching_ffmpeg() {
        // 1. Mono 48 kHz tone (12 frames -> 6080 samples) + trailing 0xFF padding packet + flush reuse.
        let mut pkt_and_pad = pad_packet(MONO_48K_PREFIX);
        pkt_and_pad.extend(std::iter::repeat_n(0xFFu8, PACKET_SIZE));
        let mut dec = Decoder::new(StreamConfig {
            version: Version::Xma2,
            sample_rate: 48_000,
            channels: 1,
            stream_channels: vec![1],
        })
        .unwrap();
        let f = dec.decode_packets_f32(&pkt_and_pad, true).unwrap();
        assert_eq!(f.len(), 6080);
        assert_anchors("mono_tone", &f, &[(0, 0.0019242), (1216, 0.0179213), (2432, -0.0370130), (3648, -0.0005879), (4864, 0.0385190)]);
        // Reuse after automatic EOS reset must produce identical samples.
        let f2 = dec.decode_packets_f32(&pad_packet(MONO_48K_PREFIX), true).unwrap();
        assert_eq!(f, f2);

        // 2. Stereo 48 kHz M/S channel transform (all-band & per-band) + quant_mod.
        let mut dec = Decoder::new(StreamConfig {
            version: Version::Xma2,
            sample_rate: 48_000,
            channels: 2,
            stream_channels: vec![2],
        })
        .unwrap();
        let f = dec.decode_packets_f32(&pad_packet(STEREO_MS_PREFIX), true).unwrap();
        assert_eq!(f.len(), 8064);
        assert_anchors("stereo_ms", &f, &[(0, 0.0189070), (1612, 0.0204287), (3225, 0.0522392), (4838, 0.0230093), (6451, 0.0085754)]);

        // 3. Variable 256/128 non-uniform subframes + scale factor resampling + RL delta coding (44.1 kHz).
        let mut dec = Decoder::new(StreamConfig {
            version: Version::Xma2,
            sample_rate: 44_100,
            channels: 2,
            stream_channels: vec![2],
        })
        .unwrap();
        let f = dec.decode_packets_f32(&pad_packet(VAR_SUBFRAMES_PREFIX), true).unwrap();
        assert_eq!(f.len(), 6016);
        assert_anchors("var_subframes", &f, &[(0, -0.0113599), (1203, -0.0181313), (2406, 0.0444796), (3609, -0.0388255), (4812, 0.0266683)]);

        // 4. All coefficient coding paths: vec4/vec2/vec1/large_val, coef_vlc 0&1, RLE escapes, quant escape, fill bits (32 kHz).
        let mut dec = Decoder::new(StreamConfig {
            version: Version::Xma2,
            sample_rate: 32_000,
            channels: 1,
            stream_channels: vec![1],
        })
        .unwrap();
        let f = dec.decode_packets_f32(&pad_packet(ALL_COEFF_PATHS_PREFIX), true).unwrap();
        assert_eq!(f.len(), 3008);
        assert_anchors("all_coeff_paths", &f, &[(0, 0.0239880), (601, 0.1342473), (1203, -0.4809944), (1804, 0.0886067), (2406, -0.2185475)]);

        // 5. XMA1 stereo 24 kHz.
        let mut dec = Decoder::new(StreamConfig {
            version: Version::Xma1,
            sample_rate: 24_000,
            channels: 2,
            stream_channels: vec![2],
        })
        .unwrap();
        let f = dec.decode_packets_f32(&pad_packet(XMA1_STEREO_PREFIX), true).unwrap();
        assert_eq!(f.len(), 8064);
        assert_anchors("xma1_stereo", &f, &[(0, -0.0285331), (1612, -0.0096696), (3225, 0.0124989), (4838, -0.0173741), (6451, 0.0275135)]);
    }

    #[test]
    fn cross_packet_frames_and_streaming_chunks_match_single_shot() {
        let data = expand_sparse(CROSS_PACKET_SPARSE);
        let cfg = StreamConfig {
            version: Version::Xma2,
            sample_rate: 48_000,
            channels: 1,
            stream_channels: vec![1],
        };

        // Single-shot EOS decode matches FFmpeg anchors & exact trimmed length (trim_start=160, trim_end=180).
        let mut dec = Decoder::new(cfg.clone()).unwrap();
        let full = dec.decode_packets_f32(&data, true).unwrap();
        assert_eq!(full.len(), 2860);
        assert_anchors("cross_packet", &full, &[(0, -0.0370088), (572, -0.0135333), (1144, -0.0329356), (1716, -0.0299850), (2288, -0.0138485)]);

        // Streaming chunk-by-chunk (packet 0 with eos=false, packet 1 with eos=true) produces identical samples!
        let mut stream_dec = Decoder::new(cfg).unwrap();
        let mut streamed = stream_dec.decode_packets_f32(&data[..PACKET_SIZE], false).unwrap();
        streamed.extend(stream_dec.decode_packets_f32(&data[PACKET_SIZE..], true).unwrap());
        assert_eq!(streamed, full);
    }

    #[test]
    fn surround_5_1_interleaves_three_streams_and_isolates_channels() {
        let data = expand_sparse(SURROUND_5_1_SPARSE);
        let mut dec = Decoder::new(StreamConfig {
            version: Version::Xma2,
            sample_rate: 48_000,
            channels: 6,
            stream_channels: vec![2, 2, 2],
        })
        .unwrap();
        let f = dec.decode_packets_f32(&data, true).unwrap();
        // 4 frames * 512 - 64 initial skip = 1984 frames/ch * 6 ch = 11904 samples.
        assert_eq!(f.len(), 11904);
        assert_anchors("surround_5_1", &f, &[(0, 0.0003502), (1766, -0.0376447), (3532, 0.0533849), (5299, 0.0099038), (7065, -0.0204287)]);

        // Verify all 6 channels carry distinct non-zero energy (stream 2 > stream 1 > stream 0 by design).
        let ch_rms: Vec<f32> = (0..6)
            .map(|ch| {
                let sum: f32 = f.iter().skip(ch).step_by(6).map(|&x| x * x).sum();
                (sum / (f.len() / 6) as f32).sqrt()
            })
            .collect();
        for (ch, &r) in ch_rms.iter().enumerate() {
            assert!(r > 0.005, "channel {ch} unexpectedly silent (rms={r})");
        }
        assert!(ch_rms[4] > ch_rms[2] && ch_rms[2] > ch_rms[0], "stream scaling mismatch: {ch_rms:?}");
    }

    #[test]
    fn corrupt_and_empty_packets_never_panic() {
        let mut extra = [0u8; 34];
        extra[0..2].copy_from_slice(&1u16.to_le_bytes());
        let cfg = StreamConfig::from_xma2_wfx(1, 48_000, &extra).unwrap();
        let mut dec = Decoder::new(cfg).unwrap();

        assert!(dec.decode_buffer_i16(&[], false).unwrap().is_empty());
        let zeros = vec![0u8; PACKET_SIZE];
        assert!(dec.decode_buffer_i16(&zeros, true).unwrap().is_empty());
        let ones = vec![0xFFu8; PACKET_SIZE];
        assert!(dec.decode_buffer_i16(&ones, true).unwrap().is_empty());
        let garbage = vec![0x5Au8; PACKET_SIZE];
        let _ = dec.decode_buffer_i16(&garbage, true);
    }
}
