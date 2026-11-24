//! RFC 6716 Opus packet framing and real decoding via Windows' `MSOpusDecoder.dll`
//! (Media Foundation `IMFTransform`).
//!
//! `mfplat.dll` and `MSOpusDecoder.dll` are loaded dynamically at run time (like DXC and
//! `xaudio2_9.dll`), so XCL has no static dependency on Media Foundation and falls back
//! cleanly if a Windows installation lacks the Media Feature Pack.
//!
//! Windows' `MSOpusDecoder.dll` decodes single-stream Opus (1 or 2 channels). Multistream
//! Opus (3–8 channels, RFC 7845 mapping family 1 / 255) is decoded by splitting each
//! packet's RFC 6716 Appendix B self-delimited sub-packets across `stream_count` MFT
//! instances (`coupled_count` stereo + uncoupled mono) and interleaving via the channel map.

use std::borrow::Cow;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::OnceLock;

use windows::core::{GUID, HRESULT, Interface};
use windows::Win32::Media::MediaFoundation::{
    IMFMediaBuffer, IMFMediaType, IMFSample, IMFTransform, MFAudioFormat_Float, MFAudioFormat_Opus,
    MFAudioFormat_PCM, MFMediaType_Audio, MFT_MESSAGE_COMMAND_FLUSH,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM,
    MFT_OUTPUT_DATA_BUFFER, MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES,
    MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_NUM_CHANNELS,
    MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_MT_USER_DATA,
    MFSTARTUP_LITE, MF_VERSION,
};
use windows::Win32::System::Com::IClassFactory;

/// CLSID of Windows' built-in Microsoft Opus Audio Decoder MFT (`MSOpusDecoder.dll`).
const CLSID_MS_OPUS_DECODER: GUID = GUID::from_u128(0x63e17c10_2d43_4c42_8fe3_8d8b63e46a6a);

/// The game's AWC containers hold Opus as `[u16 LE len][packet]…` in blocks of this size.
pub const AWC_BLOCK: usize = 2048;

/// Opus maximum packet duration (RFC 6716 §3.2.5): 120 ms = 5760 samples at 48 kHz.
pub const MAX_PACKET_SAMPLES_48K: u32 = 5760;

// ---------------------------------------------------------------------------------------
// RFC 6716 TOC & packet parser (pure Rust)
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Silk,
    Hybrid,
    Celt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketInfo {
    pub config: u8,
    pub mode: Mode,
    pub bandwidth: Bandwidth,
    pub stereo: bool,
    pub code: u8,
    pub frames: u8,
    /// Duration of one frame in samples at 48 kHz (120, 240, 480, 960, 1920, or 2880).
    pub frame_samples_48k: u32,
    /// Total duration of the packet in samples at 48 kHz (`frames * frame_samples_48k`, <= 5760).
    pub total_samples_48k: u32,
}

/// RFC 6716 §3.1 Table 2: TOC configuration number (0..=31) -> (mode, bandwidth, frame samples @ 48 kHz).
pub const fn config_info(config: u8) -> (Mode, Bandwidth, u32) {
    match config & 0x1F {
        0 => (Mode::Silk, Bandwidth::Narrow, 480),       // 10 ms
        1 => (Mode::Silk, Bandwidth::Narrow, 960),       // 20 ms
        2 => (Mode::Silk, Bandwidth::Narrow, 1920),      // 40 ms
        3 => (Mode::Silk, Bandwidth::Narrow, 2880),      // 60 ms
        4 => (Mode::Silk, Bandwidth::Medium, 480),       // 10 ms
        5 => (Mode::Silk, Bandwidth::Medium, 960),       // 20 ms
        6 => (Mode::Silk, Bandwidth::Medium, 1920),      // 40 ms
        7 => (Mode::Silk, Bandwidth::Medium, 2880),      // 60 ms
        8 => (Mode::Silk, Bandwidth::Wide, 480),         // 10 ms
        9 => (Mode::Silk, Bandwidth::Wide, 960),         // 20 ms
        10 => (Mode::Silk, Bandwidth::Wide, 1920),       // 40 ms
        11 => (Mode::Silk, Bandwidth::Wide, 2880),       // 60 ms
        12 => (Mode::Hybrid, Bandwidth::SuperWide, 480), // 10 ms
        13 => (Mode::Hybrid, Bandwidth::SuperWide, 960), // 20 ms
        14 => (Mode::Hybrid, Bandwidth::Full, 480),      // 10 ms
        15 => (Mode::Hybrid, Bandwidth::Full, 960),      // 20 ms
        16 => (Mode::Celt, Bandwidth::Narrow, 120),      // 2.5 ms
        17 => (Mode::Celt, Bandwidth::Narrow, 240),      // 5 ms
        18 => (Mode::Celt, Bandwidth::Narrow, 480),      // 10 ms
        19 => (Mode::Celt, Bandwidth::Narrow, 960),      // 20 ms
        20 => (Mode::Celt, Bandwidth::Wide, 120),        // 2.5 ms
        21 => (Mode::Celt, Bandwidth::Wide, 240),        // 5 ms
        22 => (Mode::Celt, Bandwidth::Wide, 480),        // 10 ms
        23 => (Mode::Celt, Bandwidth::Wide, 960),        // 20 ms
        24 => (Mode::Celt, Bandwidth::SuperWide, 120),   // 2.5 ms
        25 => (Mode::Celt, Bandwidth::SuperWide, 240),   // 5 ms
        26 => (Mode::Celt, Bandwidth::SuperWide, 480),   // 10 ms
        27 => (Mode::Celt, Bandwidth::SuperWide, 960),   // 20 ms
        28 => (Mode::Celt, Bandwidth::Full, 120),        // 2.5 ms
        29 => (Mode::Celt, Bandwidth::Full, 240),        // 5 ms
        30 => (Mode::Celt, Bandwidth::Full, 480),        // 10 ms
        31 => (Mode::Celt, Bandwidth::Full, 960),        // 20 ms
        _ => unreachable!(),
    }
}

/// Convert samples at 48 kHz to 100-nanosecond Media Foundation duration units.
/// Exact for every valid Opus frame size (120 samples = 2.5 ms = 25,000 * 100 ns).
pub const fn duration_100ns(samples_48k: u32) -> i64 {
    (samples_48k as i64 * 10_000_000) / 48_000
}

/// Convert samples at 48 kHz to samples at `rate` Hz (Opus supports 8/12/16/24/48 kHz).
pub const fn samples_at_rate(samples_48k: u32, rate: u32) -> u32 {
    ((samples_48k as u64 * rate as u64) / 48_000) as u32
}

/// Read an RFC 6716 §3.2.1 frame length field (1 or 2 bytes).
fn read_frame_len(data: &[u8], pos: &mut usize) -> Option<usize> {
    let b0 = *data.get(*pos)? as usize;
    *pos += 1;
    if b0 < 252 {
        Some(b0)
    } else {
        let b1 = *data.get(*pos)? as usize;
        *pos += 1;
        Some(b0 + 4 * b1)
    }
}

/// Write an RFC 6716 §3.2.1 frame length field (1 or 2 bytes, max 1275).
fn write_frame_len(len: usize, out: &mut Vec<u8>) -> Option<()> {
    if len > 1275 {
        return None;
    }
    if len < 252 {
        out.push(len as u8);
    } else {
        let b1 = (len - 252) / 4;
        let b0 = len - 4 * b1; // 252..=255
        out.push(b0 as u8);
        out.push(b1 as u8);
    }
    Some(())
}

/// Parse and validate an RFC 6716 Opus packet.
///
/// When `multistream` is false (mono/stereo, mapping family 0), validates the exact single-stream
/// byte-length rules in RFC 6716 §3.2.2–§3.2.5. When `multistream` is true (3..=8 channels,
/// mapping family 1/255), streams `0..N-2` use self-delimited framing (Appendix B) so only the
/// leading TOC / frame-count / 120 ms rules are enforced here.
pub fn parse_packet(packet: &[u8], multistream: bool) -> Option<PacketInfo> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let stereo = (toc & 0x04) != 0;
    let code = toc & 0x03;
    let (mode, bandwidth, frame_samples_48k) = config_info(config);

    let frames: u8 = match code {
        0 => {
            if !multistream && packet.len() - 1 > 1275 {
                return None;
            }
            1
        }
        1 => {
            if !multistream {
                let payload = packet.len() - 1;
                if payload % 2 != 0 || payload / 2 > 1275 {
                    return None;
                }
            }
            2
        }
        2 => {
            if !multistream {
                let mut pos = 1;
                let len1 = read_frame_len(packet, &mut pos)?;
                if pos + len1 > packet.len() {
                    return None;
                }
                let len2 = packet.len() - pos - len1;
                if len2 > 1275 {
                    return None;
                }
            } else if packet.len() < 2 {
                return None;
            }
            2
        }
        3 => {
            let fcb = *packet.get(1)?;
            let vbr = (fcb & 0x80) != 0;
            let padded = (fcb & 0x40) != 0;
            let count = fcb & 0x3F;
            if count == 0 || count > 48 {
                return None;
            }
            if !multistream {
                let mut pos = 2usize;
                let mut pad_len = 0usize;
                if padded {
                    loop {
                        let b = *packet.get(pos)? as usize;
                        pos += 1;
                        if b == 255 {
                            pad_len = pad_len.checked_add(254)?;
                        } else {
                            pad_len = pad_len.checked_add(b)?;
                            break;
                        }
                    }
                }
                if pos + pad_len > packet.len() {
                    return None;
                }
                let end = packet.len() - pad_len;
                if vbr {
                    let mut sum = 0usize;
                    for _ in 0..(count - 1) {
                        let flen = read_frame_len(packet, &mut pos)?;
                        sum = sum.checked_add(flen)?;
                    }
                    if pos + sum > end {
                        return None;
                    }
                    let last = end - pos - sum;
                    if last > 1275 {
                        return None;
                    }
                } else {
                    let rem = end - pos;
                    let m = count as usize;
                    if rem % m != 0 || rem / m > 1275 {
                        return None;
                    }
                }
            }
            count
        }
        _ => unreachable!(),
    };

    let total_samples_48k = frame_samples_48k.checked_mul(frames as u32)?;
    if total_samples_48k > MAX_PACKET_SAMPLES_48K {
        return None;
    }

    Some(PacketInfo {
        config,
        mode,
        bandwidth,
        stereo,
        code,
        frames,
        frame_samples_48k,
        total_samples_48k,
    })
}

/// Parse one RFC 6716 Appendix B self-delimited Opus packet starting at `data[*pos..]` and
/// convert it into a standard (non-self-delimited) RFC 6716 Opus packet.
fn split_self_delimited(data: &[u8], pos: &mut usize) -> Option<Vec<u8>> {
    let toc = *data.get(*pos)?;
    *pos += 1;
    let code = toc & 0x03;
    let (_, _, frame_samples) = config_info(toc >> 3);

    match code {
        0 => {
            let len = read_frame_len(data, pos)?;
            let end = pos.checked_add(len)?;
            let payload = data.get(*pos..end)?;
            *pos = end;
            let mut out = Vec::with_capacity(1 + len);
            out.push(toc);
            out.extend_from_slice(payload);
            Some(out)
        }
        1 => {
            let len = read_frame_len(data, pos)?;
            let total = len.checked_mul(2)?;
            let end = pos.checked_add(total)?;
            let payload = data.get(*pos..end)?;
            *pos = end;
            let mut out = Vec::with_capacity(1 + total);
            out.push(toc);
            out.extend_from_slice(payload);
            Some(out)
        }
        2 => {
            let len1 = read_frame_len(data, pos)?;
            let len2 = read_frame_len(data, pos)?;
            let total = len1.checked_add(len2)?;
            let end = pos.checked_add(total)?;
            let payload = data.get(*pos..end)?;
            *pos = end;
            let mut out = Vec::with_capacity(3 + total);
            out.push(toc);
            write_frame_len(len1, &mut out)?;
            out.extend_from_slice(payload);
            Some(out)
        }
        3 => {
            let fcb = *data.get(*pos)?;
            *pos += 1;
            let vbr = (fcb & 0x80) != 0;
            let padded = (fcb & 0x40) != 0;
            let count = (fcb & 0x3F) as usize;
            if count == 0 || count > 48 || frame_samples.checked_mul(count as u32)? > MAX_PACKET_SAMPLES_48K {
                return None;
            }
            let pad_hdr_start = *pos;
            let mut pad_len = 0usize;
            if padded {
                loop {
                    let b = *data.get(*pos)? as usize;
                    *pos += 1;
                    if b == 255 {
                        pad_len = pad_len.checked_add(254)?;
                    } else {
                        pad_len = pad_len.checked_add(b)?;
                        break;
                    }
                }
            }
            let pad_hdr = &data[pad_hdr_start..*pos];
            let mut out = Vec::new();
            out.push(toc);
            out.push(fcb);
            out.extend_from_slice(pad_hdr);

            let total_payload = if vbr {
                let mut sum = 0usize;
                for i in 0..count {
                    let flen = read_frame_len(data, pos)?;
                    if i + 1 < count {
                        write_frame_len(flen, &mut out)?;
                    }
                    sum = sum.checked_add(flen)?;
                }
                sum
            } else {
                let flen = read_frame_len(data, pos)?;
                flen.checked_mul(count)?
            };
            let end = pos.checked_add(total_payload)?.checked_add(pad_len)?;
            let body = data.get(*pos..end)?;
            *pos = end;
            out.extend_from_slice(body);
            Some(out)
        }
        _ => unreachable!(),
    }
}

/// Split an RFC 7845 multistream packet into `stream_count` standard single-stream packets.
pub fn split_multistream_packet<'a>(packet: &'a [u8], stream_count: usize) -> Option<Vec<Cow<'a, [u8]>>> {
    if stream_count == 0 {
        return None;
    }
    if stream_count == 1 {
        parse_packet(packet, false)?;
        return Some(vec![Cow::Borrowed(packet)]);
    }
    let mut out = Vec::with_capacity(stream_count);
    let mut pos = 0usize;
    for _ in 0..(stream_count - 1) {
        let sub = split_self_delimited(packet, &mut pos)?;
        parse_packet(&sub, false)?;
        out.push(Cow::Owned(sub));
    }
    let last = packet.get(pos..)?;
    parse_packet(last, false)?;
    out.push(Cow::Borrowed(last));
    Some(out)
}

// ---------------------------------------------------------------------------------------
// RFC 7845 OpusHead builder & parser
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    pub channels: u8,
    pub pre_skip: u16,
    pub input_rate: u32,
    pub output_gain: i16,
    pub mapping_family: u8,
    pub stream_count: u8,
    pub coupled_count: u8,
    pub mapping: Vec<u8>,
    pub raw: Vec<u8>,
}

/// Standard Vorbis surround channel mappings for RFC 7845 mapping family 1 (3..=8 channels).
fn vorbis_mapping(channels: u8) -> Option<(u8, u8, &'static [u8])> {
    match channels {
        3 => Some((2, 1, &[0, 2, 1])),
        4 => Some((2, 2, &[0, 1, 2, 3])),
        5 => Some((3, 2, &[0, 4, 1, 2, 3])),
        6 => Some((4, 2, &[0, 4, 1, 2, 3, 5])),
        7 => Some((4, 3, &[0, 4, 1, 2, 3, 5, 6])),
        8 => Some((5, 3, &[0, 6, 1, 2, 3, 4, 5, 7])),
        _ => None,
    }
}

/// Build a canonical RFC 7845 `OpusHead` identification header.
pub fn build_opus_head(channels: u8, input_rate: u32, pre_skip: u16, output_gain: i16) -> Vec<u8> {
    let ch = channels.clamp(1, 8);
    let mut h = Vec::with_capacity(if ch <= 2 { 19 } else { 21 + ch as usize });
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(ch);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_rate.to_le_bytes());
    h.extend_from_slice(&output_gain.to_le_bytes());
    if let Some((streams, coupled, map)) = vorbis_mapping(ch) {
        h.push(1); // mapping family 1 (Vorbis order)
        h.push(streams);
        h.push(coupled);
        h.extend_from_slice(map);
    } else {
        h.push(0); // mapping family 0 (mono / stereo)
    }
    h
}

/// Parse an RFC 7845 `OpusHead` header if `data` starts with `"OpusHead"`.
pub fn parse_opus_head(data: &[u8]) -> Option<OpusHead> {
    if data.len() < 19 || &data[..8] != b"OpusHead" {
        return None;
    }
    if data[8] != 1 {
        return None;
    }
    let channels = data[9];
    if channels == 0 || channels > 8 {
        return None;
    }
    let pre_skip = u16::from_le_bytes([data[10], data[11]]);
    let input_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let output_gain = i16::from_le_bytes([data[16], data[17]]);
    let mapping_family = data[18];
    let (stream_count, coupled_count, mapping, expected_len) = if mapping_family == 0 {
        if channels > 2 {
            return None;
        }
        let coupled = if channels == 2 { 1 } else { 0 };
        let map = if channels == 2 { vec![0, 1] } else { vec![0] };
        (1u8, coupled, map, 19usize)
    } else {
        let len = 21 + channels as usize;
        if data.len() < len {
            return None;
        }
        let sc = data[19];
        let cc = data[20];
        if sc == 0 || cc > sc {
            return None;
        }
        let decoded_ch = sc as usize + cc as usize;
        let map = data[21..len].to_vec();
        if map.iter().any(|&idx| idx != 255 && idx as usize >= decoded_ch) {
            return None;
        }
        (sc, cc, map, len)
    };
    Some(OpusHead {
        channels,
        pre_skip,
        input_rate,
        output_gain,
        mapping_family,
        stream_count,
        coupled_count,
        mapping,
        raw: data[..expected_len].to_vec(),
    })
}

// ---------------------------------------------------------------------------------------
// Packet extraction: Ogg Opus, length-prefixed [u16 LE len][pkt]... (+ zero padding), raw
// ---------------------------------------------------------------------------------------

/// Try parsing one contiguous block of `[u16 LE len][opus_packet]...` with optional trailing zeros.
/// Returns `(packets, had_zero_padding)` on success.
fn try_length_prefixed_block<'a>(block: &'a [u8], multistream: bool) -> Option<(Vec<&'a [u8]>, bool)> {
    if block.len() < 3 {
        return None;
    }
    let mut packets = Vec::new();
    let mut pos = 0usize;
    let mut zero_padded = false;
    while pos < block.len() {
        if pos + 2 > block.len() {
            if block[pos..].iter().all(|&b| b == 0) && !packets.is_empty() {
                zero_padded = true;
                break;
            }
            return None;
        }
        let len = u16::from_le_bytes([block[pos], block[pos + 1]]) as usize;
        if len == 0 {
            if !packets.is_empty() && block[pos..].iter().all(|&b| b == 0) {
                zero_padded = true;
                break;
            }
            return None;
        }
        let start = pos + 2;
        let end = start.checked_add(len)?;
        if end > block.len() {
            return None;
        }
        let pkt = &block[start..end];
        parse_packet(pkt, multistream)?;
        packets.push(pkt);
        pos = end;
    }
    if packets.is_empty() {
        return None;
    }
    Some((packets, zero_padded))
}

struct OggTiming {
    head: OpusHead,
    decoded_48k: u64,
    end_48k: u64,
}

impl OggTiming {
    fn pcm_range(&self, rate: u32) -> std::ops::Range<u64> {
        let start_48k = u64::from(self.head.pre_skip).min(self.end_48k);
        frames_at_rate(start_48k, rate)..frames_at_rate(self.end_48k, rate)
    }

    fn trim_pcm(&self, pcm: &mut Vec<i16>, rate: u32, channels: usize) -> bool {
        // The MFT can defer output. Only slice PCM when the complete decoded timeline
        // is present, so a short result is never mistaken for tail padding.
        if pcm.len() % channels != 0
            || (pcm.len() / channels) as u64 != frames_at_rate(self.decoded_48k, rate)
        {
            return false;
        }
        let range = self.pcm_range(rate);
        pcm.truncate(range.end as usize * channels);
        pcm.drain(..range.start as usize * channels);
        true
    }
}

struct ExtractedPackets<'a> {
    packets: Vec<Cow<'a, [u8]>>,
    timing: Option<OggTiming>,
}

/// Extract packets with the existing lacing rules. Use timing only when the submitted
/// pages describe one complete Opus logical stream with an unbroken packet sequence.
fn extract_ogg_opus(data: &[u8], multistream: bool) -> Option<ExtractedPackets<'_>> {
    if data.len() < 27 || &data[..4] != b"OggS" {
        return None;
    }
    let serial = &data[14..18];
    let mut packets = Vec::new();
    let mut cur = Vec::new();
    let mut pos = 0usize;
    let mut sequence = 0u32;
    let mut contiguous = true;
    let mut head = None;
    let mut have_tags = false;
    let mut previous_granule = None::<u64>;
    let mut decoded_48k = 0u64;
    let mut timing = None;
    while pos < data.len() {
        let page = data.get(pos..pos + 27)?;
        if &page[..4] != b"OggS" {
            return None;
        }
        let flags = page[5];
        let bos = flags & 0x02 != 0;
        let eos = flags & 0x04 != 0;
        let granule = u64::from_le_bytes(page[6..14].try_into().unwrap());
        let laces = data.get(pos + 27..pos + 27 + page[26] as usize)?;
        let mut body = pos + 27 + laces.len();
        let mut page_samples = 0u64;
        contiguous &= page[4] == 0
            && flags & !0x07 == 0
            && bos == (pos == 0)
            && (flags & 0x01 != 0) == !cur.is_empty()
            && &page[14..18] == serial
            && &page[18..22] == sequence.to_le_bytes().as_slice();
        sequence = sequence.wrapping_add(1);

        for (i, &lace) in laces.iter().enumerate() {
            let next = body.checked_add(lace as usize)?;
            let slice = data.get(body..next)?;
            body = next;
            if lace == 255 {
                cur.extend_from_slice(slice);
                continue;
            }
            let packet = if cur.is_empty() {
                Cow::Borrowed(slice)
            } else {
                cur.extend_from_slice(slice);
                Cow::Owned(std::mem::take(&mut cur))
            };
            if packet.starts_with(b"OpusHead") {
                if pos == 0 && laces.len() == 1 && granule == 0 {
                    head = parse_opus_head(&packet);
                } else {
                    contiguous = false;
                }
            } else if packet.starts_with(b"OpusTags") {
                contiguous &= head.is_some()
                    && !have_tags
                    && packet.len() >= 16
                    && i + 1 == laces.len()
                    && granule == 0;
                have_tags = true;
            } else if packet.is_empty() {
                contiguous = false;
            } else {
                let info = parse_packet(&packet, multistream)?;
                contiguous &= have_tags;
                page_samples += u64::from(info.total_samples_48k);
                packets.push(packet);
            }
        }

        if contiguous
            && eos
            && body == data.len()
            && cur.is_empty()
            && page_samples > 0
            && granule != u64::MAX
        {
            // RFC 7845 section 4.4 counts only packets completed on the EOS page.
            // If this is also the first audio page, section 4.5 uses a zero origin for
            // a short granule, or infers an initial offset when the granule is larger.
            let keep = match previous_granule {
                Some(previous) => granule.checked_sub(previous).filter(|&n| n <= page_samples),
                None => Some(granule.min(page_samples)),
            };
            if let (Some(head), Some(keep)) = (head.take(), keep) {
                timing = Some(OggTiming {
                    head,
                    decoded_48k: decoded_48k + page_samples,
                    end_48k: decoded_48k + keep,
                });
            }
        }
        if !eos && page_samples > 0 {
            contiguous &= granule != u64::MAX
                && match previous_granule {
                    Some(previous) => previous.checked_add(page_samples) == Some(granule),
                    None => granule >= page_samples,
                };
            previous_granule = Some(granule);
            decoded_48k += page_samples;
        }
        contiguous &= !eos;
        pos = body;
    }
    Some(ExtractedPackets { packets, timing })
}

/// Extract RFC 6716 Opus packets from a submitted audio buffer.
///
/// Supports:
/// 1. Ogg Opus container (`"OggS"`), skipping `OpusHead`/`OpusTags`,
/// 2. Multi-block length-prefixed `[u16 LE len][packet]...` (+ zero padding) when `block_align >= 64`,
/// 3. Single-block length-prefixed `[u16 LE len][packet]...` (2+ packets or zero-padded block,
///    such as 2048 B console AWC blocks),
/// 4. A single raw RFC 6716 Opus packet,
/// 5. A single unpadded `[u16 LE len][packet]` if raw parsing rejected the buffer.
pub fn extract_packets<'a>(data: &'a [u8], block_align: u16, multistream: bool) -> Vec<Cow<'a, [u8]>> {
    extract_buffer(data, block_align, multistream).packets
}

fn extract_buffer(data: &[u8], block_align: u16, multistream: bool) -> ExtractedPackets<'_> {
    if data.starts_with(b"OggS") {
        if let Some(ogg) = extract_ogg_opus(data, multistream) {
            return ogg;
        }
    }
    ExtractedPackets {
        packets: extract_raw_packets(data, block_align, multistream),
        timing: None,
    }
}

fn extract_raw_packets(data: &[u8], block_align: u16, multistream: bool) -> Vec<Cow<'_, [u8]>> {
    if data.is_empty() {
        return Vec::new();
    }
    // Multiple fixed-size blocks (e.g. N * 2048 B AWC blocks); a format without a usable
    // nBlockAlign gets the AWC block size
    let ba = if block_align >= 64 { block_align as usize } else { AWC_BLOCK };
    if ba >= 64 && data.len() >= ba && data.len() % ba == 0 {
        let mut all = Vec::new();
        let mut ok = true;
        for chunk in data.chunks_exact(ba) {
            if chunk.iter().all(|&b| b == 0) {
                continue;
            }
            match try_length_prefixed_block(chunk, multistream) {
                Some((pkts, _)) => all.extend(pkts.into_iter().map(Cow::Borrowed)),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && !all.is_empty() {
            return all;
        }
    }
    // Single length-prefixed buffer with >= 2 packets or zero-padding tail
    if let Some((pkts, zero_padded)) = try_length_prefixed_block(data, multistream) {
        if pkts.len() >= 2 || zero_padded {
            return pkts.into_iter().map(Cow::Borrowed).collect();
        }
        // Single length-prefixed packet with no zero tail: prefer raw RFC 6716 packet unless raw is invalid.
        if parse_packet(data, multistream).is_some() {
            return vec![Cow::Borrowed(data)];
        }
        return pkts.into_iter().map(Cow::Borrowed).collect();
    }
    if parse_packet(data, multistream).is_some() {
        return vec![Cow::Borrowed(data)];
    }
    Vec::new()
}

/// Expected PCM frames (samples per channel at `rate` Hz), using Ogg timing when available.
/// Used as a fallback estimate if Media Foundation is unavailable.
pub fn estimate_buffer_frames(data: &[u8], block_align: u16, rate: u32, multistream: bool) -> Option<u64> {
    let extracted = extract_buffer(data, block_align, multistream);
    if let Some(timing) = extracted.timing {
        let range = timing.pcm_range(rate);
        return Some(range.end - range.start);
    }
    if extracted.packets.is_empty() {
        return None;
    }
    let mut samples_48k = 0u64;
    for p in &extracted.packets {
        let info = parse_packet(p, multistream)?;
        samples_48k += u64::from(info.total_samples_48k);
    }
    Some(frames_at_rate(samples_48k, rate))
}

fn frames_at_rate(samples_48k: u64, rate: u32) -> u64 {
    ((u128::from(samples_48k) * u128::from(rate)) / 48_000) as u64
}

// ---------------------------------------------------------------------------------------
// Dynamic loader for mfplat.dll + MSOpusDecoder.dll
// ---------------------------------------------------------------------------------------

type MfStartupFn = unsafe extern "system" fn(u32, u32) -> HRESULT;
type MfCreateMediaTypeFn = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;
type MfCreateSampleFn = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;
type MfCreateMemoryBufferFn = unsafe extern "system" fn(u32, *mut *mut c_void) -> HRESULT;
type DllGetClassObjectFn = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

struct MfRuntime {
    create_media_type: MfCreateMediaTypeFn,
    create_sample: MfCreateSampleFn,
    create_memory_buffer: MfCreateMemoryBufferFn,
    opus_get_class_object: DllGetClassObjectFn,
}

static MF: OnceLock<Result<MfRuntime, String>> = OnceLock::new();

fn mf_runtime() -> Result<&'static MfRuntime, String> {
    MF.get_or_init(|| unsafe {
        let load_fn = |module: usize, name: &str| -> Result<usize, String> {
            crate::resolver::system_proc_in(
                module,
                &crate::pe::ImportName::Name { hint: 0, name: name.into() },
            )
            .ok_or_else(|| format!("missing export {name}"))
        };

        let mfplat = crate::resolver::system_module("mfplat.dll")
            .ok_or_else(|| "mfplat.dll not available".to_string())?;
        let startup: MfStartupFn = std::mem::transmute(load_fn(mfplat, "MFStartup")?);
        let create_media_type: MfCreateMediaTypeFn =
            std::mem::transmute(load_fn(mfplat, "MFCreateMediaType")?);
        let create_sample: MfCreateSampleFn =
            std::mem::transmute(load_fn(mfplat, "MFCreateSample")?);
        let create_memory_buffer: MfCreateMemoryBufferFn =
            std::mem::transmute(load_fn(mfplat, "MFCreateMemoryBuffer")?);

        let opus_dll = crate::resolver::system_module("MSOpusDecoder.dll")
            .ok_or_else(|| "MSOpusDecoder.dll not available".to_string())?;
        let opus_get_class_object: DllGetClassObjectFn =
            std::mem::transmute(load_fn(opus_dll, "DllGetClassObject")?);

        let hr = startup(MF_VERSION, MFSTARTUP_LITE);
        if hr.is_err() {
            return Err(format!("MFStartup failed: {:#x}", hr.0 as u32));
        }

        Ok(MfRuntime {
            create_media_type,
            create_sample,
            create_memory_buffer,
            opus_get_class_object,
        })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Whether an Opus decoder is available: Windows' built-in `MSOpusDecoder.dll` + Media Foundation
/// runtime, or libopus loaded from `opus.dll` (see [`libopus`]).
pub fn is_available() -> bool {
    mf_runtime().is_ok() || libopus().is_ok()
}

/// The decoder in use ("Windows MSOpusDecoder", "libopus (path)"), or why there is none. Logged once.
pub fn backend() -> String {
    match (mf_runtime(), libopus()) {
        (Ok(_), _) => "Windows' MSOpusDecoder (Media Foundation)".to_string(),
        (_, Ok(l)) => format!("libopus ({})", l.path),
        (Err(a), Err(b)) => format!("none ({a}; {b})"),
    }
}

// ---------------------------------------------------------------------------------------
// libopus from `opus.dll`, loaded at run time (where Windows' decoder does not exist: Wine)
// ---------------------------------------------------------------------------------------
//
// Not distributed with XCL (like DXC): `opus.dll` – libopus (BSD-3-Clause), built for Windows – is
// looked for in `%XCL_OPUS_DIR%`, then next to xcl.exe (`opus.dll`, `libopus.dll`, `libopus-0.dll`).
// Windows' own decoder is used first when the system has it. Only the decoder is used.

type OpusDecoderCreateFn = unsafe extern "C" fn(i32, i32, *mut i32) -> *mut c_void;
type OpusDecodeFn = unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut i16, i32, i32) -> i32;
type OpusDecoderDestroyFn = unsafe extern "C" fn(*mut c_void);
type OpusDecoderCtlFn = unsafe extern "C" fn(*mut c_void, i32, ...) -> i32;

/// `OPUS_RESET_STATE`: the decoder starts again
const OPUS_RESET_STATE: i32 = 4028;

struct LibOpus {
    path: String,
    create: OpusDecoderCreateFn,
    decode: OpusDecodeFn,
    destroy: OpusDecoderDestroyFn,
    ctl: OpusDecoderCtlFn,
}

static LIBOPUS: OnceLock<Result<LibOpus, String>> = OnceLock::new();

fn libopus() -> Result<&'static LibOpus, String> {
    LIBOPUS
        .get_or_init(|| unsafe {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::Foundation::FreeLibrary;
            use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
            let mut dirs: Vec<std::path::PathBuf> = Vec::new();
            if let Some(d) = std::env::var_os("XCL_OPUS_DIR") {
                dirs.push(d.into());
            }
            if let Some(d) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
                dirs.push(d);
            }
            for dir in &dirs {
                for name in ["opus.dll", "libopus.dll", "libopus-0.dll"] {
                    let path = dir.join(name);
                    if !path.is_file() {
                        continue;
                    }
                    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
                    let h = LoadLibraryW(wide.as_ptr()) as usize;
                    if h == 0 {
                        continue;
                    }
                    let f = |n: &str| crate::resolver::system_proc_in(h, &crate::pe::ImportName::Name { hint: 0, name: n.into() });
                    let (Some(create), Some(decode), Some(destroy), Some(ctl)) =
                        (f("opus_decoder_create"), f("opus_decode"), f("opus_decoder_destroy"), f("opus_decoder_ctl"))
                    else {
                        // a DLL that is not libopus (one of the four symbols is missing): not kept loaded
                        FreeLibrary(h as _);
                        continue;
                    };
                    return Ok(LibOpus {
                        path: path.display().to_string(),
                        create: std::mem::transmute::<usize, OpusDecoderCreateFn>(create),
                        decode: std::mem::transmute::<usize, OpusDecodeFn>(decode),
                        destroy: std::mem::transmute::<usize, OpusDecoderDestroyFn>(destroy),
                        ctl: std::mem::transmute::<usize, OpusDecoderCtlFn>(ctl),
                    });
                }
            }
            Err("no opus.dll (libopus) in %XCL_OPUS_DIR% or next to xcl.exe".to_string())
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// One libopus decoder (1 or 2 channels), the same job as [`MftDecoder`].
struct LibDecoder {
    lib: &'static LibOpus,
    state: *mut c_void,
    channels: u16,
    output_rate: u32,
    /// stream time of the next packet, as [`MftDecoder`] keeps it (0: nothing decoded since the start / the last flush)
    timestamp_100ns: i64,
}

impl LibDecoder {
    fn new(channels: u16, sample_rate: u32) -> Result<Self, String> {
        let lib = libopus()?;
        let ch = channels.clamp(1, 2);
        // libopus decodes to 8, 12, 16, 24 or 48 kHz
        let rate = if matches!(sample_rate, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) { sample_rate } else { 48_000 };
        let mut err = 0i32;
        let state = unsafe { (lib.create)(rate as i32, ch as i32, &mut err) };
        if state.is_null() || err != 0 {
            return Err(format!("opus_decoder_create: {err}"));
        }
        Ok(Self { lib, state, channels: ch, output_rate: rate, timestamp_100ns: 0 })
    }

    fn flush(&mut self) {
        unsafe {
            (self.lib.ctl)(self.state, OPUS_RESET_STATE);
        }
        self.timestamp_100ns = 0;
    }

    fn decode_packet(&mut self, packet: &[u8], out: &mut Vec<i16>) -> Result<usize, String> {
        // as MftDecoder: an empty or malformed packet is an error (libopus would conceal a lost one)
        parse_packet(packet, false).ok_or_else(|| "invalid RFC 6716 Opus packet".to_string())?;
        let max_frames = (MAX_PACKET_SAMPLES_48K as usize * self.output_rate as usize) / 48_000;
        let ch = self.channels as usize;
        let start = out.len();
        out.resize(start + max_frames * ch, 0);
        let n = unsafe { (self.lib.decode)(self.state, packet.as_ptr(), packet.len() as i32, out[start..].as_mut_ptr(), max_frames as i32, 0) };
        if n < 0 {
            out.truncate(start);
            return Err(format!("opus_decode: {n}"));
        }
        out.truncate(start + n as usize * ch);
        // the time the packet covers: its frames at the output rate
        self.timestamp_100ns = self.timestamp_100ns.saturating_add(n as i64 * 10_000_000 / self.output_rate as i64);
        Ok(n as usize)
    }
}

impl Drop for LibDecoder {
    fn drop(&mut self) {
        unsafe { (self.lib.destroy)(self.state) }
    }
}

// SAFETY: a libopus decoder state is used by one thread at a time (`&mut self`).
unsafe impl Send for LibDecoder {}

/// One single-stream decoder: Windows' MFT where there is one, else libopus.
enum SingleStreamDecoder {
    Mft(MftDecoder),
    Lib(LibDecoder),
}

impl SingleStreamDecoder {
    fn new(channels: u16, sample_rate: u32, head_bytes: Option<&[u8]>) -> Result<Self, String> {
        // the decoder actually made, logged once (Media Foundation can load and still fail to make one)
        static CHOICE: std::sync::Once = std::sync::Once::new();
        let d = match MftDecoder::new(channels, sample_rate, head_bytes) {
            Ok(d) => Ok(Self::Mft(d)),
            Err(mft) => LibDecoder::new(channels, sample_rate).map(Self::Lib).map_err(|lib| format!("{mft}; {lib}")),
        };
        CHOICE.call_once(|| match &d {
            Ok(Self::Mft(_)) => crate::xlog_info!("Opus: decoder = Windows' MSOpusDecoder (Media Foundation)"),
            Ok(Self::Lib(l)) => crate::xlog_info!("Opus: decoder = libopus ({})", l.lib.path),
            Err(e) => crate::xlog_info!("Opus: decoder = none ({e})"),
        });
        d
    }

    fn output_rate(&self) -> u32 {
        match self {
            Self::Mft(d) => d.output_rate,
            Self::Lib(d) => d.output_rate,
        }
    }

    /// stream time of the next packet (0: nothing decoded since the start / the last flush)
    fn timestamp_100ns(&self) -> i64 {
        match self {
            Self::Mft(d) => d.timestamp_100ns,
            Self::Lib(d) => d.timestamp_100ns,
        }
    }

    fn flush(&mut self) {
        match self {
            Self::Mft(d) => d.flush(),
            Self::Lib(d) => d.flush(),
        }
    }

    fn decode_packet(&mut self, packet: &[u8], out: &mut Vec<i16>) -> Result<usize, String> {
        match self {
            Self::Mft(d) => d.decode_packet(packet, out),
            Self::Lib(d) => d.decode_packet(packet, out),
        }
    }
}

fn mf_create_media_type(rt: &MfRuntime) -> Result<IMFMediaType, String> {
    let mut p = std::ptr::null_mut();
    let hr = unsafe { (rt.create_media_type)(&mut p) };
    if hr.is_err() || p.is_null() {
        return Err(format!("MFCreateMediaType: {:#x}", hr.0 as u32));
    }
    Ok(unsafe { IMFMediaType::from_raw(p) })
}

fn mf_create_sample(rt: &MfRuntime) -> Result<IMFSample, String> {
    let mut p = std::ptr::null_mut();
    let hr = unsafe { (rt.create_sample)(&mut p) };
    if hr.is_err() || p.is_null() {
        return Err(format!("MFCreateSample: {:#x}", hr.0 as u32));
    }
    Ok(unsafe { IMFSample::from_raw(p) })
}

fn mf_create_memory_buffer(rt: &MfRuntime, bytes: u32) -> Result<IMFMediaBuffer, String> {
    let mut p = std::ptr::null_mut();
    let hr = unsafe { (rt.create_memory_buffer)(bytes, &mut p) };
    if hr.is_err() || p.is_null() {
        return Err(format!("MFCreateMemoryBuffer({bytes}): {:#x}", hr.0 as u32));
    }
    Ok(unsafe { IMFMediaBuffer::from_raw(p) })
}

fn create_opus_mft(rt: &MfRuntime) -> Result<IMFTransform, String> {
    unsafe {
        let mut cf_raw = std::ptr::null_mut();
        let hr = (rt.opus_get_class_object)(&CLSID_MS_OPUS_DECODER, &IClassFactory::IID, &mut cf_raw);
        if hr.is_err() || cf_raw.is_null() {
            return Err(format!("MSOpusDecoder DllGetClassObject: {:#x}", hr.0 as u32));
        }
        let factory = IClassFactory::from_raw(cf_raw);
        factory
            .CreateInstance::<_, IMFTransform>(None)
            .map_err(|e| format!("MSOpusDecoder CreateInstance: {:#x}", e.code().0 as u32))
    }
}

// ---------------------------------------------------------------------------------------
// Single-stream MSOpusDecoder MFT (1 or 2 channels) + Multistream Decoder (1..=8 channels)
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputSampleFormat {
    Pcm16,
    Float32,
}

struct MftDecoder {
    mft: IMFTransform,
    channels: u16,
    output_rate: u32,
    output_fmt: OutputSampleFormat,
    mft_provides_samples: bool,
    min_output_bytes: u32,
    timestamp_100ns: i64,
}

impl MftDecoder {
    fn new(channels: u16, sample_rate: u32, head_bytes: Option<&[u8]>) -> Result<Self, String> {
        let rt = mf_runtime()?;
        let mft = create_opus_mft(rt)?;
        let ch = channels.clamp(1, 2);
        let rate = if sample_rate == 0 { 48_000 } else { sample_rate };

        unsafe {
            let in_type = mf_create_media_type(rt)?;
            in_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
                .map_err(|e| format!("SetGUID(MAJOR): {e}"))?;
            in_type
                .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_Opus)
                .map_err(|e| format!("SetGUID(Opus): {e}"))?;
            in_type
                .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, ch as u32)
                .map_err(|e| format!("SetUINT32(CH): {e}"))?;
            in_type
                .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, rate)
                .map_err(|e| format!("SetUINT32(RATE): {e}"))?;
            if let Some(ud) = head_bytes {
                in_type
                    .SetBlob(&MF_MT_USER_DATA, ud)
                    .map_err(|e| format!("SetBlob(USER_DATA): {e}"))?;
            }

            if let Err(e) = mft.SetInputType(0, &in_type, 0) {
                if rate != 48_000 {
                    in_type
                        .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, 48_000)
                        .map_err(|e2| format!("SetUINT32(48k): {e2}"))?;
                    mft.SetInputType(0, &in_type, 0)
                        .map_err(|e2| format!("SetInputType(Opus 48k): {e2}"))?;
                } else {
                    return Err(format!("SetInputType(Opus): {e}"));
                }
            }

            let mut dec = Self {
                mft,
                channels: ch,
                output_rate: rate,
                output_fmt: OutputSampleFormat::Pcm16,
                mft_provides_samples: false,
                min_output_bytes: 0,
                timestamp_100ns: 0,
            };
            dec.configure_output()?;
            let _ = dec.mft.ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
            let _ = dec.mft.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
            Ok(dec)
        }
    }

    unsafe fn configure_output(&mut self) -> Result<(), String> {
        let mut chosen: Option<(IMFMediaType, OutputSampleFormat, u32, u16)> = None;
        for idx in 0..32u32 {
            let Ok(mt) = self.mft.GetOutputAvailableType(0, idx) else {
                break;
            };
            let Ok(sub) = mt.GetGUID(&MF_MT_SUBTYPE) else {
                continue;
            };
            let bits = mt.GetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE).unwrap_or(0);
            let rate = mt.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND).unwrap_or(48_000);
            let ch = mt.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS).unwrap_or(self.channels as u32) as u16;
            if sub == MFAudioFormat_PCM && bits == 16 {
                chosen = Some((mt, OutputSampleFormat::Pcm16, rate, ch));
                break;
            }
            if sub == MFAudioFormat_Float && bits == 32 && chosen.is_none() {
                chosen = Some((mt, OutputSampleFormat::Float32, rate, ch));
            }
        }
        let (mt, fmt, rate, ch) =
            chosen.ok_or_else(|| "MSOpusDecoder offered no PCM16 or Float32 output type".to_string())?;
        self.mft
            .SetOutputType(0, &mt, 0)
            .map_err(|e| format!("SetOutputType: {e}"))?;
        let info = self
            .mft
            .GetOutputStreamInfo(0)
            .map_err(|e| format!("GetOutputStreamInfo: {e}"))?;
        let provides = (info.dwFlags
            & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32
                | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0 as u32))
            != 0;
        self.output_fmt = fmt;
        self.output_rate = rate;
        self.channels = ch;
        self.mft_provides_samples = provides;
        let bytes_per_sample = match fmt {
            OutputSampleFormat::Pcm16 => 2u32,
            OutputSampleFormat::Float32 => 4u32,
        };
        let max_pkt_bytes = MAX_PACKET_SAMPLES_48K * ch as u32 * bytes_per_sample;
        self.min_output_bytes = info.cbSize.max(max_pkt_bytes);
        Ok(())
    }

    fn flush(&mut self) {
        unsafe {
            let _ = self.mft.ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0);
            let _ = self.mft.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
        }
        self.timestamp_100ns = 0;
    }

    fn decode_packet(&mut self, packet: &[u8], out: &mut Vec<i16>) -> Result<usize, String> {
        let info = parse_packet(packet, false)
            .ok_or_else(|| "invalid RFC 6716 Opus packet".to_string())?;
        let rt = mf_runtime()?;
        let dur = duration_100ns(info.total_samples_48k);
        let before = out.len();

        unsafe {
            let sample = mf_create_sample(rt)?;
            let buf = mf_create_memory_buffer(rt, packet.len() as u32)?;
            let mut ptr = std::ptr::null_mut();
            buf.Lock(&mut ptr, None, None)
                .map_err(|e| format!("input Lock: {e}"))?;
            std::ptr::copy_nonoverlapping(packet.as_ptr(), ptr, packet.len());
            let _ = buf.Unlock();
            buf.SetCurrentLength(packet.len() as u32)
                .map_err(|e| format!("input SetCurrentLength: {e}"))?;
            sample
                .AddBuffer(&buf)
                .map_err(|e| format!("input AddBuffer: {e}"))?;
            let _ = sample.SetSampleTime(self.timestamp_100ns);
            let _ = sample.SetSampleDuration(dur);
            self.timestamp_100ns = self.timestamp_100ns.saturating_add(dur);

            match self.mft.ProcessInput(0, &sample, 0) {
                Ok(()) => {}
                Err(e) if e.code() == MF_E_NOTACCEPTING => {
                    self.drain_output(rt, out)?;
                    self.mft
                        .ProcessInput(0, &sample, 0)
                        .map_err(|e2| format!("ProcessInput retry: {e2}"))?;
                }
                Err(e) => return Err(format!("ProcessInput: {e}")),
            }

            self.drain_output(rt, out)?;
        }

        let added = out.len() - before;
        Ok(added / self.channels.max(1) as usize)
    }

    unsafe fn drain_output(&mut self, rt: &MfRuntime, out: &mut Vec<i16>) -> Result<(), String> {
        loop {
            let alloc_sample = if self.mft_provides_samples {
                None
            } else {
                let s = mf_create_sample(rt)?;
                let b = mf_create_memory_buffer(rt, self.min_output_bytes)?;
                s.AddBuffer(&b).map_err(|e| format!("output AddBuffer: {e}"))?;
                Some(s)
            };

            let mut out_buf = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(alloc_sample),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0u32;
            let res = self.mft.ProcessOutput(0, &mut out_buf, &mut status);

            // Always take ownership back out of ManuallyDrop so COM refcounts never leak.
            let sample = ManuallyDrop::into_inner(std::ptr::read(&out_buf[0].pSample));
            let _events = ManuallyDrop::into_inner(std::ptr::read(&out_buf[0].pEvents));

            match res {
                Ok(()) => {
                    if let Some(s) = sample {
                        self.append_sample_pcm16(&s, out)?;
                    }
                }
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => break,
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    self.configure_output()?;
                    continue;
                }
                Err(e) => return Err(format!("ProcessOutput: {e}")),
            }
        }
        Ok(())
    }

    unsafe fn append_sample_pcm16(&self, sample: &IMFSample, out: &mut Vec<i16>) -> Result<(), String> {
        let buf = sample
            .ConvertToContiguousBuffer()
            .map_err(|e| format!("ConvertToContiguousBuffer: {e}"))?;
        let mut ptr = std::ptr::null_mut();
        let mut cur_len = 0u32;
        buf.Lock(&mut ptr, None, Some(&mut cur_len))
            .map_err(|e| format!("output Lock: {e}"))?;
        if cur_len > 0 && !ptr.is_null() {
            let bytes = std::slice::from_raw_parts(ptr, cur_len as usize);
            match self.output_fmt {
                OutputSampleFormat::Pcm16 => {
                    let n = bytes.len() / 2;
                    out.reserve(n);
                    for chunk in bytes.chunks_exact(2) {
                        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                OutputSampleFormat::Float32 => {
                    let n = bytes.len() / 4;
                    out.reserve(n);
                    for chunk in bytes.chunks_exact(4) {
                        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        out.push(f32_to_i16(f));
                    }
                }
            }
        }
        let _ = buf.Unlock();
        Ok(())
    }
}

/// Stateful RFC 6716 / RFC 7845 Opus decoder (1–8 channels) backed by `MSOpusDecoder.dll`.
///
/// `MSOpusDecoder.dll` registers `ThreadingModel = Both` (free-threaded); access is serialized
/// via `&mut self` (or an external `Mutex`).
pub struct Decoder {
    streams: Vec<SingleStreamDecoder>,
    channels: u16,
    coupled_count: u8,
    mapping: Vec<u8>,
    output_rate: u32,
    multistream: bool,
    pre_skip: u16,
    pre_skip_left: usize,
    output_gain_scale: f32,
}

// SAFETY: MSOpusDecoder.dll is ThreadingModel=Both (MTA-safe) and Decoder requires &mut self.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Create a new Opus decoder for `channels` (1..=8), `sample_rate` Hz, and optional
    /// codec-private `extra_data` (e.g. RFC 7845 `OpusHead` from `WAVEFORMATEX` extra bytes).
    pub fn new(channels: u16, sample_rate: u32, extra_data: &[u8]) -> Result<Self, String> {
        let rate = if sample_rate == 0 { 48_000 } else { sample_rate };
        let parsed = parse_opus_head(extra_data);
        let head = match parsed {
            Some(h) => h,
            None => {
                let raw = build_opus_head(channels.clamp(1, 8) as u8, rate, 0, 0);
                parse_opus_head(&raw).unwrap()
            }
        };

        let ch = head.channels as u16;
        let multistream = head.stream_count > 1 || ch > 2;
        let mut streams = Vec::with_capacity(head.stream_count as usize);
        if !multistream {
            let ud = build_opus_head(ch as u8, rate, 0, 0);
            streams.push(SingleStreamDecoder::new(ch, rate, Some(&ud))?);
        } else {
            for s in 0..head.stream_count {
                let sch = if s < head.coupled_count { 2 } else { 1 };
                let ud = build_opus_head(sch as u8, rate, 0, 0);
                streams.push(SingleStreamDecoder::new(sch, rate, Some(&ud))?);
            }
        }
        let output_rate = streams[0].output_rate();
        let pre_skip_scaled = ((head.pre_skip as u64 * output_rate as u64) / 48_000) as usize;
        let output_gain_scale = gain_scale(head.output_gain);

        Ok(Self {
            streams,
            channels: ch,
            coupled_count: head.coupled_count,
            mapping: head.mapping,
            output_rate,
            multistream,
            pre_skip: head.pre_skip,
            pre_skip_left: pre_skip_scaled,
            output_gain_scale,
        })
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    pub fn is_multistream(&self) -> bool {
        self.multistream
    }

    /// Reset internal decoder state (e.g. on `FlushSourceBuffers` or stream boundary).
    pub fn flush(&mut self) {
        for s in &mut self.streams {
            s.flush();
        }
        self.pre_skip_left = ((self.pre_skip as u64 * self.output_rate as u64) / 48_000) as usize;
    }

    /// Decode one RFC 6716 (single-stream or multistream) Opus packet and append interleaved
    /// PCM16 samples to `out`. Returns the number of PCM frames (samples per channel) appended.
    pub fn decode_packet(&mut self, packet: &[u8], out: &mut Vec<i16>) -> Result<usize, String> {
        let start = out.len();
        self.decode_packet_pcm(packet, out)?;
        self.trim_pre_skip(out, start);
        Ok((out.len() - start) / self.channels as usize)
    }

    fn decode_packet_pcm(&mut self, packet: &[u8], out: &mut Vec<i16>) -> Result<(), String> {
        let start = out.len();
        let ch = self.channels as usize;

        if !self.multistream {
            self.streams[0].decode_packet(packet, out)?;
        } else {
            let subs = split_multistream_packet(packet, self.streams.len())
                .ok_or_else(|| "invalid RFC 7845 multistream Opus packet".to_string())?;
            let mut decoded: Vec<Vec<i16>> = Vec::with_capacity(self.streams.len());
            let mut frames = usize::MAX;
            for (i, sub) in subs.iter().enumerate() {
                let mut buf = Vec::new();
                let n = self.streams[i].decode_packet(sub, &mut buf)?;
                frames = frames.min(n);
                decoded.push(buf);
            }
            if frames == usize::MAX {
                frames = 0;
            }
            // Map decoded stream channels to output channels per RFC 7845 mapping table.
            let coupled = self.coupled_count as usize;
            out.reserve(frames * ch);
            for f in 0..frames {
                for &idx in &self.mapping {
                    let sample = if idx == 255 {
                        0i16
                    } else {
                        let idx = idx as usize;
                        if idx < coupled * 2 {
                            let s = idx / 2;
                            let c = idx % 2;
                            decoded[s][f * 2 + c]
                        } else {
                            let s = coupled + (idx - coupled * 2);
                            decoded[s][f]
                        }
                    };
                    out.push(sample);
                }
            }
        }

        // Apply RFC 7845 output gain if non-zero.
        if (self.output_gain_scale - 1.0).abs() > 1e-6 {
            for s in &mut out[start..] {
                let v = (*s as f32 * self.output_gain_scale).round();
                *s = v.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            }
        }

        Ok(())
    }

    fn trim_pre_skip(&mut self, out: &mut Vec<i16>, start: usize) {
        // Trim any remaining pre-skip configured on the voice.
        if self.pre_skip_left > 0 {
            let ch = self.channels as usize;
            let produced_frames = (out.len() - start) / ch;
            let drop_frames = produced_frames.min(self.pre_skip_left);
            if drop_frames > 0 {
                out.drain(start..start + drop_frames * ch);
                self.pre_skip_left -= drop_frames;
            }
        }
    }

    /// Extract and decode all Opus packets from `data` into interleaved PCM16.
    pub fn decode_buffer(&mut self, data: &[u8], block_align: u16) -> Result<Vec<i16>, String> {
        let ExtractedPackets { packets, timing } = extract_buffer(data, block_align, self.multistream);
        if packets.is_empty() {
            return Err("buffer contains no valid RFC 6716 Opus packets".to_string());
        }
        let mut out = Vec::new();
        let timing = timing.filter(|t| {
            // Reuse the voice's configured MFTs only at a decoder stream boundary
            // and when the Ogg headers describe the same channel layout and gain.
            u16::from(t.head.channels) == self.channels
                && usize::from(t.head.stream_count) == self.streams.len()
                && t.head.coupled_count == self.coupled_count
                && t.head.mapping == self.mapping
                && gain_scale(t.head.output_gain) == self.output_gain_scale
                && self.streams.iter().all(|s| s.timestamp_100ns() == 0)
        });
        if let Some(timing) = timing {
            for pkt in &packets {
                let start = out.len();
                if let Err(err) = self.decode_packet_pcm(pkt, &mut out) {
                    out.truncate(start);
                    self.trim_pre_skip(&mut out, 0);
                    return Err(err);
                }
            }
            if timing.trim_pcm(&mut out, self.output_rate, self.channels as usize) {
                self.pre_skip_left = 0;
            } else {
                self.trim_pre_skip(&mut out, 0);
            }
        } else {
            for pkt in &packets {
                self.decode_packet(pkt, &mut out)?;
            }
        }
        Ok(out)
    }
}

fn gain_scale(gain: i16) -> f32 {
    10.0f32.powf(f32::from(gain) / (20.0 * 256.0))
}

#[inline]
fn f32_to_i16(x: f32) -> i16 {
    if !x.is_finite() {
        return 0;
    }
    let scaled = (x * 32768.0).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn rfc6716_table2_all_32_configs() {
        let expected: [(Mode, Bandwidth, u32); 32] = [
            (Mode::Silk, Bandwidth::Narrow, 480),
            (Mode::Silk, Bandwidth::Narrow, 960),
            (Mode::Silk, Bandwidth::Narrow, 1920),
            (Mode::Silk, Bandwidth::Narrow, 2880),
            (Mode::Silk, Bandwidth::Medium, 480),
            (Mode::Silk, Bandwidth::Medium, 960),
            (Mode::Silk, Bandwidth::Medium, 1920),
            (Mode::Silk, Bandwidth::Medium, 2880),
            (Mode::Silk, Bandwidth::Wide, 480),
            (Mode::Silk, Bandwidth::Wide, 960),
            (Mode::Silk, Bandwidth::Wide, 1920),
            (Mode::Silk, Bandwidth::Wide, 2880),
            (Mode::Hybrid, Bandwidth::SuperWide, 480),
            (Mode::Hybrid, Bandwidth::SuperWide, 960),
            (Mode::Hybrid, Bandwidth::Full, 480),
            (Mode::Hybrid, Bandwidth::Full, 960),
            (Mode::Celt, Bandwidth::Narrow, 120),
            (Mode::Celt, Bandwidth::Narrow, 240),
            (Mode::Celt, Bandwidth::Narrow, 480),
            (Mode::Celt, Bandwidth::Narrow, 960),
            (Mode::Celt, Bandwidth::Wide, 120),
            (Mode::Celt, Bandwidth::Wide, 240),
            (Mode::Celt, Bandwidth::Wide, 480),
            (Mode::Celt, Bandwidth::Wide, 960),
            (Mode::Celt, Bandwidth::SuperWide, 120),
            (Mode::Celt, Bandwidth::SuperWide, 240),
            (Mode::Celt, Bandwidth::SuperWide, 480),
            (Mode::Celt, Bandwidth::SuperWide, 960),
            (Mode::Celt, Bandwidth::Full, 120),
            (Mode::Celt, Bandwidth::Full, 240),
            (Mode::Celt, Bandwidth::Full, 480),
            (Mode::Celt, Bandwidth::Full, 960),
        ];
        for (cfg, exp) in expected.iter().enumerate() {
            assert_eq!(config_info(cfg as u8), *exp, "config {cfg}");
        }
        assert_eq!(duration_100ns(120), 25_000);
        assert_eq!(duration_100ns(240), 50_000);
        assert_eq!(duration_100ns(480), 100_000);
        assert_eq!(duration_100ns(960), 200_000);
        assert_eq!(duration_100ns(1920), 400_000);
        assert_eq!(duration_100ns(2880), 600_000);
    }

    #[test]
    fn rfc6716_packet_codes_and_limits() {
        let p = parse_packet(&[0xFC, 0x01, 0x02], false).unwrap();
        assert_eq!((p.config, p.stereo, p.code, p.frames, p.total_samples_48k), (31, true, 0, 1, 960));

        assert!(parse_packet(&[0xF9, 0xAA, 0xBB], false).is_some());
        assert!(parse_packet(&[0xF9, 0xAA], false).is_none());

        let p = parse_packet(&[0xFA, 3, 1, 2, 3, 4, 5], false).unwrap();
        assert_eq!((p.frames, p.total_samples_48k), (2, 1920));
        assert!(parse_packet(&[0xFA, 10, 1, 2], false).is_none());

        let p = parse_packet(&[0x7B, 3, 1, 2, 3, 4, 5, 6], false).unwrap();
        assert_eq!((p.frames, p.total_samples_48k), (3, 2880));
        assert!(parse_packet(&[0x7B, 3, 1, 2, 3, 4], false).is_none());

        let p = parse_packet(&[0x73, 0xC2, 2, 2, 0xAA, 0xBB, 0xCC, 0, 0], false).unwrap();
        assert_eq!((p.frames, p.total_samples_48k), (2, 960));

        assert!(parse_packet(&[0xFB, 6, 0, 0, 0, 0, 0, 0], false).is_some());
        assert!(parse_packet(&[0xFB, 7, 0, 0, 0, 0, 0, 0, 0], false).is_none());
        assert!(parse_packet(&[0xFB, 0], false).is_none());
    }

    #[test]
    fn opus_head_roundtrip_mono_stereo_surround() {
        for ch in 1..=8u8 {
            let raw = build_opus_head(ch, 48_000, 312, -256);
            let h = parse_opus_head(&raw).unwrap();
            assert_eq!(h.channels, ch);
            assert_eq!(h.input_rate, 48_000);
            assert_eq!(h.pre_skip, 312);
            assert_eq!(h.output_gain, -256);
            assert_eq!(h.mapping_family, if ch <= 2 { 0 } else { 1 });
        }
    }

    /// A sound of the game (`XCL_SAMPLE_AWC`: an AWC with one Opus stream, its data chunk as
    /// the console stores it): every packet decodes, the length matches the AWC's sample count,
    /// it is not silence. With `XCL_OPUS_WAV` the result is written there to listen to.
    #[test]
    fn decodes_a_game_sound() {
        let Ok(path) = std::env::var("XCL_SAMPLE_AWC") else {
            println!("XCL_SAMPLE_AWC not set – skipping");
            return;
        };
        if !is_available() {
            println!("MSOpusDecoder.dll not available – skipping");
            return;
        }
        let awc = std::fs::read(path).unwrap();
        let (data, samples, channels) = awc_stream(&awc);
        let mut d = Decoder::new(channels, 48_000, &[]).unwrap();
        // block_align 1: the AWC block size is assumed
        let pcm = d.decode_buffer(data, 1).unwrap();
        let frames = pcm.len() / channels as usize;
        println!("{} B of Opus → {frames} samples per channel (the AWC says {samples})", data.len());
        // whole 20 ms frames: up to three frames past the AWC's count (its encoder delay / end)
        assert!(frames >= samples && frames - samples < 3 * 960, "{frames} samples for {samples}");
        let rms = (pcm.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / pcm.len() as f64).sqrt();
        assert!(rms > 100.0, "silence (rms {rms})");
        if let Ok(out) = std::env::var("XCL_OPUS_WAV") {
            let mut w = Vec::new();
            let bytes = (pcm.len() * 2) as u32;
            w.extend_from_slice(b"RIFF");
            w.extend_from_slice(&(36 + bytes).to_le_bytes());
            w.extend_from_slice(b"WAVEfmt ");
            for v in [16u32, 1 | (channels as u32) << 16, 48_000, 96_000 * channels as u32, (2 * channels as u32) | 16 << 16] {
                w.extend_from_slice(&v.to_le_bytes());
            }
            w.extend_from_slice(b"data");
            w.extend_from_slice(&bytes.to_le_bytes());
            pcm.iter().for_each(|s| w.extend_from_slice(&s.to_le_bytes()));
            std::fs::write(&out, w).unwrap();
            println!("written to {out} (rms {rms:.0})");
        }
    }

    /// (data chunk, samples, channels) of a single-stream AWC: header "ADAT", version, flags,
    /// streams, data offset; chunk table of u64 (type byte 7, size bits 28-55, offset 0-27).
    fn awc_stream(b: &[u8]) -> (&[u8], usize, u16) {
        assert_eq!(&b[..4], b"ADAT");
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (flags, streams) = (u16_at(6), u32_at(8) as usize);
        assert_eq!(streams, 1, "a single-stream AWC");
        let mut o = 16 + if flags & 1 != 0 { 2 * streams } else { 0 };
        let count = (u32_at(o) >> 29) as usize;
        o += 4;
        let chunk = |ty: u64| {
            (0..count)
                .map(|c| u64::from_le_bytes(b[o + 8 * c..o + 8 * c + 8].try_into().unwrap()))
                .find(|v| v >> 56 == ty)
                .map(|v| ((v & 0xFFF_FFFF) as usize, ((v >> 28) & 0xFFF_FFFF) as usize))
                .unwrap()
        };
        let (doff, dsz) = chunk(0x55);
        let (foff, _) = chunk(0x76);
        assert_eq!(b[foff + 28], 0x0B, "Opus");
        (&b[doff..doff + dsz], u32_at(foff) as usize, b[foff + 29].max(1) as u16)
    }

    fn push_ogg_page(
        ogg: &mut Vec<u8>,
        serial: u32,
        sequence: u32,
        flags: u8,
        granule: u64,
        laces: &[u8],
        body: &[u8],
    ) {
        assert_eq!(laces.iter().map(|&l| usize::from(l)).sum::<usize>(), body.len());
        let start = ogg.len();
        ogg.extend_from_slice(b"OggS\x00");
        ogg.push(flags);
        ogg.extend_from_slice(&granule.to_le_bytes());
        ogg.extend_from_slice(&serial.to_le_bytes());
        ogg.extend_from_slice(&sequence.to_le_bytes());
        ogg.extend_from_slice(&[0; 4]);
        ogg.push(u8::try_from(laces.len()).unwrap());
        ogg.extend_from_slice(laces);
        ogg.extend_from_slice(body);

        let mut crc = 0u32;
        for &byte in &ogg[start..] {
            crc ^= u32::from(byte) << 24;
            for _ in 0..8 {
                let polynomial = if crc & 0x8000_0000 != 0 { 0x04C1_1DB7 } else { 0 };
                crc = (crc << 1) ^ polynomial;
            }
        }
        ogg[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
    }

    fn push_ogg_packets(
        ogg: &mut Vec<u8>,
        serial: u32,
        sequence: u32,
        flags: u8,
        granule: u64,
        packets: &[&[u8]],
    ) {
        let mut laces = Vec::new();
        for packet in packets {
            laces.extend(std::iter::repeat(255).take(packet.len() / 255));
            laces.push((packet.len() % 255) as u8);
        }
        push_ogg_page(ogg, serial, sequence, flags, granule, &laces, &packets.concat());
    }

    fn ogg_headers(head: &[u8], serial: u32) -> Vec<u8> {
        let mut ogg = Vec::new();
        push_ogg_packets(&mut ogg, serial, 0, 0x02, 0, &[head]);
        push_ogg_packets(&mut ogg, serial, 1, 0, 0, &[b"OpusTags\0\0\0\0\0\0\0\0"]);
        ogg
    }

    #[test]
    fn ogg_eos_frame_estimate_handles_preskip_and_initial_granules() {
        let head = build_opus_head(1, 48_000, 313, 0);
        let packet = &[0xF8, 0xFF, 0xFE];
        let mut ogg = ogg_headers(&head, 7);
        let headers_len = ogg.len();

        for (granule, keep) in [(800, 800), (1960, 960)] {
            ogg.truncate(headers_len);
            push_ogg_packets(&mut ogg, 7, 2, 0x04, granule, &[packet]);
            assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(keep - 313));
        }

        ogg.truncate(headers_len);
        // A 5000-frame initial offset puts both absolute granules beyond the decoded length.
        push_ogg_packets(&mut ogg, 7, 2, 0, 5960, &[packet]);
        push_ogg_packets(&mut ogg, 7, 3, 0x04, 6724, &[packet]);
        assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(1724 - 313));
        assert_eq!(estimate_buffer_frames(&ogg, 0, 24_000, false), Some(862 - 156));

        ogg.truncate(headers_len);
        push_ogg_packets(&mut ogg, 7, 2, 0, 5960, &[packet]);
        push_ogg_packets(&mut ogg, 7, 3, 0x04, 7684, &[packet, packet]);
        assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(2684 - 313));

        for granule in [200, 313] {
            ogg.truncate(headers_len);
            push_ogg_packets(&mut ogg, 7, 2, 0x04, granule, &[packet]);
            assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(0));
        }
    }

    #[test]
    fn ogg_timing_requires_one_complete_contiguous_link() {
        let head = build_opus_head(1, 48_000, 0, 0);
        let packet = &[0xF8, 0xFF, 0xFE];
        let mut ogg = ogg_headers(&head, 7);
        let audio_start = ogg.len();
        push_ogg_packets(&mut ogg, 7, 2, 0x04, 800, &[packet]);
        assert_eq!(estimate_buffer_frames(&ogg[audio_start..], 0, 48_000, false), Some(960));

        ogg.extend_from_slice(&ogg_headers(&head, 8));
        push_ogg_packets(&mut ogg, 8, 2, 0x04, 800, &[packet]);
        assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(1920));

        ogg.truncate(audio_start);
        push_ogg_packets(&mut ogg, 7, 2, 0, 960, &[packet]);
        assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(960));
        push_ogg_packets(&mut ogg, 7, 4, 0x04, 1600, &[packet]);
        assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(1920));
    }

    #[test]
    fn ogg_timing_counts_continued_packet_on_its_completion_page() {
        let head = build_opus_head(1, 48_000, 0, 0);
        let mut ogg = ogg_headers(&head, 7);
        let packet = &[0xF8, 0xFF, 0xFE];
        let mut continued = [0u8; 256];
        // One 20 ms frame in code-3 framing with 251 bytes of padding.
        continued[..5].copy_from_slice(&[0xFB, 0x41, 251, 0xFF, 0xFE]);
        push_ogg_packets(&mut ogg, 7, 2, 0, 5960, &[packet]);
        push_ogg_page(&mut ogg, 7, 3, 0, u64::MAX, &[255], &continued[..255]);
        push_ogg_page(&mut ogg, 7, 4, 0x05, 6724, &[1], &continued[255..]);

        let packets = extract_packets(&ogg, 0, false);
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].as_ref(), packet);
        assert_eq!(packets[1].as_ref(), continued.as_slice());
        assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(1724));
    }

    #[test]
    fn ogg_pcm_trim_requires_full_mft_output() {
        let timing = OggTiming {
            head: parse_opus_head(&build_opus_head(2, 48_000, 313, 0)).unwrap(),
            decoded_48k: 1920,
            end_48k: 1724,
        };
        let pcm: Vec<i16> = (0..1920).collect(); // 960 stereo frames at 24 kHz.
        let mut primed = pcm[..1440].to_vec();
        assert!(!timing.trim_pcm(&mut primed, 24_000, 2));
        assert_eq!(primed.as_slice(), &pcm[..1440]);

        let mut full = pcm.clone();
        assert!(timing.trim_pcm(&mut full, 24_000, 2));
        assert_eq!(full.as_slice(), &pcm[156 * 2..862 * 2]);

        let shorter_than_preskip = OggTiming {
            head: parse_opus_head(&build_opus_head(2, 48_000, 313, 0)).unwrap(),
            decoded_48k: 1920,
            end_48k: 200,
        };
        let mut discarded = pcm.clone();
        assert!(shorter_than_preskip.trim_pcm(&mut discarded, 24_000, 2));
        assert!(discarded.is_empty());
    }

    #[test]
    fn framing_detection_awc_blocks_and_raw() {
        let pkt1 = [0x78u8, 0x11, 0x22, 0x33];
        let pkt2 = [0xF8u8, 0x44, 0x55];
        let mut block = vec![0u8; 2048];
        block[0..2].copy_from_slice(&(pkt1.len() as u16).to_le_bytes());
        block[2..6].copy_from_slice(&pkt1);
        block[6..8].copy_from_slice(&(pkt2.len() as u16).to_le_bytes());
        block[8..11].copy_from_slice(&pkt2);
        let got = extract_packets(&block, 0, false);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].as_ref(), &pkt1);
        assert_eq!(got[1].as_ref(), &pkt2);

        let mut two = block.clone();
        two.extend_from_slice(&block);
        let got2 = extract_packets(&two, 2048, false);
        assert_eq!(got2.len(), 4);
        assert_eq!(estimate_buffer_frames(&two, 2048, 48_000, false), Some(3840));

        let mut tricky = vec![0u8; 250];
        tricky[0] = 0xF8;
        tricky[1] = 0x00;
        let got_raw = extract_packets(&tricky, 1, false);
        assert_eq!(got_raw.len(), 1);
        assert_eq!(got_raw[0].as_ref(), tricky.as_slice());
        assert_eq!(estimate_buffer_frames(&tricky, 1, 48_000, false), Some(960));
    }

    // Compact real libopus RFC 6716 test packets (440 Hz mono, 440/880 Hz stereo, SILK/Hybrid/CELT, 5.1).
    const MONO_HYBRID_20MS_HEAD: &[u8] = &[79, 112, 117, 115, 72, 101, 97, 100, 1, 1, 56, 1, 128, 187, 0, 0, 0, 0, 0];
    pub(crate) const MONO_HYBRID_20MS: &[&[u8]] = &[
        &[0x78, 0x81, 0x7b, 0xc6, 0x11, 0x76, 0xf4, 0x75, 0x00, 0x00, 0x07, 0x8e, 0xca, 0x69, 0xdb, 0xbb, 0xcc, 0x48, 0xdc, 0x16, 0xbd, 0x9b, 0xd3, 0x58, 0x27, 0x78, 0x93, 0xbc, 0x37, 0x44, 0x5e, 0xcf, 0x2d, 0x97, 0x00, 0x69, 0x80, 0x11, 0x80, 0x2c, 0x03, 0x38, 0xab, 0x45, 0x61, 0xfd, 0x89, 0xb7, 0x7e, 0x0d, 0x17, 0xbf, 0x20, 0xa5, 0x2b, 0xb5, 0xea, 0x2b, 0x5e, 0xd3, 0xa7, 0x48, 0xd1, 0x18, 0xf2, 0x0b, 0x5c, 0x01, 0x4f, 0x5f, 0xa1, 0x96, 0x21, 0xae, 0xde, 0xe0, 0x53, 0x00, 0xd6, 0x65, 0x65, 0x3f, 0x1f, 0xa6, 0xcb, 0xb9, 0x77, 0xbf, 0x97, 0x62, 0xc1, 0x12, 0xcf, 0x8d, 0xc2, 0x7d, 0xfd, 0x00, 0x80, 0xff, 0xfa, 0xc6, 0x88, 0xb6, 0xde, 0x10, 0xb8, 0x4f, 0xc6, 0x23, 0x6b, 0xea, 0x87, 0xc9, 0x05],
        &[0x78, 0x9e, 0x1e, 0xa1, 0xe7, 0xfc, 0x95, 0x4d, 0x4e, 0xaa, 0x18, 0x0a, 0xca, 0x5d, 0x07, 0x64, 0x44, 0x34, 0x94, 0x45, 0x3e, 0x66, 0x66, 0xca, 0x44, 0xed, 0x1f, 0x0d, 0x44, 0x6b, 0xf3, 0x4f, 0xcc, 0xa5, 0x55, 0x45, 0x41, 0x9e, 0xb4, 0xf3, 0xfc, 0x13, 0x38, 0x65, 0x79, 0xed, 0x99, 0xf0, 0x1b, 0x7f, 0xbd, 0x3c, 0x92, 0x73, 0xd5, 0xee, 0x8f, 0xeb, 0xab, 0x76, 0x32, 0xa5, 0xf4, 0xa1, 0x0c, 0x35, 0xbe, 0xab, 0xbb, 0x61, 0x99, 0xbf, 0xbd],
        &[0x78, 0x99, 0xc2, 0x5f, 0x75, 0x9c, 0xfc, 0x4b, 0x3a, 0x45, 0xab, 0x5a, 0x7e, 0x7d, 0x7b, 0x37, 0x7c, 0x02, 0xb4, 0x4a, 0x61, 0xf0, 0x89, 0x27, 0x87, 0xce, 0x50, 0xfb, 0xe3, 0xc3, 0x04, 0xcb, 0xaf, 0x99, 0x64, 0x7c, 0x17, 0x1a, 0x33, 0x8d, 0xeb, 0xae, 0xba, 0x19, 0x9b, 0xdb, 0xe1, 0xd7, 0x19, 0xb5, 0xc8, 0xa1, 0x79, 0xd6, 0x1c, 0xb3, 0x68, 0xe5, 0x2b, 0x25, 0x98, 0x0c, 0x5c, 0xbd, 0x64, 0x08, 0x11, 0xbb, 0x14, 0xd0, 0xbb, 0x08, 0x0d, 0xf9, 0xff, 0xdb, 0x05],
        &[0x78, 0x99, 0xc2, 0x5f, 0x73, 0x2d, 0xd2, 0x2d, 0xed, 0xd7, 0x7f, 0xc4, 0x72, 0x9e, 0xe4, 0xeb, 0x27, 0x57, 0xa3, 0x16, 0xf3, 0xb3, 0xb6, 0x2c, 0x03, 0xfc, 0x49, 0x19, 0x83, 0x78, 0x64, 0x9d, 0x41, 0x4c, 0x6a, 0x16, 0xfc, 0x80, 0xea, 0x13, 0xc6, 0x03, 0x49, 0xbf, 0xa5, 0x19, 0x46, 0x91, 0x46, 0x8c, 0xc6, 0x1e, 0x3d, 0x86, 0xd9, 0x6e, 0x57, 0xa8, 0x95, 0x55, 0x90, 0x38, 0x9d, 0xf3, 0x03, 0x88, 0x71, 0x90, 0x83, 0x6f, 0x80, 0x8a, 0x96, 0x45, 0xb9, 0x64, 0xc3, 0xb6, 0x4d],
        &[0x78, 0x99, 0xc2, 0x5f, 0x73, 0x2d, 0xd2, 0x31, 0x33, 0xbf, 0x7b, 0x34, 0x8a, 0x71, 0x0f, 0x77, 0xd5, 0xd5, 0x1b, 0xda, 0x78, 0xa4, 0x47, 0xbb, 0x8c, 0x85, 0x53, 0xbe, 0x21, 0xd6, 0x5f, 0x9f, 0x2e, 0xba, 0x38, 0xfa, 0x2d, 0x32, 0xbc, 0xa1, 0xd7, 0xce, 0xb0, 0x1b, 0x08, 0x44, 0x28, 0x21, 0xcb, 0x80, 0x40, 0xaf, 0xc4, 0x08, 0xdc, 0x96, 0xfd, 0xb6, 0x95, 0x96, 0xda, 0xf3, 0xae, 0x6c, 0x2b, 0x04, 0xc4, 0x23, 0xb8, 0x12, 0x45, 0x51, 0xf9, 0x5f, 0x9b, 0xb0, 0xc6, 0x86, 0xed, 0x08, 0x0c],
    ];
    const STEREO_CELT_20MS: &[&[u8]] = &[
        &[0xfc, 0xb4, 0x7b, 0xa1, 0xad, 0xc7, 0x47, 0x6a, 0x91, 0x46, 0xf4, 0x59, 0x8e, 0x70, 0x13, 0x22, 0xc8, 0x06, 0x0d, 0xc3, 0x87, 0x60, 0xdb, 0x63, 0xcf, 0x46, 0x6a, 0x3f, 0xb0, 0x32, 0x0f, 0x00, 0xb5, 0xda, 0x9d, 0x0e, 0xeb, 0xe3, 0xb5, 0xe7, 0xb8, 0xc5, 0x47, 0x74, 0x15, 0xc8, 0x58, 0x58, 0xab, 0xc8, 0xf4, 0xf7, 0x8f, 0x0a, 0x8a, 0x1f, 0xde, 0x14, 0xa4, 0x36, 0x79, 0xb6, 0x93, 0x2f, 0x70, 0x45, 0x0f, 0x6a, 0x8c, 0x33, 0xd6, 0xe9, 0xaf, 0x21, 0x3b, 0x81, 0xaa, 0x0a, 0xf1, 0xb2, 0x88, 0x66, 0x82, 0x45, 0x81, 0x0d, 0x05, 0xd1, 0x9b, 0xf3, 0x73, 0x25, 0xbe, 0x04, 0x05, 0x1f, 0x79, 0x1c, 0xc9, 0x4a, 0x6e, 0x9c, 0x4e, 0x65, 0xdb, 0x89, 0x59, 0xc6, 0x64, 0x01, 0x54, 0x77, 0x0e, 0x60, 0x4f, 0xa1, 0x2a, 0xa6, 0x2a, 0xc8, 0xc9, 0x5d, 0x79, 0xb5, 0x7b, 0x6f, 0xfa, 0x43, 0x74, 0xa0, 0x09, 0x72, 0xf7, 0x84, 0x9d, 0xca, 0x03, 0x9a, 0xcd, 0x16, 0xd2, 0xa1, 0x9a, 0x67, 0xaa, 0x3e, 0x75, 0x2b, 0x68, 0x65, 0x1d, 0xe1, 0x0e, 0x9e, 0x85, 0x8c, 0xa9, 0x51, 0x61, 0xf8, 0xc4, 0xc4, 0xf3, 0x37, 0x9c, 0x87, 0xb4, 0x38, 0xd4, 0x5b, 0xb6, 0x71, 0xd1, 0x31, 0x8c, 0xaf, 0xb2, 0x88, 0x24, 0x29, 0x14, 0x43, 0x9f, 0x42, 0xb6, 0xd8, 0xa4, 0x43, 0x9f, 0x47, 0x28, 0x2e, 0x1f, 0x70, 0x3f, 0xbf, 0x8d, 0xde, 0xa9, 0x12, 0xaf, 0x8a, 0xfe, 0x7c, 0xa0, 0x7e, 0xb6, 0xc1, 0xb4, 0x47, 0xb9, 0xfa, 0x0e, 0x4f, 0xc8, 0x27, 0x9b, 0xbb, 0xaa, 0x06, 0x60, 0x61, 0xbc, 0x02, 0x6e, 0x41, 0xda, 0x0d, 0x1c, 0x26, 0xe8, 0xa3, 0x8d, 0xd7, 0x9b, 0xbf, 0xd9, 0xd5, 0x70, 0xbd, 0x20, 0x4f, 0xaf, 0x17, 0xee, 0x4c, 0xdc, 0xad, 0xc4, 0x86, 0x6a, 0x68, 0x14, 0x1e, 0x53, 0x20, 0x8d, 0xab, 0x4b, 0xc9, 0x90, 0x7a, 0xcc, 0xd0, 0xc6, 0x73, 0x1b, 0x81, 0x12, 0xdb, 0x1e, 0x9a, 0xdf, 0x5e, 0xca, 0xbc, 0xd3, 0x94, 0xe0, 0xd2, 0x8d, 0x8d, 0x27, 0x69, 0xb1, 0x39, 0x64, 0x53, 0xda, 0x73, 0x44, 0x96, 0xe8, 0x35, 0xef, 0xcc, 0x0d, 0x15, 0x74, 0x88, 0x09, 0x30, 0xb1, 0x8b, 0xea, 0xb6, 0x13, 0x6c],
        &[0xfc, 0xb1, 0xb2, 0x09, 0x29, 0x22, 0x3e, 0xc9, 0xe3, 0x9f, 0x48, 0xad, 0xdb, 0x86, 0xb6, 0x82, 0xc3, 0x0f, 0xaf, 0xbc, 0x6f, 0x2e, 0x98, 0x92, 0x63, 0x99, 0x63, 0x03, 0x9c, 0xbb, 0xcc, 0x64, 0xfe, 0xc7, 0xb7, 0xed, 0xa9, 0x6b, 0x1f, 0x10, 0x88, 0xd4, 0x26, 0xd7, 0x86, 0xae, 0xf3, 0x8c, 0xab, 0xd0, 0x79, 0xfd, 0xa8, 0x05, 0xa3, 0x03, 0x34, 0x27, 0x41, 0xe0, 0x64, 0x40, 0x8e, 0x6e, 0x90, 0x34, 0xef, 0x22, 0xad, 0xb1, 0xee, 0xf3, 0x00, 0x42, 0x0c, 0x03, 0x91, 0x9c, 0x76, 0xb9, 0xfe, 0x0d, 0xdb, 0x48, 0x8b, 0xb4, 0xdc, 0x31, 0x54, 0x01, 0xc2, 0x19, 0x46, 0xc9, 0x3f, 0x3c, 0x92, 0xbe, 0x9f, 0xf1, 0x35, 0xcb, 0x11, 0x45, 0x94, 0x3a, 0x3a, 0xc1, 0x59, 0x6f, 0xbc, 0xdd, 0xac, 0x08, 0x77, 0x0d, 0x43, 0xce, 0x38, 0x0f, 0x30, 0x57, 0xee, 0xe8, 0x07, 0xe0, 0xd0, 0x06, 0x71, 0x4d, 0x13, 0xbe, 0xa9, 0xf2, 0xe2, 0x09, 0xee, 0x22, 0x8f, 0xec, 0xcc, 0x8d, 0xba, 0x4e, 0x02, 0xfe, 0x5a, 0x4d, 0x01, 0xb2, 0xe5, 0x61, 0x28, 0xf0, 0xf7, 0x49, 0xf7, 0xed, 0xb6, 0x60, 0xbd, 0xaa, 0x41, 0xc0, 0xb4, 0xfe, 0x54, 0xfe, 0x93, 0x04, 0xee, 0xec, 0xcf, 0xa4, 0xda, 0x53, 0x64, 0xb9, 0xdc, 0x41, 0xc8, 0x92, 0x9d, 0x16, 0x27, 0xa6, 0x5c, 0xe4, 0x07, 0x5b, 0xde, 0x21, 0x90, 0xdc, 0x1f, 0x46, 0xb4, 0x4f, 0xae],
        &[0xfc, 0xb0, 0x51, 0xe8, 0xef, 0xfb, 0x57, 0x24, 0x51, 0x09, 0xde, 0xfb, 0x85, 0x76, 0x41, 0x89, 0x29, 0xbc, 0x9b, 0xc3, 0x81, 0x49, 0xea, 0xd5, 0xad, 0xf5, 0xa3, 0x70, 0xf4, 0x81, 0x7a, 0x7a, 0xe6, 0x12, 0x4f, 0x27, 0x49, 0x71, 0x45, 0x43, 0x33, 0x8b, 0xfb, 0x1e, 0xed, 0x17, 0x08, 0x00, 0x9d, 0xa2, 0x59, 0x3f, 0x0a, 0x0a, 0x42, 0xc5, 0x81, 0xd9, 0xb4, 0x10, 0x30, 0x1b, 0x83, 0xc2, 0x17, 0xa2, 0x77, 0xcf, 0x92, 0x93, 0xdb, 0x8b, 0x20, 0xf3, 0x18, 0x35, 0x38, 0x0a, 0x76, 0x7a, 0x9b, 0x30, 0x0a, 0x60, 0x3a, 0x98, 0x67, 0xc6, 0x95, 0xc4, 0x10, 0x8f, 0x26, 0x68, 0x1d, 0x86, 0x40, 0x66, 0xa8, 0x08, 0xde, 0x2f, 0xd8, 0x9c, 0x81, 0x21, 0x93, 0xc8, 0x39, 0x48, 0x03, 0xf7, 0xb1, 0x0b, 0xc7, 0x7c, 0xc9, 0x82, 0xa0, 0xab, 0xf3, 0x7d, 0xff, 0xd7, 0x74, 0x00, 0x8c, 0x7e, 0x32, 0xe9, 0x7a, 0x88, 0x55, 0x16, 0x4d, 0x1d, 0xd1, 0x15, 0xb4, 0xb4, 0x0c, 0x8e, 0x38, 0xd7, 0x7b, 0xbf, 0x9d, 0x45, 0x35, 0x9b, 0xd1, 0x17, 0x52, 0x05, 0x37, 0x1b, 0x94, 0x72, 0xc2, 0xc5, 0x55, 0x61, 0x62, 0x14, 0xb4, 0x51, 0xad, 0x62, 0xca, 0x31, 0x34, 0x9d, 0x0f, 0x68, 0x17, 0x8f, 0x64, 0x3c, 0x28, 0x73, 0x3e, 0xea, 0xa4, 0x34, 0x26, 0x32, 0xd4, 0x75, 0x4c, 0x17, 0x93, 0x16, 0x73, 0x7c, 0xb5, 0x7d, 0xc9, 0xd9, 0x21, 0xfb, 0xa9, 0xcd, 0xae],
        &[0xfc, 0xad, 0xbb, 0x41, 0xf7, 0x90, 0x00, 0xc1, 0x26, 0x90, 0x40, 0x93, 0x56, 0x00, 0x2b, 0x9f, 0xe7, 0x5e, 0x4d, 0x4e, 0x7c, 0x92, 0xcf, 0x1e, 0x84, 0xfc, 0xa4, 0xde, 0x43, 0xca, 0x2e, 0x47, 0x3a, 0xb5, 0x36, 0xcf, 0x00, 0xd2, 0x9b, 0x63, 0xec, 0xfa, 0x29, 0x73, 0xa9, 0x57, 0xea, 0x4a, 0x54, 0x8f, 0xd3, 0xd5, 0x4f, 0xff, 0xb2, 0x1f, 0xcb, 0xd0, 0x6d, 0xec, 0xc2, 0x88, 0x76, 0x3b, 0x82, 0xb7, 0xec, 0xbd, 0x31, 0x74, 0x90, 0x4d, 0x9e, 0xb0, 0x96, 0x9d, 0xc2, 0x14, 0x98, 0x2a, 0x31, 0x2b, 0x91, 0xc6, 0xc8, 0xae, 0x94, 0xed, 0xf6, 0xbc, 0xb1, 0xe8, 0xcd, 0x86, 0x5e, 0xf4, 0xcc, 0xe9, 0x9e, 0x49, 0x85, 0xeb, 0x22, 0x79, 0xda, 0xf0, 0x37, 0xfe, 0x0f, 0x54, 0xdc, 0xf3, 0x3c, 0x4f, 0x33, 0x51, 0x32, 0xd3, 0x7d, 0x9e, 0xd7, 0xd8, 0x42, 0x3d, 0x2f, 0x79, 0xcf, 0x56, 0xed, 0x64, 0x89, 0x2a, 0x87, 0x30, 0xff, 0x22, 0x32, 0x2c, 0x47, 0x70, 0xc0, 0xdc, 0xb4, 0x6d, 0xf3, 0x45, 0xa8, 0xf5, 0xb9, 0x32, 0x71, 0x4a, 0x72, 0x9a, 0xce, 0xb2, 0x49, 0x67, 0xbd, 0x36, 0x71, 0x22, 0xb4, 0x1c, 0x1a, 0x78, 0x2b, 0xb3, 0xe2, 0xe1, 0xea, 0xcd, 0x56, 0xec, 0x28, 0xc3, 0x74, 0x4c, 0x1a, 0xcf, 0x73, 0x9d, 0x8b, 0x34, 0xd5, 0x02, 0x10, 0xfe, 0x1b, 0x95, 0x08, 0xbb, 0x79, 0x14, 0xa5, 0x49, 0xff, 0x57, 0xae],
        &[0xfc, 0xb0, 0x60, 0xea, 0x2d, 0x99, 0x1b, 0xcf, 0xdf, 0x0c, 0xc9, 0x9b, 0xd9, 0x34, 0x5c, 0x21, 0xcc, 0x4f, 0x1e, 0xe3, 0xde, 0x4d, 0xcf, 0x08, 0xdc, 0xf6, 0x42, 0xb1, 0x42, 0xcd, 0xfd, 0xab, 0x00, 0x38, 0xe6, 0xc6, 0xca, 0xb3, 0x89, 0xea, 0xe6, 0x1d, 0x72, 0xe5, 0xeb, 0x77, 0x8c, 0x42, 0xc6, 0x90, 0x5a, 0xa3, 0xc5, 0xc6, 0x66, 0x42, 0xd4, 0xcf, 0x67, 0xfb, 0x43, 0xf3, 0xb7, 0x49, 0x24, 0x94, 0xf1, 0x90, 0xb4, 0x3c, 0x63, 0x06, 0x0d, 0x72, 0x87, 0xf3, 0xfb, 0x52, 0x8d, 0x19, 0xe4, 0x14, 0xe0, 0xd1, 0xa7, 0x0a, 0xdb, 0x17, 0x31, 0xd2, 0x6b, 0xa3, 0x61, 0xf7, 0x5f, 0x67, 0xd0, 0xfd, 0x7c, 0x7f, 0x9a, 0x65, 0x44, 0xd3, 0xc2, 0x1c, 0x99, 0xbc, 0xcd, 0x27, 0x0f, 0x45, 0xad, 0x36, 0x86, 0x58, 0xaa, 0x6c, 0xd4, 0xdb, 0x2e, 0x96, 0x9d, 0x0d, 0xcc, 0x21, 0xfb, 0x54, 0xd0, 0x15, 0x1d, 0x6a, 0x64, 0x1e, 0x06, 0x1c, 0xf7, 0x96, 0x84, 0xf3, 0x06, 0xcf, 0xe7, 0x96, 0xe8, 0x25, 0x4b, 0x31, 0xed, 0x73, 0x8d, 0x65, 0x90, 0x53, 0xed, 0xc7, 0xe9, 0xf7, 0x4b, 0xe9, 0xf2, 0x5e, 0x7e, 0x8c, 0x0b, 0x2e, 0x9c, 0xb2, 0x84, 0x66, 0xee, 0x62, 0xf9, 0xd0, 0x12, 0xc5, 0xb9, 0x11, 0x98, 0x1f, 0xd9, 0x43, 0x5d, 0xb5, 0x2a, 0x70, 0x4a, 0x3f, 0xcd, 0x6c, 0xad, 0x87, 0x92, 0x7b, 0xb9, 0x69, 0xb7, 0xd7, 0xae],
    ];
    const MONO_SILK_40MS: &[&[u8]] = &[
        &[0x50, 0xc1, 0x2d, 0xdb, 0xb6, 0x08, 0x9f, 0xc2, 0x01, 0xd7, 0xa1, 0xbc, 0xc9, 0xf9, 0xc5, 0xe5, 0xb1, 0xb4, 0x4f, 0xa8, 0x69, 0xb3, 0x4c, 0x0f, 0x60, 0xfb, 0xc6, 0x51, 0x7a, 0x19, 0x6f, 0x66, 0x7e, 0xdf, 0xd5, 0x46, 0x29, 0x70, 0xa9, 0x90, 0x23, 0xac, 0x5b, 0xde, 0x63, 0x0e, 0x9b, 0x59, 0x3c, 0xd2, 0x70, 0xb9, 0x76, 0x1e, 0x05, 0xb8, 0xad, 0x28, 0x3d, 0x4e, 0x74, 0xd1, 0xa7, 0x99, 0x4f, 0x3f, 0xf2, 0x01, 0xf4, 0xd1, 0x98, 0x78, 0xde, 0x01, 0xbd, 0x4d, 0xc8, 0xa5, 0x4c, 0x9d, 0x44, 0x0b, 0x12, 0xba, 0x43, 0x13, 0x1d],
        &[0x50, 0xce, 0x49, 0xc9, 0x2b, 0x68, 0x06, 0x73, 0x42, 0xe7, 0x63, 0xc0, 0x86, 0xd7, 0xb9, 0x98, 0x31, 0x71, 0xe0, 0xa2, 0xc4, 0xcf, 0x34, 0x54, 0xe2, 0xef, 0x43, 0x47, 0xde, 0x9e, 0x23, 0xd6, 0x17, 0xa8, 0x0e, 0x75, 0x52, 0xd1, 0x42, 0x1f, 0xd8, 0x93, 0xbc, 0xf2, 0xa7, 0x4e, 0xc2, 0x2f, 0x05, 0xf1, 0x2d, 0xfc, 0x76, 0xaa, 0x83, 0x37, 0x7c, 0x1a, 0x31, 0x09, 0x54, 0xdd, 0xb6, 0x20, 0x28, 0xfd, 0x6f, 0xd5, 0x9c, 0xa9, 0xcf, 0x0b, 0x8a, 0x81, 0xa5, 0xb5, 0xf3, 0x01, 0x45, 0xb1],
    ];
    const MONO_CODE3_60MS: &[&[u8]] = &[
        &[0x7b, 0x83, 0x27, 0x27, 0x82, 0x02, 0xb0, 0x35, 0xed, 0x11, 0xad, 0xa5, 0x17, 0xe6, 0x7f, 0x8f, 0x87, 0xad, 0x63, 0x65, 0x38, 0x0b, 0x91, 0x53, 0x0d, 0x78, 0xde, 0xf4, 0xc5, 0xd4, 0x80, 0x81, 0x3b, 0x4b, 0xab, 0x38, 0x52, 0x5b, 0x09, 0x7f, 0x17, 0x1b, 0x79, 0xaa, 0x7e, 0x84, 0x2f, 0x24, 0x6b, 0xa0, 0xbc, 0xfd, 0x0a, 0x5b, 0xb9, 0x1d, 0x64, 0xfc, 0x53, 0xcc, 0x60, 0x3c, 0x99, 0x6c, 0x66, 0x38, 0x1e, 0x5b, 0x74, 0x51, 0x1c, 0xe5, 0x39, 0x44, 0x4e, 0xba, 0x2b, 0x66, 0xb2, 0xb4, 0xd2, 0x06, 0x9f, 0x6d, 0x4e, 0x97, 0x56, 0x68, 0x72, 0x72, 0xdc, 0xa2, 0x63, 0x5c, 0x47, 0x6c, 0x9a, 0x78, 0xde, 0x47, 0x6e, 0xbe, 0x0e, 0xb8, 0x4e, 0x73, 0x0c, 0x42, 0x45, 0xa5, 0x40, 0x16, 0x94, 0x29, 0x8b],
    ];
    const MONO_CELT_10MS: &[&[u8]] = &[
        &[0xf0, 0xb4, 0xcd, 0x66, 0xcc, 0xcb, 0xa8, 0xe4, 0xa4, 0x4b, 0x77, 0x2a, 0xfd, 0x0c, 0x8f, 0xb9, 0xb8, 0x29, 0x30, 0x4a, 0xb4, 0x8d, 0x09, 0xbd, 0x99, 0x19, 0xb0, 0xc8, 0xed, 0xd3, 0xb1, 0x82, 0xcd, 0x45, 0x49, 0x24, 0x13, 0xa8, 0x02, 0x55, 0xed, 0x28, 0x1a, 0x0f, 0x70, 0x77, 0x00, 0x5b, 0xfd, 0xd8, 0x99, 0x7b, 0x14, 0x89, 0xb8, 0x50, 0x96, 0xbd, 0x84, 0x98, 0xf0, 0x72, 0xa6, 0x32, 0xbb, 0x21, 0xaf, 0xa6, 0xca, 0xc1, 0x8b, 0x18, 0x6d, 0x4f, 0xd1, 0x16, 0x14, 0xe1, 0x49, 0x94, 0x3f, 0x54, 0x27, 0x99, 0x9c, 0x23, 0xff, 0x3b, 0xc7, 0xbf, 0x53, 0x0e, 0x18, 0x1e, 0x4c, 0x17, 0xef, 0x6d, 0x27, 0xf7, 0x9d, 0x5e, 0xc8, 0xed, 0x0f, 0x54, 0x0b, 0x9d, 0x14, 0xbf, 0xa8, 0xb9, 0x72, 0x6b, 0x3e, 0x0c, 0x5a, 0xd4, 0xd0, 0x9f, 0x47, 0x87, 0x6d, 0x25, 0xc2, 0x9f, 0xa9, 0x63, 0x9e, 0xea, 0xa1, 0x16, 0x99, 0xa1, 0x25, 0xd6, 0x7f, 0x84, 0x11, 0xde, 0x81, 0xeb, 0xde, 0x56, 0xd3, 0x64, 0x85, 0xca, 0xdb, 0x95, 0x1f, 0x27, 0x34, 0x31, 0x9f, 0x3c, 0x77, 0x6d],
        &[0xf0, 0xb3, 0x8b, 0x65, 0x03, 0x2f, 0xa0, 0x43, 0x16, 0x98, 0x92, 0x38, 0x19, 0x4e, 0x50, 0x2b, 0x86, 0x08, 0x36, 0x63, 0xb1, 0x41, 0xec, 0xed, 0xae, 0x10, 0x5b, 0x9f, 0xea, 0x59, 0xdb, 0x9b, 0x37, 0x2d, 0x8c, 0xc6, 0x38, 0x4b, 0x9a, 0x21, 0x9a, 0x7f, 0x90, 0xd7, 0xfb, 0x13, 0xb4, 0xfd, 0xbe, 0x38, 0x5f, 0x15, 0xb8, 0xc9, 0x3c, 0x10, 0x4e, 0x43, 0x5c, 0x91, 0x63, 0x6f, 0xe7, 0x27, 0xed, 0xae],
    ];
    const MONO_CELT_5MS: &[&[u8]] = &[
        &[0xe8, 0xb5, 0x34, 0x27, 0x9d, 0xd6, 0x3d, 0xfd, 0x6a, 0x91, 0xc0, 0xea, 0x6b, 0x56, 0x74, 0x99, 0xca, 0xbc, 0x57, 0x8d, 0x8f, 0x15, 0xfb, 0xdc, 0x45, 0x17, 0xd5, 0xd4, 0xae, 0x69, 0xc2, 0x53, 0x3a, 0x9a, 0x72, 0x5f, 0x1b, 0x28, 0xc9, 0xd4, 0xc0, 0xda, 0x55, 0xc9, 0x52, 0xf0, 0x6d, 0xc6, 0x88, 0x0e, 0xce, 0x5b, 0x6e, 0x98, 0x37, 0x47, 0xd6, 0xed, 0xcc, 0x53, 0x41, 0xbf, 0xd9, 0x85, 0x5e, 0x01, 0x6f, 0x2c, 0x74, 0x35, 0x2c],
        &[0xe8, 0xb3, 0x10, 0xa2, 0x14, 0x17, 0x6b, 0x67, 0xbe, 0x9c, 0x54, 0x83, 0xc5, 0x29, 0xf3, 0x21, 0xcb, 0xdb, 0xea, 0xc4, 0x78, 0x11, 0x93, 0x64, 0xd8, 0x52, 0x51, 0xed, 0x9b, 0xf2, 0x1f, 0xc4, 0x03, 0x2b, 0x2f, 0xdb, 0x6e],
    ];
    const MONO_CELT_2_5MS: &[&[u8]] = &[
        &[0xe0, 0x73, 0xf6, 0x70, 0x46, 0x04, 0xc8, 0x84, 0x73, 0x77, 0x5f, 0xb0, 0x74, 0x6d, 0xe1, 0xb0, 0x0f, 0x51, 0xa7, 0x3a, 0x67, 0xba, 0x98, 0x50, 0xfc, 0xbd, 0x94, 0x02],
        &[0xe0, 0xb4, 0x44, 0xd2, 0x0f, 0x30, 0xd6, 0xdb, 0x01, 0x69, 0xf3, 0xf7, 0xb4, 0xc7, 0x56, 0x3e, 0xdf, 0x4e, 0x1d, 0x45, 0xae],
    ];
    const SURROUND51_20MS_HEAD: &[u8] = &[79, 112, 117, 115, 72, 101, 97, 100, 1, 6, 56, 1, 128, 187, 0, 0, 0, 0, 1, 4, 2, 0, 4, 1, 2, 3, 5];
    const SURROUND51_20MS: &[&[u8]] = &[
        &[0xfc, 0x02, 0xff, 0xfe, 0xfc, 0x02, 0xff, 0xfe, 0x78, 0x5c, 0x81, 0x7b, 0xc6, 0x11, 0x76, 0xf4, 0x75, 0x00, 0x00, 0x07, 0x8e, 0xca, 0x69, 0xdb, 0xbb, 0xcc, 0x48, 0xdc, 0x16, 0xbd, 0x9b, 0xd3, 0x58, 0x27, 0x78, 0x93, 0xbc, 0x37, 0x44, 0x5e, 0xcf, 0x2d, 0x97, 0x00, 0x69, 0x80, 0x11, 0x80, 0x2c, 0x03, 0x38, 0xab, 0x45, 0x61, 0xfd, 0x89, 0xb7, 0x7e, 0x0d, 0x17, 0xbf, 0x20, 0xa5, 0x2b, 0xb5, 0xd8, 0x89, 0x9b, 0x71, 0x04, 0x2a, 0x3b, 0x68, 0x89, 0xb8, 0x5a, 0xef, 0x37, 0x35, 0xa4, 0x0f, 0x35, 0x4b, 0xc7, 0x72, 0x05, 0x78, 0x74, 0xff, 0xc1, 0xe3, 0x95, 0x9b, 0x67, 0x7e, 0xcb, 0x20, 0xd5, 0xff, 0x17, 0x1b, 0x02, 0x98, 0xff, 0xfe],
        &[0xfc, 0x02, 0xff, 0xfe, 0xfc, 0x02, 0xff, 0xfe, 0x78, 0x34, 0x9e, 0x1e, 0xa1, 0xe7, 0xfc, 0x95, 0x4d, 0x4e, 0xaa, 0x18, 0x01, 0xbc, 0x6b, 0xfc, 0x27, 0xeb, 0xac, 0xa4, 0xee, 0xeb, 0xe4, 0xd6, 0x01, 0x11, 0x26, 0x0b, 0xc0, 0xae, 0xfb, 0x24, 0x73, 0x57, 0xd4, 0xe5, 0x70, 0xdf, 0xc1, 0x94, 0xdd, 0xc4, 0xa4, 0xa5, 0x71, 0xe2, 0x96, 0xe1, 0x6b, 0x31, 0x09, 0xd1, 0xb7, 0xa6, 0x98, 0xff, 0xfe],
    ];

    fn goertzel_energy(samples: &[f64], rate: f64, freq_hz: f64) -> f64 {
        let n = samples.len();
        let k = (0.5 + (n as f64 * freq_hz) / rate).floor();
        let w = (2.0 * std::f64::consts::PI * k) / n as f64;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0, 0.0);
        for &x in samples {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt() / (n as f64)
    }

    #[test]
    fn decode_silk_hybrid_celt_all_frame_sizes() {
        if !is_available() {
            eprintln!("skipping: MSOpusDecoder.dll not available");
            return;
        }
        let cases: &[(&str, &[&[u8]], usize)] = &[
            ("Hybrid FB 20ms", MONO_HYBRID_20MS, 960),
            ("SILK WB 40ms", MONO_SILK_40MS, 1920),
            ("Hybrid code-3 60ms", MONO_CODE3_60MS, 2880),
            ("CELT FB 10ms", MONO_CELT_10MS, 480),
            ("CELT FB 5ms", MONO_CELT_5MS, 240),
            ("CELT FB 2.5ms", MONO_CELT_2_5MS, 120),
        ];
        for &(name, pkts, expected_per_pkt) in cases {
            let mut dec = Decoder::new(1, 48_000, &[]).expect(name);
            let mut pcm = Vec::new();
            for &pkt in pkts {
                let n = dec.decode_packet(pkt, &mut pcm).expect(name);
                assert_eq!(n, expected_per_pkt, "{name}: frame count");
            }
            let peak = pcm.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
            assert!(peak > 2000, "{name}: decoded PCM looks silent (peak={peak})");
        }
        // Verify 440 Hz tone accuracy on the 100 ms Hybrid mono stream
        let mut dec = Decoder::new(1, 48_000, &[]).unwrap();
        let mut pcm = Vec::new();
        for &pkt in MONO_HYBRID_20MS {
            dec.decode_packet(pkt, &mut pcm).unwrap();
        }
        let f: Vec<f64> = pcm.iter().skip(480).map(|&s| s as f64 / 32768.0).collect();
        let e440 = goertzel_energy(&f, 48_000.0, 440.0);
        let e880 = goertzel_energy(&f, 48_000.0, 880.0);
        assert!(e440 > 0.04 && e440 > e880 * 10.0, "440 Hz vs 880 Hz: {e440} vs {e880}");
    }

    #[test]
    fn decode_stereo_and_multistream_5_1() {
        if !is_available() {
            eprintln!("skipping: MSOpusDecoder.dll not available");
            return;
        }
        // Stereo: 440 Hz on left, 880 Hz on right
        let mut dec = Decoder::new(2, 48_000, &[]).unwrap();
        let mut pcm = Vec::new();
        for &pkt in STEREO_CELT_20MS {
            assert_eq!(dec.decode_packet(pkt, &mut pcm).unwrap(), 960);
        }
        let left: Vec<f64> = pcm.chunks_exact(2).skip(480).map(|c| c[0] as f64 / 32768.0).collect();
        let right: Vec<f64> = pcm.chunks_exact(2).skip(480).map(|c| c[1] as f64 / 32768.0).collect();
        assert!(goertzel_energy(&left, 48_000.0, 440.0) > goertzel_energy(&left, 48_000.0, 880.0) * 10.0);
        assert!(goertzel_energy(&right, 48_000.0, 880.0) > goertzel_energy(&right, 48_000.0, 440.0) * 10.0);

        // 5.1 surround (6 channels, mapping family 1): channel 1 (FC) carries the 440 Hz tone
        let mut dec51 = Decoder::new(6, 48_000, SURROUND51_20MS_HEAD).unwrap();
        let mut pcm51 = Vec::new();
        let f0 = dec51.decode_packet(SURROUND51_20MS[0], &mut pcm51).unwrap();
        let f1 = dec51.decode_packet(SURROUND51_20MS[1], &mut pcm51).unwrap();
        // OpusHead declares pre_skip = 312 -> first packet trimmed by 312 samples
        assert_eq!((f0, f1), (960 - 312, 960));
        let peak_ch = |c: usize| pcm51.chunks_exact(6).map(|fr| fr[c].unsigned_abs()).max().unwrap_or(0);
        assert!(peak_ch(1) > 2000, "center channel should carry the tone");
        assert!(peak_ch(0) < 50 && peak_ch(5) < 50, "silent channels should remain near zero");
    }

    #[test]
    fn decode_ogg_eos_matches_raw_pcm_window_mono_and_stereo() {
        if !is_available() {
            eprintln!("skipping: MSOpusDecoder.dll not available");
            return;
        }
        let cases = [
            (1, &MONO_HYBRID_20MS[..2], 0, 0),
            (2, &STEREO_CELT_20MS[..2], 312, 5000),
        ];
        for (channels, packets, pre_skip, initial_granule) in cases {
            let head = build_opus_head(channels, 48_000, pre_skip, 0);
            let mut ogg = ogg_headers(&head, 7);
            push_ogg_packets(&mut ogg, 7, 2, 0, initial_granule + 960, &packets[..1]);
            push_ogg_packets(&mut ogg, 7, 3, 0x04, initial_granule + 1724, &packets[1..]);

            let mut raw_decoder = Decoder::new(u16::from(channels), 48_000, &[]).unwrap();
            let mut raw = Vec::new();
            for pkt in packets {
                let pcm = raw_decoder.decode_buffer(pkt, 0).unwrap();
                assert_eq!(pcm.len(), 960 * usize::from(channels));
                raw.extend(pcm);
            }
            let ch = usize::from(channels);
            for extra in [&[][..], head.as_slice()] {
                let mut dec = Decoder::new(u16::from(channels), 48_000, extra).unwrap();
                let pcm = dec.decode_buffer(&ogg, 0).unwrap();
                assert_eq!(pcm.as_slice(), &raw[usize::from(pre_skip) * ch..1724 * ch]);
                assert_eq!(
                    estimate_buffer_frames(&ogg, 0, 48_000, false),
                    Some((pcm.len() / ch) as u64),
                );
            }
        }
    }

    #[test]
    fn decode_awc_block_preskip_rate_and_flush() {
        if !is_available() {
            eprintln!("skipping: MSOpusDecoder.dll not available");
            return;
        }
        // Pack 5 mono 20 ms packets into a 2048-byte zero-padded AWC block
        let mut block = vec![0u8; 2048];
        let mut pos = 0;
        for &pkt in MONO_HYBRID_20MS {
            block[pos..pos + 2].copy_from_slice(&(pkt.len() as u16).to_le_bytes());
            block[pos + 2..pos + 2 + pkt.len()].copy_from_slice(pkt);
            pos += 2 + pkt.len();
        }
        let mut dec = Decoder::new(1, 48_000, MONO_HYBRID_20MS_HEAD).unwrap();
        let pcm = dec.decode_buffer(&block, 2048).unwrap();
        assert_eq!(pcm.len(), 5 * 960 - 312);

        // Flush resets pre-skip and decoder state cleanly
        dec.flush();
        let pcm2 = dec.decode_buffer(&block, 2048).unwrap();
        assert_eq!(pcm2.len(), 5 * 960 - 312);

        // 24 kHz output rate: 20 ms = 480 samples
        let mut d24 = Decoder::new(1, 24_000, &[]).unwrap();
        assert_eq!(d24.output_rate(), 24_000);
        let mut out24 = Vec::new();
        assert_eq!(d24.decode_packet(MONO_HYBRID_20MS[0], &mut out24).unwrap(), 480);

        // Invalid packet returns Err without breaking subsequent valid decodes
        assert!(d24.decode_packet(&[0xFB, 0], &mut out24).is_err());
        assert_eq!(d24.decode_packet(MONO_HYBRID_20MS[1], &mut out24).unwrap(), 480);
    }
}
