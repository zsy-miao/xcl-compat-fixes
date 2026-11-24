//! xaudio2_9.dll – console XAudio2 on top of the Windows XAudio2 2.9.
//!
//! The console API is the Windows 2.9 API plus compressed source voices (XMA2, Opus) and a
//! few console-only flags. `XAudio2Create` / `XAudio2CreateWithVersionInfo` create the
//! genuine Windows engine with console-only flags dropped and patch its vtables **in place**
//! (the Windows classes have internal virtual methods after the public ones, so a vtable
//! copy of the public part is not safe):
//!
//! * IXAudio2 slot 5 `CreateSourceVoice` – Opus formats (`WAVE_FORMAT_OPUS` 0x704F or
//!   `WAVE_FORMAT_EXTENSIBLE` with `MFAudioFormat_Opus`) become PCM16 voices backed by a
//!   real RFC 6716 / RFC 7845 decoder (`hle::opus`, using Windows' `MSOpusDecoder.dll`);
//!   XMA/XMA2 (`0x165` / `0x166`) become PCM16 voices backed by the pure-Rust XMA decoder
//!   (`hle::xma2`); any other format Windows rejects becomes a silent PCM16 voice of
//!   matching length,
//! * IXAudio2 slot 7 `CreateMasteringVoice` – console device ids are ignored (default
//!   device); on failure retried with default channels/rate and the game-effects category,
//! * IXAudio2SourceVoice slot 21 `SubmitSourceBuffer` – Opus and XMA/XMA2 voices decode
//!   the submitted packets to PCM16 and keep each decoded buffer alive until `OnBufferEnd`
//!   (or voice destruction); silent voices submit silence of matching length, preserving
//!   callbacks,
//! * IXAudio2SourceVoice slot 22 `FlushSourceBuffers` – flushes the Opus / XMA decoder
//!   state,
//! * IXAudio2SourceVoice slot 25 `GetState` – maps internal buffer tokens back to the
//!   title's original `pBufferContext`,
//! * IXAudio2SourceVoice slot 18 `DestroyVoice` – tears down the voice, then releases any
//!   remaining decoded buffers and callback wrapper.
//!
//! Exports we do not implement (X3DAudio*, CreateFX, CreateAudioReverb…) fall back to the
//! system xaudio2_9.dll (see `hle::build`).

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::hle::{opus, xma2};
use crate::util::{E_FAIL, E_POINTER, HRESULT, S_OK};

pub const WAVE_FORMAT_PCM: u16 = 1;
pub const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
pub const WAVE_FORMAT_XMA: u16 = 0x165;
pub const WAVE_FORMAT_XMA2: u16 = 0x166;
pub const WAVE_FORMAT_OPUS: u16 = 0x704F;
pub const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// `MFAudioFormat_Opus` / `KSDATAFORMAT_SUBTYPE_OPUS`: `{0000704F-0000-0010-8000-00AA00389B71}`.
pub const SUBFORMAT_OPUS: [u8; 16] = [
    0x4F, 0x70, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// XAudio2Create flags Windows accepts: STOP_ENGINE_WHEN_IDLE, 1024_QUANTUM, NO_VIRTUAL_AUDIO_CLIENT.
pub const ENGINE_FLAGS: u32 = 0x2000 | 0x8000 | 0x1_0000;
/// CreateSourceVoice flags Windows accepts: NOPITCH, NOSRC, USEFILTER.
pub const VOICE_FLAGS: u32 = 0x2 | 0x4 | 0x8;
/// CreateMasteringVoice flags Windows accepts: NO_VIRTUAL_AUDIO_CLIENT.
pub const MASTERING_FLAGS: u32 = 0x1_0000;
const XAUDIO2_DEFAULT_PROCESSOR: u32 = 1;
const NTDDI_WIN10_RS5: u32 = 0x0A00_0006;
const AUDIO_CATEGORY_GAME_EFFECTS: u32 = 6;
const XAUDIO2_END_OF_STREAM: u32 = 0x40;
const CO_E_NOTINITIALIZED: HRESULT = 0x8004_01F0u32 as i32;

const ENGINE_CREATE_SOURCE_VOICE: usize = 5;
const ENGINE_CREATE_MASTERING_VOICE: usize = 7;
const VOICE_DESTROY: usize = 18;
const VOICE_SUBMIT_SOURCE_BUFFER: usize = 21;
const VOICE_FLUSH_SOURCE_BUFFERS: usize = 22;
const VOICE_GET_STATE: usize = 25;

/// Size of the shared silence buffer (zero pages are only backed once read).
const SILENCE_BYTES: usize = 32 << 20;

#[repr(C, packed)]
#[derive(Clone, Copy, Default, Debug)]
pub struct WaveFormatEx {
    pub tag: u16,
    pub channels: u16,
    pub rate: u32,
    pub avg_bytes: u32,
    pub block_align: u16,
    pub bits: u16,
    pub cb_size: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default, Debug)]
pub struct WaveFormatExtensible {
    pub wfx: WaveFormatEx,
    pub samples: u16,
    pub channel_mask: u32,
    pub sub_format: [u8; 16],
}

/// XMA2WAVEFORMATEX (xma2defs.h, packed).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Xma2WaveFormatEx {
    pub wfx: WaveFormatEx,
    pub num_streams: u16,
    pub channel_mask: u32,
    pub samples_encoded: u32,
    pub bytes_per_block: u32,
    pub play_begin: u32,
    pub play_length: u32,
    pub loop_begin: u32,
    pub loop_length: u32,
    pub loop_count: u8,
    pub encoder_version: u8,
    pub block_count: u16,
}

/// XAUDIO2_BUFFER (packed).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct XAudio2Buffer {
    pub flags: u32,
    pub audio_bytes: u32,
    pub audio_data: *const u8,
    pub play_begin: u32,
    pub play_length: u32,
    pub loop_begin: u32,
    pub loop_length: u32,
    pub loop_count: u32,
    pub context: *mut c_void,
}

/// XAUDIO2_VOICE_STATE (packed).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct XAudio2VoiceState {
    pub current_buffer_context: *mut c_void,
    pub buffers_queued: u32,
    pub samples_played: u64,
}

const _: () = assert!(std::mem::size_of::<WaveFormatEx>() == 18);
const _: () = assert!(std::mem::size_of::<WaveFormatExtensible>() == 40);
const _: () = assert!(std::mem::size_of::<Xma2WaveFormatEx>() == 52);
const _: () = assert!(std::mem::size_of::<XAudio2Buffer>() == 44);
const _: () = assert!(std::mem::size_of::<XAudio2VoiceState>() == 20);

type CreateWithVersionFn = unsafe extern "system" fn(*mut *mut c_void, u32, u32, u32) -> HRESULT;
type CreateFn = unsafe extern "system" fn(*mut *mut c_void, u32, u32) -> HRESULT;
type CreateSourceVoiceFn = unsafe extern "system" fn(
    *mut c_void,
    *mut *mut c_void,
    *const WaveFormatEx,
    u32,
    f32,
    *mut c_void,
    *const c_void,
    *const c_void,
) -> HRESULT;
type CreateMasteringVoiceFn =
    unsafe extern "system" fn(*mut c_void, *mut *mut c_void, u32, u32, u32, *const u16, *const c_void, u32) -> HRESULT;
type DestroyVoiceFn = unsafe extern "system" fn(*mut c_void);
type SubmitSourceBufferFn = unsafe extern "system" fn(*mut c_void, *const XAudio2Buffer, *const c_void) -> HRESULT;
type FlushSourceBuffersFn = unsafe extern "system" fn(*mut c_void) -> HRESULT;
type GetStateFn = unsafe extern "system" fn(*mut c_void, *mut XAudio2VoiceState, u32);

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![XAudio2Create, XAudio2CreateWithVersionInfo]
}

// ---------------------------------------------------------------------------------------
// statistics (report.json → audio)
// ---------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioStats {
    pub engines: u64,
    pub engine_errors: u64,
    pub mastering_voices: u64,
    pub mastering_retries: u64,
    pub mastering_errors: u64,
    pub source_voices: u64,
    /// XMA/XMA2 voices decoded to PCM16 via `hle::xma2`
    pub xma_voices: u64,
    /// XMA/XMA2 buffers decoded to PCM16
    pub xma_buffers_decoded: u64,
    /// XMA/XMA2 buffers that failed decoding (fell back to silence)
    pub xma_decode_errors: u64,
    /// Opus voices decoded to PCM16 via MSOpusDecoder
    pub opus_voices: u64,
    /// Opus buffers decoded to PCM16
    pub opus_buffers_decoded: u64,
    /// Opus buffers that failed decoding (fell back to silence)
    pub opus_decode_errors: u64,
    /// other formats the Windows engine rejected, replaced the same way (tags, e.g. "0x161")
    pub unsupported_formats: Vec<String>,
    pub silent_voices: u64,
    pub buffers_silenced: u64,
    pub source_voice_errors: u64,
    /// console-only flags we dropped, e.g. "XAudio2Create:0x1"
    pub dropped_flags: Vec<String>,
    /// the console's decoder (xapu, `hle::acphal`): Opus packets decoded, packets that failed (silence
    /// went out), frames produced, samples that are not 0
    pub xapu_packets_decoded: u64,
    pub xapu_decode_errors: u64,
    pub xapu_frames: u64,
    pub xapu_nonzero_samples: u64,
}

static ENGINES: AtomicU64 = AtomicU64::new(0);
static ENGINE_ERRORS: AtomicU64 = AtomicU64::new(0);
static MASTERING: AtomicU64 = AtomicU64::new(0);
static MASTERING_RETRIES: AtomicU64 = AtomicU64::new(0);
static MASTERING_ERRORS: AtomicU64 = AtomicU64::new(0);
static SOURCE_VOICES: AtomicU64 = AtomicU64::new(0);
static XMA_VOICES: AtomicU64 = AtomicU64::new(0);
static XMA_BUFFERS_DECODED: AtomicU64 = AtomicU64::new(0);
static XMA_DECODE_ERRORS: AtomicU64 = AtomicU64::new(0);
static OPUS_VOICES_CREATED: AtomicU64 = AtomicU64::new(0);
static OPUS_BUFFERS_DECODED: AtomicU64 = AtomicU64::new(0);
static OPUS_DECODE_ERRORS: AtomicU64 = AtomicU64::new(0);
static SILENT_VOICES: AtomicU64 = AtomicU64::new(0);
static SILENCED: AtomicU64 = AtomicU64::new(0);
static UNSUPPORTED: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SOURCE_ERRORS: AtomicU64 = AtomicU64::new(0);
static DROPPED: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn stats() -> AudioStats {
    AudioStats {
        engines: ENGINES.load(Ordering::Relaxed),
        engine_errors: ENGINE_ERRORS.load(Ordering::Relaxed),
        mastering_voices: MASTERING.load(Ordering::Relaxed),
        mastering_retries: MASTERING_RETRIES.load(Ordering::Relaxed),
        mastering_errors: MASTERING_ERRORS.load(Ordering::Relaxed),
        source_voices: SOURCE_VOICES.load(Ordering::Relaxed),
        xma_voices: XMA_VOICES.load(Ordering::Relaxed),
        xma_buffers_decoded: XMA_BUFFERS_DECODED.load(Ordering::Relaxed),
        xma_decode_errors: XMA_DECODE_ERRORS.load(Ordering::Relaxed),
        opus_voices: OPUS_VOICES_CREATED.load(Ordering::Relaxed),
        opus_buffers_decoded: OPUS_BUFFERS_DECODED.load(Ordering::Relaxed),
        opus_decode_errors: OPUS_DECODE_ERRORS.load(Ordering::Relaxed),
        unsupported_formats: UNSUPPORTED.lock().map(|d| d.clone()).unwrap_or_default(),
        silent_voices: SILENT_VOICES.load(Ordering::Relaxed),
        buffers_silenced: SILENCED.load(Ordering::Relaxed),
        source_voice_errors: SOURCE_ERRORS.load(Ordering::Relaxed),
        dropped_flags: DROPPED.lock().map(|d| d.clone()).unwrap_or_default(),
        xapu_packets_decoded: PCM_PACKETS.load(Ordering::Relaxed),
        xapu_decode_errors: PCM_PACKET_ERRORS.load(Ordering::Relaxed),
        xapu_frames: PCM_DECODED_FRAMES.load(Ordering::Relaxed),
        xapu_nonzero_samples: PCM_NONZERO.load(Ordering::Relaxed),
    }
}

fn note_dropped(what: &str, flags: u32, allowed: u32) {
    let extra = flags & !allowed;
    if extra == 0 {
        return;
    }
    let key = format!("{what}:{extra:#x}");
    let mut d = DROPPED.lock().unwrap();
    if !d.contains(&key) {
        crate::xlog_info!("XAudio2: {what} – dropping console flags {extra:#x}");
        d.push(key);
    }
}

// ---------------------------------------------------------------------------------------
// pure helpers (unit tested)
// ---------------------------------------------------------------------------------------

/// What we remember about a silent voice (XMA or another format Windows cannot play)
/// to size its buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XmaInfo {
    pub channels: u16,
    pub rate: u32,
    /// WAVEFORMATEX.nAvgBytesPerSec of the original format
    pub avg_bytes: u32,
    pub block_align: u16,
    pub samples_encoded: u32,
    pub bytes_per_block: u32,
    pub block_count: u16,
    pub is_opus: bool,
}

/// Parsed fields of an Opus `WAVEFORMATEX` / `WAVEFORMATEXTENSIBLE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusFormat<'a> {
    pub channels: u16,
    pub rate: u32,
    pub block_align: u16,
    pub avg_bytes: u32,
    pub extra: &'a [u8],
}

pub fn is_xma(tag: u16) -> bool {
    tag == WAVE_FORMAT_XMA || tag == WAVE_FORMAT_XMA2
}

/// Formats the Windows engine decodes itself – never replaced.
pub fn is_plain_pcm(tag: u16) -> bool {
    matches!(tag, WAVE_FORMAT_PCM | WAVE_FORMAT_IEEE_FLOAT | WAVE_FORMAT_EXTENSIBLE)
}

/// Recognize an Opus format (`WAVE_FORMAT_OPUS` 0x704F or `WAVE_FORMAT_EXTENSIBLE` with
/// `MFAudioFormat_Opus`) and return its header fields + codec-private extra bytes.
///
/// # Safety
/// `fmt` must point to a `WAVEFORMATEX` followed by `cbSize` readable bytes.
pub unsafe fn opus_format<'a>(fmt: *const WaveFormatEx) -> Option<OpusFormat<'a>> {
    let w = fmt.read_unaligned();
    let channels = w.channels.clamp(1, 8);
    let rate = if w.rate == 0 { 48_000 } else { w.rate };
    if w.tag == WAVE_FORMAT_OPUS {
        let extra = if w.cb_size > 0 {
            std::slice::from_raw_parts((fmt as *const u8).add(std::mem::size_of::<WaveFormatEx>()), w.cb_size as usize)
        } else {
            &[]
        };
        return Some(OpusFormat { channels, rate, block_align: w.block_align, avg_bytes: w.avg_bytes, extra });
    }
    if w.tag == WAVE_FORMAT_EXTENSIBLE && w.cb_size >= 22 {
        let ext = (fmt as *const WaveFormatExtensible).read_unaligned();
        if ext.sub_format == SUBFORMAT_OPUS {
            let extra_len = (w.cb_size - 22) as usize;
            let extra = if extra_len > 0 {
                std::slice::from_raw_parts((fmt as *const u8).add(std::mem::size_of::<WaveFormatExtensible>()), extra_len)
            } else {
                &[]
            };
            return Some(OpusFormat { channels, rate, block_align: w.block_align, avg_bytes: w.avg_bytes, extra });
        }
    }
    None
}

/// Parse an XMA1 (`0x0165`) or XMA2 (`0x0166`) `WAVEFORMATEX` + extradata into a decoder config.
///
/// # Safety
/// `fmt` must point to a `WAVEFORMATEX` followed by `cbSize` readable bytes.
pub unsafe fn xma_stream_config(fmt: *const WaveFormatEx) -> Option<Result<xma2::StreamConfig, String>> {
    let w = fmt.read_unaligned();
    if !is_xma(w.tag) {
        return None;
    }
    let extra = if w.cb_size > 0 {
        std::slice::from_raw_parts((fmt as *const u8).add(std::mem::size_of::<WaveFormatEx>()), w.cb_size as usize)
    } else {
        &[]
    };
    Some(if w.tag == WAVE_FORMAT_XMA2 {
        xma2::StreamConfig::from_xma2_wfx(w.channels, w.rate, extra)
    } else {
        xma2::StreamConfig::from_xma1_wfx(w.channels, w.rate, extra)
    })
}

/// Voice info from a format (XMA2WAVEFORMATEX fields only when cbSize says they are there).
///
/// # Safety
/// `fmt` must point to a WAVEFORMATEX followed by `cbSize` readable bytes.
pub unsafe fn xma_info(fmt: *const WaveFormatEx) -> XmaInfo {
    let w = fmt.read_unaligned();
    let mut i = XmaInfo {
        channels: w.channels.clamp(1, 8),
        rate: if w.rate == 0 { 48_000 } else { w.rate },
        avg_bytes: w.avg_bytes,
        block_align: w.block_align,
        is_opus: opus_format(fmt).is_some(),
        ..Default::default()
    };
    if w.tag == WAVE_FORMAT_XMA2 && w.cb_size as usize >= std::mem::size_of::<Xma2WaveFormatEx>() - 18 {
        let x = (fmt as *const Xma2WaveFormatEx).read_unaligned();
        i.samples_encoded = x.samples_encoded;
        i.bytes_per_block = x.bytes_per_block;
        i.block_count = x.block_count;
    }
    i
}

/// PCM16 format with the channel count and sample rate of a voice.
pub fn pcm16_for_rate(channels: u16, rate: u32) -> WaveFormatEx {
    let ch = channels.clamp(1, 8);
    let block = ch * 2;
    WaveFormatEx {
        tag: WAVE_FORMAT_PCM,
        channels: ch,
        rate,
        avg_bytes: rate * block as u32,
        block_align: block,
        bits: 16,
        cb_size: 0,
    }
}

/// PCM16 format with the channel count and sample rate of an XMA voice.
pub fn pcm16_for(i: &XmaInfo) -> WaveFormatEx {
    pcm16_for_rate(i.channels, i.rate)
}

/// Clamp play/loop sample regions to `total_frames` so XAudio2 never rejects a decoded
/// buffer with `XAUDIO2_E_INVALID_CALL`.
pub fn clamp_regions(b: &mut XAudio2Buffer, total_frames: u32) {
    if total_frames == 0 {
        b.play_begin = 0;
        b.play_length = 0;
        b.loop_begin = 0;
        b.loop_length = 0;
        b.loop_count = 0;
        return;
    }
    if b.play_begin >= total_frames {
        b.play_begin = 0;
        b.play_length = 0;
    } else if b.play_length == 0 {
        if b.play_begin > 0 {
            b.play_length = total_frames - b.play_begin;
        }
    } else if b.play_length > total_frames - b.play_begin {
        b.play_length = total_frames - b.play_begin;
    }
    if b.loop_count == 0 {
        b.loop_begin = 0;
        b.loop_length = 0;
        return;
    }
    let play_end = if b.play_length > 0 {
        b.play_begin + b.play_length
    } else {
        total_frames
    };
    if b.loop_begin >= play_end {
        b.loop_begin = 0;
        b.loop_length = 0;
        b.loop_count = 0;
    } else if b.loop_length == 0 {
        if b.loop_begin > 0 || play_end < total_frames {
            b.loop_length = play_end - b.loop_begin;
        }
    } else {
        let max_loop = play_end - b.loop_begin;
        if b.loop_length > max_loop {
            b.loop_length = max_loop;
        }
        if b.loop_begin.saturating_add(b.loop_length) <= b.play_begin {
            b.loop_begin = 0;
            b.loop_length = 0;
            b.loop_count = 0;
        }
    }
}

/// Frames (samples per channel) a compressed buffer of `audio_bytes` decodes to: XMA2 in
/// proportion to SamplesEncoded over the whole stream (BlockCount × BytesPerBlock), else from
/// the format's average byte rate, else ~4 frames per byte per channel (≈ 8:1 against PCM16).
pub fn estimate_frames(i: &XmaInfo, audio_bytes: u32) -> u64 {
    let total_bytes = i.block_count as u64 * i.bytes_per_block as u64;
    let f = if i.samples_encoded > 0 && total_bytes > 0 {
        i.samples_encoded as u64 * audio_bytes as u64 / total_bytes
    } else if i.avg_bytes > 0 {
        audio_bytes as u64 * i.rate as u64 / i.avg_bytes as u64
    } else {
        audio_bytes as u64 * 4 / i.channels.max(1) as u64
    };
    f.max(128)
}

const XAUDIO2_MAX_BUFFER_BYTES: u64 = 0x8000_0000;

/// The PCM16 silence buffer that stands in for an XMA (or fallback) buffer: same play/loop
/// region and flags/context; length from PlayBegin+PlayLength (or RFC 6716 Opus packet
/// durations / XMA2 estimate), clamped to `max_bytes`.
pub fn silent_buffer(i: &XmaInfo, b: &XAudio2Buffer, silence: *const u8, max_bytes: usize) -> XAudio2Buffer {
    let opus_est = if i.is_opus && !b.audio_data.is_null() && b.audio_bytes > 0 {
        let data = unsafe { std::slice::from_raw_parts(b.audio_data, b.audio_bytes as usize) };
        opus::estimate_buffer_frames(data, i.block_align, i.rate, i.channels > 2)
    } else {
        None
    };
    let mut nb = *b;
    let (pb, pl, lb, ll) = (b.play_begin as u64, b.play_length as u64, b.loop_begin as u64, b.loop_length as u64);
    let mut frames = if pl > 0 {
        pb + pl
    } else {
        opus_est.unwrap_or_else(|| estimate_frames(i, b.audio_bytes)).max(pb + 1)
    };
    if b.loop_count > 0 {
        frames = frames.max(lb + if ll == 0 && lb > 0 { 1 } else { ll });
    }
    let frame_bytes = i.channels.max(1) as u64 * 2;
    let max_frames = (max_bytes as u64).min(XAUDIO2_MAX_BUFFER_BYTES) / frame_bytes;
    if frames > max_frames {
        frames = max_frames;
        nb.play_begin = 0;
        nb.play_length = 0;
        nb.loop_begin = 0;
        nb.loop_length = 0;
    }
    clamp_regions(&mut nb, frames as u32);
    nb.audio_bytes = (frames * frame_bytes) as u32;
    nb.audio_data = silence;
    nb
}

// ---------------------------------------------------------------------------------------
// Decoded voice state (Opus & XMA1/XMA2) & IXAudio2VoiceCallback wrapper
// ---------------------------------------------------------------------------------------

/// Owned decoded PCM16 buffer kept alive at a stable heap address until `OnBufferEnd`
/// or `DestroyVoice`.
struct DecodedBuffer {
    orig_context: usize,
    _pcm: Box<[u8]>,
}

enum VoiceDecoder {
    Opus(opus::Decoder),
    Xma { decoder: xma2::Decoder, info: XmaInfo },
}

impl VoiceDecoder {
    fn flush(&mut self) {
        match self {
            VoiceDecoder::Opus(d) => d.flush(),
            VoiceDecoder::Xma { decoder, .. } => decoder.flush(),
        }
    }
}

struct DecodedVoiceState {
    decoder: Mutex<VoiceDecoder>,
    block_align: u16,
    channels: u16,
    rate: u32,
    avg_bytes: u32,
    title_cb: usize,
    /// Maps stable token (`&*DecodedBuffer as usize`) → owned `DecodedBuffer`.
    live_buffers: Mutex<HashMap<usize, Box<DecodedBuffer>>>,
}

struct DecodedVoiceHandle {
    state: Arc<DecodedVoiceState>,
    cb_ptr: usize,
}

// IXAudio2VoiceCallback vtable (7 methods, no IUnknown).
#[repr(C)]
struct VoiceCallbackVtbl {
    pass_start: unsafe extern "system" fn(*mut c_void, u32),
    pass_end: unsafe extern "system" fn(*mut c_void),
    stream_end: unsafe extern "system" fn(*mut c_void),
    buffer_start: unsafe extern "system" fn(*mut c_void, *mut c_void),
    buffer_end: unsafe extern "system" fn(*mut c_void, *mut c_void),
    loop_end: unsafe extern "system" fn(*mut c_void, *mut c_void),
    voice_error: unsafe extern "system" fn(*mut c_void, *mut c_void, HRESULT),
}

#[repr(C)]
struct DecodedCallback {
    vtbl: &'static VoiceCallbackVtbl,
    state: Arc<DecodedVoiceState>,
}

unsafe fn title_cb_vtbl(cb: usize) -> Option<(*mut c_void, *const VoiceCallbackVtbl)> {
    if cb == 0 {
        return None;
    }
    let obj = cb as *mut c_void;
    let vtbl = *(obj as *const *const VoiceCallbackVtbl);
    if vtbl.is_null() {
        return None;
    }
    Some((obj, vtbl))
}

unsafe extern "system" fn decoded_cb_pass_start(this: *mut c_void, bytes: u32) {
    let cb = &*(this as *const DecodedCallback);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).pass_start)(obj, bytes);
    }
}

unsafe extern "system" fn decoded_cb_pass_end(this: *mut c_void) {
    let cb = &*(this as *const DecodedCallback);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).pass_end)(obj);
    }
}

unsafe extern "system" fn decoded_cb_stream_end(this: *mut c_void) {
    let cb = &*(this as *const DecodedCallback);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).stream_end)(obj);
    }
}

unsafe extern "system" fn decoded_cb_buffer_start(this: *mut c_void, ctx: *mut c_void) {
    let cb = &*(this as *const DecodedCallback);
    let orig = cb
        .state
        .live_buffers
        .lock()
        .unwrap()
        .get(&(ctx as usize))
        .map(|b| b.orig_context as *mut c_void)
        .unwrap_or(ctx);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).buffer_start)(obj, orig);
    }
}

unsafe extern "system" fn decoded_cb_buffer_end(this: *mut c_void, ctx: *mut c_void) {
    let cb = &*(this as *const DecodedCallback);
    // Remove and drop the decoded PCM buffer before invoking the title's callback so
    // the lock is released and no double-free or deadlock can occur.
    let removed = cb.state.live_buffers.lock().unwrap().remove(&(ctx as usize));
    let orig = removed.as_ref().map(|b| b.orig_context as *mut c_void).unwrap_or(ctx);
    drop(removed);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).buffer_end)(obj, orig);
    }
}

unsafe extern "system" fn decoded_cb_loop_end(this: *mut c_void, ctx: *mut c_void) {
    let cb = &*(this as *const DecodedCallback);
    let orig = cb
        .state
        .live_buffers
        .lock()
        .unwrap()
        .get(&(ctx as usize))
        .map(|b| b.orig_context as *mut c_void)
        .unwrap_or(ctx);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).loop_end)(obj, orig);
    }
}

unsafe extern "system" fn decoded_cb_voice_error(this: *mut c_void, ctx: *mut c_void, hr: HRESULT) {
    let cb = &*(this as *const DecodedCallback);
    let orig = cb
        .state
        .live_buffers
        .lock()
        .unwrap()
        .get(&(ctx as usize))
        .map(|b| b.orig_context as *mut c_void)
        .unwrap_or(ctx);
    if let Some((obj, vt)) = title_cb_vtbl(cb.state.title_cb) {
        ((*vt).voice_error)(obj, orig, hr);
    }
}

static DECODED_CALLBACK_VTBL: VoiceCallbackVtbl = VoiceCallbackVtbl {
    pass_start: decoded_cb_pass_start,
    pass_end: decoded_cb_pass_end,
    stream_end: decoded_cb_stream_end,
    buffer_start: decoded_cb_buffer_start,
    buffer_end: decoded_cb_buffer_end,
    loop_end: decoded_cb_loop_end,
    voice_error: decoded_cb_voice_error,
};

// ---------------------------------------------------------------------------------------
// the system engine and in-place vtable patches
// ---------------------------------------------------------------------------------------

/// Originals of patched slots: (vtable address, slot) → function.
static ORIGINALS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
/// Live silent voices (voice pointer → info).
static XMA: Mutex<Option<HashMap<usize, XmaInfo>>> = Mutex::new(None);
/// Live decoded voices (Opus & XMA1/XMA2: voice pointer → handle).
static DECODED_VOICES: Mutex<Option<HashMap<usize, DecodedVoiceHandle>>> = Mutex::new(None);
/// Count of live decoded voices (fast path for plain PCM voices).
static DECODED_LIVE_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe fn vtable(obj: *mut c_void) -> *mut usize {
    *(obj as *const *mut usize)
}

/// Patch `slot` of `obj`'s vtable in place (once per vtable).
unsafe fn patch(obj: *mut c_void, slot: usize, hook: usize) {
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_READWRITE};
    let vt = vtable(obj);
    let p = vt.add(slot);
    let cur = p.read();
    if cur == hook {
        return;
    }
    let mut g = ORIGINALS.lock().unwrap();
    if g.iter().any(|&(v, s, _)| v == vt as usize && s == slot) {
        return;
    }
    let mut old = 0u32;
    if VirtualProtect(p as *const c_void, 8, PAGE_READWRITE, &mut old) == 0 {
        crate::xlog_error!("XAudio2: cannot make vtable {vt:p} writable (slot {slot})");
        return;
    }
    g.push((vt as usize, slot, cur));
    p.write(hook);
    VirtualProtect(p as *const c_void, 8, old, &mut old);
}

/// The original implementation of `slot` for `obj`'s class.
unsafe fn original(obj: *mut c_void, slot: usize) -> Option<usize> {
    let vt = vtable(obj) as usize;
    ORIGINALS.lock().unwrap().iter().find(|&&(v, s, _)| v == vt && s == slot).map(|&(_, _, f)| f)
}

fn system_create() -> Option<(Option<CreateWithVersionFn>, Option<CreateFn>)> {
    static F: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    let &(a, b) = F
        .get_or_init(|| {
            let h = crate::resolver::system_module("xaudio2_9.dll")?;
            let name = |n: &str| crate::resolver::system_proc_in(h, &crate::pe::ImportName::Name { hint: 0, name: n.into() }).unwrap_or(0);
            Some((name("XAudio2CreateWithVersionInfo"), name("XAudio2Create")))
        })
        .as_ref()?;
    unsafe {
        Some((
            (a != 0).then(|| std::mem::transmute::<usize, CreateWithVersionFn>(a)),
            (b != 0).then(|| std::mem::transmute::<usize, CreateFn>(b)),
        ))
    }
}

unsafe fn create_engine(pp: *mut *mut c_void, flags: u32, processor: u32) -> HRESULT {
    if pp.is_null() {
        return E_POINTER;
    }
    note_dropped("XAudio2Create", flags, ENGINE_FLAGS);
    let _ = processor; // console core masks mean nothing here
    let Some((with_version, plain)) = system_create() else {
        crate::xlog_error!("XAudio2: no system xaudio2_9.dll");
        ENGINE_ERRORS.fetch_add(1, Ordering::Relaxed);
        return E_FAIL;
    };
    let clean = flags & ENGINE_FLAGS;
    let hr = match (with_version, plain) {
        (Some(f), _) => f(pp, clean, XAUDIO2_DEFAULT_PROCESSOR, NTDDI_WIN10_RS5),
        (None, Some(f)) => f(pp, clean, XAUDIO2_DEFAULT_PROCESSOR),
        _ => E_FAIL,
    };
    if hr < 0 || (*pp).is_null() {
        crate::xlog_error!("XAudio2: creating the engine failed ({:#x})", hr as u32);
        ENGINE_ERRORS.fetch_add(1, Ordering::Relaxed);
        return if hr < 0 { hr } else { E_FAIL };
    }
    patch(*pp, ENGINE_CREATE_SOURCE_VOICE, create_source_voice as usize);
    patch(*pp, ENGINE_CREATE_MASTERING_VOICE, create_mastering_voice as usize);
    ENGINES.fetch_add(1, Ordering::Relaxed);
    crate::xlog_info!("XAudio2: engine created (system XAudio2 2.9, flags {clean:#x})");
    S_OK
}

pub unsafe extern "system" fn XAudio2Create(pp: *mut *mut c_void, flags: u32, processor: u32) -> HRESULT {
    create_engine(pp, flags, processor)
}

pub unsafe extern "system" fn XAudio2CreateWithVersionInfo(pp: *mut *mut c_void, flags: u32, processor: u32, _ntddi: u32) -> HRESULT {
    create_engine(pp, flags, processor)
}

unsafe fn create_decoded_voice(
    orig: CreateSourceVoiceFn,
    this: *mut c_void,
    pp: *mut *mut c_void,
    flags: u32,
    max_ratio: f32,
    cb: *mut c_void,
    sends: *const c_void,
    chain: *const c_void,
    decoder: VoiceDecoder,
    channels: u16,
    rate: u32,
    block_align: u16,
    avg_bytes: u32,
    label: &str,
) -> HRESULT {
    let pcm = pcm16_for_rate(channels, rate);
    let state = Arc::new(DecodedVoiceState {
        decoder: Mutex::new(decoder),
        block_align,
        channels: pcm.channels,
        rate: pcm.rate,
        avg_bytes,
        title_cb: cb as usize,
        live_buffers: Mutex::new(HashMap::new()),
    });
    let wrapped_cb = Box::into_raw(Box::new(DecodedCallback {
        vtbl: &DECODED_CALLBACK_VTBL,
        state: Arc::clone(&state),
    }));
    let hr = orig(this, pp, &pcm, flags, max_ratio, wrapped_cb as *mut c_void, sends, chain);
    if hr < 0 || (*pp).is_null() {
        drop(Box::from_raw(wrapped_cb));
        SOURCE_ERRORS.fetch_add(1, Ordering::Relaxed);
        crate::xlog_warn!("XAudio2: CreateSourceVoice({label}→PCM16) failed ({:#x})", hr as u32);
        return hr;
    }
    let voice = *pp;
    patch_source_voice(voice);
    DECODED_VOICES
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(voice as usize, DecodedVoiceHandle { state, cb_ptr: wrapped_cb as usize });
    DECODED_LIVE_COUNT.fetch_add(1, Ordering::Release);
    SOURCE_VOICES.fetch_add(1, Ordering::Relaxed);
    hr
}

unsafe extern "system" fn create_source_voice(
    this: *mut c_void,
    pp: *mut *mut c_void,
    fmt: *const WaveFormatEx,
    flags: u32,
    max_ratio: f32,
    cb: *mut c_void,
    sends: *const c_void,
    chain: *const c_void,
) -> HRESULT {
    let Some(orig) = original(this, ENGINE_CREATE_SOURCE_VOICE) else { return E_FAIL };
    let orig: CreateSourceVoiceFn = std::mem::transmute(orig);
    note_dropped("CreateSourceVoice", flags, VOICE_FLAGS);
    let flags = flags & VOICE_FLAGS;
    if pp.is_null() || fmt.is_null() {
        return orig(this, pp, fmt, flags, max_ratio, cb, sends, chain);
    }
    let tag = (*fmt).tag;

    // 1. Opus voice (WAVE_FORMAT_OPUS 0x704F or WAVE_FORMAT_EXTENSIBLE with MFAudioFormat_Opus)
    if let Some(of) = opus_format(fmt) {
        match opus::Decoder::new(of.channels, of.rate, of.extra) {
            Ok(dec) => {
                let pcm_ch = dec.channels();
                let pcm_rate = dec.output_rate();
                let hr = create_decoded_voice(
                    orig,
                    this,
                    pp,
                    flags,
                    max_ratio,
                    cb,
                    sends,
                    chain,
                    VoiceDecoder::Opus(dec),
                    pcm_ch,
                    pcm_rate,
                    of.block_align,
                    of.avg_bytes,
                    "Opus",
                );
                if hr >= 0 && OPUS_VOICES_CREATED.fetch_add(1, Ordering::Relaxed) == 0 {
                    crate::xlog_info!("XAudio2: Opus voice ({pcm_ch} ch., {pcm_rate} Hz) → real PCM16 decoding via MSOpusDecoder");
                }
                return hr;
            }
            Err(e) => {
                crate::xlog_warn!("XAudio2: Opus decoder unavailable ({e}) – falling back to silence");
            }
        }
    }

    // 2. XMA / XMA2 voice (WAVE_FORMAT_XMA 0x0165 or WAVE_FORMAT_XMA2 0x0166)
    if let Some(cfg_res) = xma_stream_config(fmt) {
        let info = xma_info(fmt);
        match cfg_res.and_then(xma2::Decoder::new) {
            Ok(dec) => {
                let pcm_ch = dec.channels();
                let pcm_rate = dec.sample_rate();
                let label = if tag == WAVE_FORMAT_XMA2 { "XMA2" } else { "XMA" };
                let hr = create_decoded_voice(
                    orig,
                    this,
                    pp,
                    flags,
                    max_ratio,
                    cb,
                    sends,
                    chain,
                    VoiceDecoder::Xma { decoder: dec, info },
                    pcm_ch,
                    pcm_rate,
                    info.block_align,
                    info.avg_bytes,
                    label,
                );
                if hr >= 0 && XMA_VOICES.fetch_add(1, Ordering::Relaxed) == 0 {
                    crate::xlog_info!("XAudio2: {label} voice ({pcm_ch} ch., {pcm_rate} Hz) → real PCM16 decoding via hle::xma2");
                }
                return hr;
            }
            Err(e) => {
                crate::xlog_warn!("XAudio2: XMA decoder init failed ({e}) – falling back to silence");
            }
        }
    }

    // 3. Fallback for unknown compressed formats or unavailable decoder
    let mut silent = (is_xma(tag) || opus_format(fmt).is_some()).then(|| xma_info(fmt));
    let mut hr = match &silent {
        Some(i) => orig(this, pp, &pcm16_for(i), flags, max_ratio, cb, sends, chain),
        None => orig(this, pp, fmt, flags, max_ratio, cb, sends, chain),
    };
    if hr < 0 && silent.is_none() && !is_plain_pcm(tag) {
        // a compressed format the Windows engine does not know
        let i = xma_info(fmt);
        let h2 = orig(this, pp, &pcm16_for(&i), flags, max_ratio, cb, sends, chain);
        if h2 >= 0 {
            let key = format!("{tag:#x}");
            let mut u = UNSUPPORTED.lock().unwrap();
            if !u.contains(&key) {
                crate::xlog_warn!(
                    "XAudio2: format {key} not supported by Windows ({:#x}) → PCM16 voice ({} ch., {} Hz) playing silence",
                    hr as u32,
                    i.channels,
                    i.rate
                );
                u.push(key);
            }
            hr = h2;
            silent = Some(i);
        }
    }
    if hr < 0 || (*pp).is_null() {
        SOURCE_ERRORS.fetch_add(1, Ordering::Relaxed);
        crate::xlog_warn!("XAudio2: CreateSourceVoice (format {tag:#x}) failed ({:#x})", hr as u32);
        return hr;
    }
    let voice = *pp;
    patch_source_voice(voice);
    SOURCE_VOICES.fetch_add(1, Ordering::Relaxed);
    if let Some(i) = silent {
        XMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(voice as usize, i);
        SILENT_VOICES.fetch_add(1, Ordering::Relaxed);
        if is_xma(tag) && XMA_VOICES.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!(
                "XAudio2: XMA{} voice ({} ch., {} Hz) → PCM16; no XMA decoder – buffers play silence of the same length",
                if tag == WAVE_FORMAT_XMA2 { "2" } else { "" },
                i.channels,
                i.rate
            );
        }
    }
    hr
}

unsafe fn patch_source_voice(voice: *mut c_void) {
    patch(voice, VOICE_SUBMIT_SOURCE_BUFFER, submit_source_buffer as usize);
    patch(voice, VOICE_FLUSH_SOURCE_BUFFERS, flush_source_buffers as usize);
    patch(voice, VOICE_GET_STATE, get_state as usize);
    patch(voice, VOICE_DESTROY, destroy_voice as usize);
}

unsafe extern "system" fn create_mastering_voice(
    this: *mut c_void,
    pp: *mut *mut c_void,
    channels: u32,
    rate: u32,
    flags: u32,
    device: *const u16,
    chain: *const c_void,
    category: u32,
) -> HRESULT {
    let Some(orig) = original(this, ENGINE_CREATE_MASTERING_VOICE) else { return E_FAIL };
    let orig: CreateMasteringVoiceFn = std::mem::transmute(orig);
    note_dropped("CreateMasteringVoice", flags, MASTERING_FLAGS);
    if !device.is_null() {
        crate::xlog_info!("XAudio2: CreateMasteringVoice – console device id ignored (default device)");
    }
    let flags = flags & MASTERING_FLAGS;
    let mut hr = orig(this, pp, channels, rate, flags, std::ptr::null(), chain, category);
    if hr == CO_E_NOTINITIALIZED {
        // the console needs no COM for audio; Windows' device enumeration does
        let ci = windows_sys::Win32::System::Com::CoInitializeEx(std::ptr::null(), windows_sys::Win32::System::Com::COINIT_MULTITHREADED as u32);
        crate::xlog_info!("XAudio2: COM was not initialized on the title thread – CoInitializeEx(MTA) → {:#x}", ci as u32);
        hr = orig(this, pp, channels, rate, flags, std::ptr::null(), chain, category);
    }
    if hr < 0 {
        MASTERING_RETRIES.fetch_add(1, Ordering::Relaxed);
        crate::xlog_warn!(
            "XAudio2: CreateMasteringVoice({channels} ch., {rate} Hz, category {category}) → {:#x}; retrying with defaults",
            hr as u32
        );
        hr = orig(this, pp, 0, 0, flags, std::ptr::null(), chain, AUDIO_CATEGORY_GAME_EFFECTS);
        if hr < 0 && !chain.is_null() {
            hr = orig(this, pp, 0, 0, flags, std::ptr::null(), std::ptr::null(), AUDIO_CATEGORY_GAME_EFFECTS);
        }
    }
    if hr < 0 {
        MASTERING_ERRORS.fetch_add(1, Ordering::Relaxed);
        crate::xlog_error!("XAudio2: CreateMasteringVoice failed ({:#x}) – no audio device?", hr as u32);
    } else {
        MASTERING.fetch_add(1, Ordering::Relaxed);
    }
    hr
}

fn silence() -> *const u8 {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READONLY};
    static P: OnceLock<usize> = OnceLock::new();
    *P.get_or_init(|| unsafe { VirtualAlloc(std::ptr::null(), SILENCE_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY) as usize }) as *const u8
}

unsafe fn submit_decoded_buffer(
    this: *mut c_void,
    orig: SubmitSourceBufferFn,
    state: &DecodedVoiceState,
    orig_buf: XAudio2Buffer,
) -> HRESULT {
    let data = if !orig_buf.audio_data.is_null() && orig_buf.audio_bytes > 0 {
        std::slice::from_raw_parts(orig_buf.audio_data, orig_buf.audio_bytes as usize)
    } else {
        &[]
    };
    let eos = (orig_buf.flags & XAUDIO2_END_OF_STREAM) != 0;

    let (decoded_pcm, fallback_info, is_xma_voice) = {
        let mut dec = state.decoder.lock().unwrap();
        match &mut *dec {
            VoiceDecoder::Opus(d) => {
                let res = d.decode_buffer(data, state.block_align);
                if eos {
                    d.flush();
                }
                let info = XmaInfo {
                    channels: state.channels,
                    rate: state.rate,
                    avg_bytes: state.avg_bytes,
                    block_align: state.block_align,
                    is_opus: true,
                    ..Default::default()
                };
                (res, info, false)
            }
            VoiceDecoder::Xma { decoder, info } => {
                let res = decoder.decode_buffer_i16(data, eos);
                (res, *info, true)
            }
        }
    };

    let mut nb = orig_buf;
    let pcm_bytes: Box<[u8]> = match decoded_pcm {
        Ok(samples) if !samples.is_empty() => {
            let total_frames = (samples.len() / state.channels.max(1) as usize) as u32;
            clamp_regions(&mut nb, total_frames);
            let mut bytes = Vec::with_capacity(samples.len() * 2);
            for s in samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            if is_xma_voice {
                XMA_BUFFERS_DECODED.fetch_add(1, Ordering::Relaxed);
            } else {
                OPUS_BUFFERS_DECODED.fetch_add(1, Ordering::Relaxed);
            }
            bytes.into_boxed_slice()
        }
        res => {
            let err_counter = if is_xma_voice { &XMA_DECODE_ERRORS } else { &OPUS_DECODE_ERRORS };
            let codec_name = if is_xma_voice { "XMA" } else { "Opus" };
            let prev = err_counter.fetch_add(1, Ordering::Relaxed);
            if prev == 0 {
                match res {
                    Err(e) => crate::xlog_warn!("XAudio2: {codec_name} decode error ({e}) – substituting silence"),
                    Ok(_) => crate::xlog_warn!("XAudio2: {codec_name} buffer decoded 0 samples – substituting silence"),
                }
            }
            let sb = silent_buffer(&fallback_info, &orig_buf, std::ptr::null(), SILENCE_BYTES);
            nb = sb;
            SILENCED.fetch_add(1, Ordering::Relaxed);
            vec![0u8; sb.audio_bytes as usize].into_boxed_slice()
        }
    };

    let pcm_ptr = pcm_bytes.as_ptr();
    let pcm_len = pcm_bytes.len() as u32;
    let boxed = Box::new(DecodedBuffer { orig_context: orig_buf.context as usize, _pcm: pcm_bytes });
    let token = &*boxed as *const DecodedBuffer as usize;
    state.live_buffers.lock().unwrap().insert(token, boxed);

    nb.audio_data = pcm_ptr;
    nb.audio_bytes = pcm_len;
    nb.context = token as *mut c_void;

    let hr = orig(this, &nb, std::ptr::null());
    if hr < 0 {
        state.live_buffers.lock().unwrap().remove(&token);
    }
    hr
}

unsafe extern "system" fn submit_source_buffer(this: *mut c_void, buf: *const XAudio2Buffer, wma: *const c_void) -> HRESULT {
    let Some(orig) = original(this, VOICE_SUBMIT_SOURCE_BUFFER) else { return E_FAIL };
    let orig: SubmitSourceBufferFn = std::mem::transmute(orig);
    if !buf.is_null() && DECODED_LIVE_COUNT.load(Ordering::Acquire) != 0 {
        let state = DECODED_VOICES
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(&(this as usize)).map(|h| Arc::clone(&h.state)));
        if let Some(state) = state {
            return submit_decoded_buffer(this, orig, &state, buf.read_unaligned());
        }
    }
    let info = XMA.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied());
    match info {
        Some(i) if !buf.is_null() => {
            let s = silence();
            if s.is_null() {
                return E_FAIL;
            }
            let nb = silent_buffer(&i, &buf.read_unaligned(), s, SILENCE_BYTES);
            SILENCED.fetch_add(1, Ordering::Relaxed);
            orig(this, &nb, std::ptr::null())
        }
        _ => orig(this, buf, wma),
    }
}

unsafe extern "system" fn flush_source_buffers(this: *mut c_void) -> HRESULT {
    let Some(orig) = original(this, VOICE_FLUSH_SOURCE_BUFFERS) else { return E_FAIL };
    let orig: FlushSourceBuffersFn = std::mem::transmute(orig);
    let hr = orig(this);
    if hr >= 0 && DECODED_LIVE_COUNT.load(Ordering::Acquire) != 0 {
        let state = DECODED_VOICES
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(&(this as usize)).map(|h| Arc::clone(&h.state)));
        if let Some(state) = state {
            state.decoder.lock().unwrap().flush();
        }
    }
    hr
}

unsafe extern "system" fn get_state(this: *mut c_void, state_out: *mut XAudio2VoiceState, flags: u32) {
    let Some(orig) = original(this, VOICE_GET_STATE) else { return };
    let orig: GetStateFn = std::mem::transmute(orig);
    orig(this, state_out, flags);
    if !state_out.is_null() && DECODED_LIVE_COUNT.load(Ordering::Acquire) != 0 {
        let voice_state = DECODED_VOICES
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(&(this as usize)).map(|h| Arc::clone(&h.state)));
        if let Some(voice_state) = voice_state {
            let cur = (*state_out).current_buffer_context as usize;
            if cur != 0 {
                if let Some(buf) = voice_state.live_buffers.lock().unwrap().get(&cur) {
                    (*state_out).current_buffer_context = buf.orig_context as *mut c_void;
                }
            }
        }
    }
}

unsafe extern "system" fn destroy_voice(this: *mut c_void) {
    let Some(orig) = original(this, VOICE_DESTROY) else { return };
    let orig: DestroyVoiceFn = std::mem::transmute(orig);
    if let Some(m) = XMA.lock().unwrap().as_mut() {
        m.remove(&(this as usize));
    }
    let decoded_handle = if DECODED_LIVE_COUNT.load(Ordering::Acquire) != 0 {
        DECODED_VOICES.lock().unwrap().as_mut().and_then(|m| m.remove(&(this as usize)))
    } else {
        None
    };
    if decoded_handle.is_some() {
        DECODED_LIVE_COUNT.fetch_sub(1, Ordering::Release);
    }
    // Destroy the Windows voice first so its audio thread stops using the callback and any
    // queued PCM buffers before we free them.
    orig(this);
    if let Some(h) = decoded_handle {
        h.state.live_buffers.lock().unwrap().clear();
        drop(Box::from_raw(h.cb_ptr as *mut DecodedCallback));
    }
}

// ---------------------------------------------------------------------------------------
// the console audio processor's decoder (hle::acphal): what it did, for report.json
// ---------------------------------------------------------------------------------------

static PCM_DECODED_FRAMES: AtomicU64 = AtomicU64::new(0);
static PCM_NONZERO: AtomicU64 = AtomicU64::new(0);
static PCM_PACKETS: AtomicU64 = AtomicU64::new(0);
static PCM_PACKET_ERRORS: AtomicU64 = AtomicU64::new(0);

/// What the console decoder (xapu) did: `packets` Opus packets decoded to `frames` frames of which
/// `nonzero` samples are not 0, `errors` packets that did not decode (silence went out).
pub fn note_console_decode(packets: u64, errors: u64, frames: u64, nonzero: u64) {
    PCM_PACKETS.fetch_add(packets, Ordering::Relaxed);
    PCM_PACKET_ERRORS.fetch_add(errors, Ordering::Relaxed);
    PCM_DECODED_FRAMES.fetch_add(frames, Ordering::Relaxed);
    PCM_NONZERO.fetch_add(nonzero, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xma2(channels: u16, rate: u32, samples: u32, bpb: u32, blocks: u16) -> Xma2WaveFormatEx {
        Xma2WaveFormatEx {
            wfx: WaveFormatEx { tag: WAVE_FORMAT_XMA2, channels, rate, avg_bytes: 0, block_align: 1, bits: 16, cb_size: 34 },
            samples_encoded: samples,
            bytes_per_block: bpb,
            block_count: blocks,
            ..Default::default()
        }
    }

    fn buffer(bytes: u32, pb: u32, pl: u32, lb: u32, ll: u32, lc: u32) -> XAudio2Buffer {
        XAudio2Buffer {
            flags: 0x40,
            audio_bytes: bytes,
            audio_data: 0x1000 as *const u8,
            play_begin: pb,
            play_length: pl,
            loop_begin: lb,
            loop_length: ll,
            loop_count: lc,
            context: 0x1234 as *mut c_void,
        }
    }

    #[test]
    fn xma2_format_to_pcm16() {
        let f = xma2(2, 44_100, 441_000, 0x8000, 4);
        let i = unsafe { xma_info(&f as *const Xma2WaveFormatEx as *const WaveFormatEx) };
        assert_eq!(
            i,
            XmaInfo {
                channels: 2,
                rate: 44_100,
                avg_bytes: 0,
                block_align: 1,
                samples_encoded: 441_000,
                bytes_per_block: 0x8000,
                block_count: 4,
                is_opus: false,
            }
        );
        let p = pcm16_for(&i);
        assert_eq!((p.tag, p.channels, p.rate, p.block_align, p.bits, p.avg_bytes), (1, 2, 44_100, 4, 16, 176_400));
        // plain XMA (or a short cbSize): no XMA2 fields are read
        let mut w = f.wfx;
        w.cb_size = 0;
        assert_eq!(unsafe { xma_info(&w) }.samples_encoded, 0);
        assert!(is_xma(0x166) && is_xma(0x165) && !is_xma(1));
    }

    #[test]
    fn opus_format_detection_and_region_clamping() {
        // Plain WAVE_FORMAT_OPUS (0x704F) with OpusHead extra data
        let head = opus::build_opus_head(2, 48_000, 312, 0);
        let mut raw = Vec::with_capacity(18 + head.len());
        let wfx = WaveFormatEx { tag: WAVE_FORMAT_OPUS, channels: 2, rate: 48_000, avg_bytes: 16_000, block_align: 2048, bits: 0, cb_size: head.len() as u16 };
        raw.extend_from_slice(unsafe { std::slice::from_raw_parts(&wfx as *const _ as *const u8, 18) });
        raw.extend_from_slice(&head);
        let of = unsafe { opus_format(raw.as_ptr() as *const WaveFormatEx) }.unwrap();
        assert_eq!((of.channels, of.rate, of.block_align, of.extra), (2, 48_000, 2048, head.as_slice()));

        // WAVE_FORMAT_EXTENSIBLE with MFAudioFormat_Opus subformat
        let ext = WaveFormatExtensible {
            wfx: WaveFormatEx { tag: WAVE_FORMAT_EXTENSIBLE, channels: 6, rate: 48_000, avg_bytes: 0, block_align: 0, bits: 0, cb_size: 22 },
            samples: 0,
            channel_mask: 0x3F,
            sub_format: SUBFORMAT_OPUS,
        };
        let of2 = unsafe { opus_format(&ext as *const _ as *const WaveFormatEx) }.unwrap();
        assert_eq!((of2.channels, of2.rate, of2.extra.len()), (6, 48_000, 0));

        // Region clamping
        let mut b = buffer(100, 200, 5000, 100, 4000, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }), (200, 760, 100, 860));
        let mut b2 = buffer(100, 1000, 100, 1000, 100, 1);
        clamp_regions(&mut b2, 960);
        assert_eq!(({ b2.play_begin }, { b2.play_length }, { b2.loop_count }), (0, 0, 0));
    }

    #[test]
    fn silence_has_the_length_of_the_xma_buffer() {
        let i = XmaInfo { channels: 2, rate: 48_000, avg_bytes: 0, samples_encoded: 480_000, bytes_per_block: 0x10000, block_count: 2, ..Default::default() };
        let z = 0x5000 as *const u8;
        // explicit play region: PlayBegin + PlayLength frames
        let b = silent_buffer(&i, &buffer(4096, 128, 4800, 0, 0, 0), z, 1 << 20);
        assert_eq!({ b.audio_bytes }, (128 + 4800) * 4);
        assert_eq!(({ b.play_begin }, { b.play_length }, { b.flags }, b.audio_data), (128, 4800, 0x40, z));
        assert_eq!({ b.context } as usize, 0x1234);
        // whole buffer: in proportion to SamplesEncoded (1/32 of the stream here)
        let b = silent_buffer(&i, &buffer(0x1000, 0, 0, 0, 0, 0), z, 1 << 20);
        assert_eq!({ b.audio_bytes } / 4, 480_000 / 32);
        // the loop region must fit
        let b = silent_buffer(&i, &buffer(0x1000, 0, 100, 0, 9000, 255), z, 1 << 20);
        assert_eq!({ b.audio_bytes } / 4, 9000);
        // too long for the silence buffer → clamped, regions reset
        let b = silent_buffer(&i, &buffer(0x1000, 1000, 1_000_000, 0, 0, 0), z, 4000);
        assert_eq!(({ b.audio_bytes }, { b.play_begin }, { b.play_length }), (4000, 0, 0));
        // no XMA2 metadata: rough ratio, never empty
        let j = XmaInfo { channels: 1, rate: 48_000, ..Default::default() };
        assert_eq!(estimate_frames(&j, 2048), 8192);
        assert_eq!(estimate_frames(&j, 0), 128);
        // other compressed formats: from the average byte rate (Opus at 64 kbit/s: 8000 B/s)
        let o = XmaInfo { channels: 1, rate: 48_000, avg_bytes: 8000, ..Default::default() };
        assert_eq!(estimate_frames(&o, 8000), 48_000);
        assert!(is_plain_pcm(1) && is_plain_pcm(3) && is_plain_pcm(0xFFFE) && !is_plain_pcm(0x704F));
    }

    #[test]
    fn engine_from_the_system_dll() {
        // the genuine Windows engine is created and patched; console flags are dropped
        unsafe {
            let mut eng: *mut c_void = std::ptr::null_mut();
            let hr = XAudio2Create(&mut eng, 0x1 | 0x8000, 0xff);
            assert_eq!(hr, S_OK, "{:#x}", hr as u32);
            assert!(!eng.is_null());
            let vt = vtable(eng);
            assert_eq!(vt.add(ENGINE_CREATE_SOURCE_VOICE).read(), create_source_voice as usize);
            assert!(original(eng, ENGINE_CREATE_SOURCE_VOICE).is_some());
            // a second engine shares the vtable: no double patch, originals kept
            let mut eng2: *mut c_void = std::ptr::null_mut();
            assert_eq!(XAudio2CreateWithVersionInfo(&mut eng2, 0, 1, 0), S_OK);
            assert_eq!(ORIGINALS.lock().unwrap().iter().filter(|o| o.1 == ENGINE_CREATE_SOURCE_VOICE).count(), 1);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vtable(eng2).add(2).read());
            release(eng2);
            release(eng);
            assert!(stats().dropped_flags.iter().any(|f| f == "XAudio2Create:0x1"));
        }
    }

    #[test]
    fn opus_source_voice_decodes_and_manages_lifetime() {
        if !opus::is_available() {
            eprintln!("skipping: MSOpusDecoder.dll not available");
            return;
        }
        use std::sync::atomic::AtomicUsize;

        static ENDED: AtomicUsize = AtomicUsize::new(0);
        static STARTED: AtomicUsize = AtomicUsize::new(0);
        static STREAM_END: AtomicUsize = AtomicUsize::new(0);

        unsafe extern "system" fn on_pass_start(_: *mut c_void, _: u32) {}
        unsafe extern "system" fn on_nothing(_: *mut c_void) {}
        unsafe extern "system" fn on_stream_end(_: *mut c_void) {
            STREAM_END.fetch_add(1, Ordering::SeqCst);
        }
        unsafe extern "system" fn on_buf_start(_: *mut c_void, ctx: *mut c_void) {
            STARTED.fetch_add(ctx as usize, Ordering::SeqCst);
        }
        unsafe extern "system" fn on_buf_end(_: *mut c_void, ctx: *mut c_void) {
            ENDED.fetch_add(ctx as usize, Ordering::SeqCst);
        }
        unsafe extern "system" fn on_loop(_: *mut c_void, _: *mut c_void) {}
        unsafe extern "system" fn on_err(_: *mut c_void, _: *mut c_void, _: HRESULT) {}

        static TEST_VTBL: VoiceCallbackVtbl = VoiceCallbackVtbl {
            pass_start: on_pass_start,
            pass_end: on_nothing,
            stream_end: on_stream_end,
            buffer_start: on_buf_start,
            buffer_end: on_buf_end,
            loop_end: on_loop,
            voice_error: on_err,
        };
        #[repr(C)]
        struct TestCb {
            vtbl: &'static VoiceCallbackVtbl,
        }
        static TEST_CB: TestCb = TestCb { vtbl: &TEST_VTBL };

        unsafe {
            let mut eng: *mut c_void = std::ptr::null_mut();
            assert_eq!(XAudio2Create(&mut eng, 0, 1), S_OK);
            let vt_eng = vtable(eng);
            let create_master: CreateMasteringVoiceFn = std::mem::transmute(vt_eng.add(ENGINE_CREATE_MASTERING_VOICE).read());
            let mut master: *mut c_void = std::ptr::null_mut();
            if create_master(eng, &mut master, 0, 0, 0, std::ptr::null(), std::ptr::null(), AUDIO_CATEGORY_GAME_EFFECTS) < 0 {
                let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vt_eng.add(2).read());
                release(eng);
                eprintln!("skipping: no audio device");
                return;
            }

            let create_src: CreateSourceVoiceFn = std::mem::transmute(vt_eng.add(ENGINE_CREATE_SOURCE_VOICE).read());
            let opus_wfx = WaveFormatEx { tag: WAVE_FORMAT_OPUS, channels: 1, rate: 48_000, avg_bytes: 4000, block_align: 2048, bits: 0, cb_size: 0 };
            let mut voice: *mut c_void = std::ptr::null_mut();
            let hr = create_src(eng, &mut voice, &opus_wfx, 0, 2.0, &TEST_CB as *const _ as *mut c_void, std::ptr::null(), std::ptr::null());
            assert_eq!(hr, S_OK, "CreateSourceVoice(Opus) → {:#x}", hr as u32);
            assert!(!voice.is_null());

            // Pack 2 packets into a 2048 B AWC block + 1 raw packet
            let mut block = vec![0u8; 2048];
            let mut pos = 0;
            for &pkt in &opus::tests::MONO_HYBRID_20MS[..2] {
                block[pos..pos + 2].copy_from_slice(&(pkt.len() as u16).to_le_bytes());
                block[pos + 2..pos + 2 + pkt.len()].copy_from_slice(pkt);
                pos += 2 + pkt.len();
            }

            let vt_v = vtable(voice);
            let submit: SubmitSourceBufferFn = std::mem::transmute(vt_v.add(VOICE_SUBMIT_SOURCE_BUFFER).read());
            let get_st: GetStateFn = std::mem::transmute(vt_v.add(VOICE_GET_STATE).read());
            let start: unsafe extern "system" fn(*mut c_void, u32, u32) -> HRESULT = std::mem::transmute(vt_v.add(19).read());
            let flush: FlushSourceBuffersFn = std::mem::transmute(vt_v.add(VOICE_FLUSH_SOURCE_BUFFERS).read());
            let destroy: DestroyVoiceFn = std::mem::transmute(vt_v.add(VOICE_DESTROY).read());

            const CTX1: usize = 0x1001;
            const CTX2: usize = 0x2002;
            let b1 = XAudio2Buffer {
                flags: 0,
                audio_bytes: block.len() as u32,
                audio_data: block.as_ptr(),
                play_begin: 0,
                play_length: 0,
                loop_begin: 0,
                loop_length: 0,
                loop_count: 0,
                context: CTX1 as *mut c_void,
            };
            let raw = opus::tests::MONO_HYBRID_20MS[2];
            let b2 = XAudio2Buffer {
                flags: XAUDIO2_END_OF_STREAM,
                audio_bytes: raw.len() as u32,
                audio_data: raw.as_ptr(),
                play_begin: 0,
                play_length: 0,
                loop_begin: 0,
                loop_length: 0,
                loop_count: 0,
                context: CTX2 as *mut c_void,
            };

            let before_decoded = stats().opus_buffers_decoded;
            assert_eq!(submit(voice, &b1, std::ptr::null()), S_OK);
            assert_eq!(submit(voice, &b2, std::ptr::null()), S_OK);
            assert_eq!(stats().opus_buffers_decoded, before_decoded + 2);

            // GetState must return the title's original CTX1, not our internal token!
            let mut st = XAudio2VoiceState { current_buffer_context: std::ptr::null_mut(), buffers_queued: 0, samples_played: 0 };
            get_st(voice, &mut st, 0);
            assert_eq!({ st.buffers_queued }, 2);
            assert_eq!({ st.current_buffer_context } as usize, CTX1);

            // Mute voice so the test runs silently on developer machines, then play to completion
            let set_vol: unsafe extern "system" fn(*mut c_void, f32, u32) -> HRESULT = std::mem::transmute(vt_v.add(12).read());
            let _ = set_vol(voice, 0.0, 0);
            assert_eq!(start(voice, 0, 0), S_OK);

            let t0 = std::time::Instant::now();
            while (ENDED.load(Ordering::SeqCst) != CTX1 + CTX2 || STREAM_END.load(Ordering::SeqCst) == 0)
                && t0.elapsed() < std::time::Duration::from_secs(3)
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(STARTED.load(Ordering::SeqCst), CTX1 + CTX2);
            assert_eq!(ENDED.load(Ordering::SeqCst), CTX1 + CTX2);
            assert!(STREAM_END.load(Ordering::SeqCst) >= 1);

            // All decoded buffers must have been freed on OnBufferEnd!
            let remaining = DECODED_VOICES
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|m| m.get(&(voice as usize)).map(|h| h.state.live_buffers.lock().unwrap().len()))
                .unwrap_or(999);
            assert_eq!(remaining, 0, "decoded PCM buffers leaked after OnBufferEnd");

            // Test FlushSourceBuffers while stopped + DestroyVoice cleanup
            let stop: unsafe extern "system" fn(*mut c_void, u32, u32) -> HRESULT = std::mem::transmute(vt_v.add(20).read());
            let _ = stop(voice, 0, 0);
            assert_eq!(submit(voice, &b2, std::ptr::null()), S_OK);
            assert_eq!(flush(voice), S_OK);

            destroy(voice);
            destroy(master);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vt_eng.add(2).read());
            release(eng);
        }
    }

    #[test]
    fn xma2_source_voice_decodes_and_manages_lifetime() {
        use std::sync::atomic::AtomicUsize;

        static ENDED: AtomicUsize = AtomicUsize::new(0);
        static STARTED: AtomicUsize = AtomicUsize::new(0);
        static STREAM_END: AtomicUsize = AtomicUsize::new(0);

        unsafe extern "system" fn on_pass_start(_: *mut c_void, _: u32) {}
        unsafe extern "system" fn on_nothing(_: *mut c_void) {}
        unsafe extern "system" fn on_stream_end(_: *mut c_void) {
            STREAM_END.fetch_add(1, Ordering::SeqCst);
        }
        unsafe extern "system" fn on_buf_start(_: *mut c_void, ctx: *mut c_void) {
            STARTED.fetch_add(ctx as usize, Ordering::SeqCst);
        }
        unsafe extern "system" fn on_buf_end(_: *mut c_void, ctx: *mut c_void) {
            ENDED.fetch_add(ctx as usize, Ordering::SeqCst);
        }
        unsafe extern "system" fn on_loop(_: *mut c_void, _: *mut c_void) {}
        unsafe extern "system" fn on_err(_: *mut c_void, _: *mut c_void, _: HRESULT) {}

        static TEST_VTBL: VoiceCallbackVtbl = VoiceCallbackVtbl {
            pass_start: on_pass_start,
            pass_end: on_nothing,
            stream_end: on_stream_end,
            buffer_start: on_buf_start,
            buffer_end: on_buf_end,
            loop_end: on_loop,
            voice_error: on_err,
        };
        #[repr(C)]
        struct TestCb {
            vtbl: &'static VoiceCallbackVtbl,
        }
        static TEST_CB: TestCb = TestCb { vtbl: &TEST_VTBL };

        unsafe {
            let mut eng: *mut c_void = std::ptr::null_mut();
            assert_eq!(XAudio2Create(&mut eng, 0, 1), S_OK);
            let vt_eng = vtable(eng);
            let create_master: CreateMasteringVoiceFn = std::mem::transmute(vt_eng.add(ENGINE_CREATE_MASTERING_VOICE).read());
            let mut master: *mut c_void = std::ptr::null_mut();
            if create_master(eng, &mut master, 0, 0, 0, std::ptr::null(), std::ptr::null(), AUDIO_CATEGORY_GAME_EFFECTS) < 0 {
                let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vt_eng.add(2).read());
                release(eng);
                eprintln!("skipping: no audio device");
                return;
            }

            let create_src: CreateSourceVoiceFn = std::mem::transmute(vt_eng.add(ENGINE_CREATE_SOURCE_VOICE).read());
            let mut xma_fmt = xma2(1, 48_000, 6080, 2048, 1);
            xma_fmt.num_streams = 1;
            let mut voice: *mut c_void = std::ptr::null_mut();
            let hr = create_src(
                eng,
                &mut voice,
                &xma_fmt as *const Xma2WaveFormatEx as *const WaveFormatEx,
                0,
                2.0,
                &TEST_CB as *const _ as *mut c_void,
                std::ptr::null(),
                std::ptr::null(),
            );
            assert_eq!(hr, S_OK, "CreateSourceVoice(XMA2) → {:#x}", hr as u32);
            assert!(!voice.is_null());

            let pkt = xma2::tests::pad_packet(xma2::tests::MONO_48K_PREFIX);
            let bad_pkt = vec![0x5Au8; xma2::PACKET_SIZE];

            let vt_v = vtable(voice);
            let submit: SubmitSourceBufferFn = std::mem::transmute(vt_v.add(VOICE_SUBMIT_SOURCE_BUFFER).read());
            let get_st: GetStateFn = std::mem::transmute(vt_v.add(VOICE_GET_STATE).read());
            let start: unsafe extern "system" fn(*mut c_void, u32, u32) -> HRESULT = std::mem::transmute(vt_v.add(19).read());
            let stop: unsafe extern "system" fn(*mut c_void, u32, u32) -> HRESULT = std::mem::transmute(vt_v.add(20).read());
            let flush: FlushSourceBuffersFn = std::mem::transmute(vt_v.add(VOICE_FLUSH_SOURCE_BUFFERS).read());
            let destroy: DestroyVoiceFn = std::mem::transmute(vt_v.add(VOICE_DESTROY).read());

            const CTX_VALID: usize = 0x3003;
            const CTX_BAD: usize = 0x4004;
            let b_valid = XAudio2Buffer {
                flags: XAUDIO2_END_OF_STREAM,
                audio_bytes: pkt.len() as u32,
                audio_data: pkt.as_ptr(),
                play_begin: 0,
                play_length: 2400, // 50 ms slice of the 6080 decoded samples
                loop_begin: 0,
                loop_length: 0,
                loop_count: 0,
                context: CTX_VALID as *mut c_void,
            };
            // Corrupt buffer must gracefully fall back to silence and increment xma_decode_errors.
            let b_bad = XAudio2Buffer {
                flags: XAUDIO2_END_OF_STREAM,
                audio_bytes: bad_pkt.len() as u32,
                audio_data: bad_pkt.as_ptr(),
                play_begin: 0,
                play_length: 480,
                loop_begin: 0,
                loop_length: 0,
                loop_count: 0,
                context: CTX_BAD as *mut c_void,
            };

            let before_decoded = stats().xma_buffers_decoded;
            let before_errors = stats().xma_decode_errors;
            assert_eq!(submit(voice, &b_valid, std::ptr::null()), S_OK);
            assert_eq!(submit(voice, &b_bad, std::ptr::null()), S_OK);
            assert_eq!(stats().xma_buffers_decoded, before_decoded + 1);
            assert_eq!(stats().xma_decode_errors, before_errors + 1);

            let mut st = XAudio2VoiceState { current_buffer_context: std::ptr::null_mut(), buffers_queued: 0, samples_played: 0 };
            get_st(voice, &mut st, 0);
            assert_eq!({ st.buffers_queued }, 2);
            assert_eq!({ st.current_buffer_context } as usize, CTX_VALID);

            let set_vol: unsafe extern "system" fn(*mut c_void, f32, u32) -> HRESULT = std::mem::transmute(vt_v.add(12).read());
            let _ = set_vol(voice, 0.0, 0);
            assert_eq!(start(voice, 0, 0), S_OK);

            let t0 = std::time::Instant::now();
            while (ENDED.load(Ordering::SeqCst) != CTX_VALID + CTX_BAD || STREAM_END.load(Ordering::SeqCst) < 2)
                && t0.elapsed() < std::time::Duration::from_secs(3)
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(STARTED.load(Ordering::SeqCst), CTX_VALID + CTX_BAD);
            assert_eq!(ENDED.load(Ordering::SeqCst), CTX_VALID + CTX_BAD);
            assert!(STREAM_END.load(Ordering::SeqCst) >= 2);

            let remaining = DECODED_VOICES
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|m| m.get(&(voice as usize)).map(|h| h.state.live_buffers.lock().unwrap().len()))
                .unwrap_or(999);
            assert_eq!(remaining, 0, "decoded XMA2 PCM buffers leaked after OnBufferEnd");

            let _ = stop(voice, 0, 0);
            assert_eq!(submit(voice, &b_valid, std::ptr::null()), S_OK);
            assert_eq!(flush(voice), S_OK);

            destroy(voice);
            destroy(master);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vt_eng.add(2).read());
            release(eng);
        }
    }

    #[test]
    fn clamp_regions_enforces_play_end_and_loop_invariants() {
        // Loop extending past PlayBegin + PlayLength is clamped to play_end (300).
        let mut b = buffer(100, 100, 200, 50, 400, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }, { b.loop_count }), (100, 200, 50, 250, 1));

        // Loop starting at or past play_end is disabled.
        let mut b = buffer(100, 100, 200, 350, 50, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 0, 0));

        // Loop ending at or before PlayBegin (including exact loop_end == play_begin) is disabled.
        let mut b = buffer(100, 200, 400, 50, 100, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 0, 0));
        let mut b = buffer(100, 200, 400, 50, 150, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 0, 0));

        // PlayBegin > 0 with PlayLength == 0 gets explicit PlayLength = total_frames - PlayBegin.
        let mut b = buffer(100, 100, 0, 0, 0, 0);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.play_begin }, { b.play_length }), (100, 860));

        // Non-zero LoopBegin with LoopLength == 0 gets explicit LoopLength = play_end - loop_begin.
        let mut b = buffer(100, 0, 0, 100, 0, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.loop_begin }, { b.loop_length }, { b.loop_count }), (100, 860, 1));

        // LoopBegin == 0, LoopLength == 0 with restricted PlayLength gets explicit LoopLength = play_end.
        let mut b = buffer(100, 100, 200, 0, 0, 1);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 300, 1));

        // LoopCount == 0 clears stale LoopBegin/LoopLength.
        let mut b = buffer(100, 0, 0, 100, 200, 0);
        clamp_regions(&mut b, 960);
        assert_eq!(({ b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 0, 0));

        // total_frames == u32::MAX with play_begin + play_length overflowing u32 does not panic or wrap.
        let mut b = buffer(100, 1, u32::MAX, 0, u32::MAX, 1);
        clamp_regions(&mut b, u32::MAX);
        assert_eq!(({ b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }, { b.loop_count }), (1, u32::MAX - 1, 0, u32::MAX, 1));
    }

    #[test]
    fn silent_buffer_clamps_regions_and_zero_max_bytes() {
        let i = XmaInfo { channels: 2, rate: 48_000, ..Default::default() };
        let z = 0x5000 as *const u8;

        // Loop longer than play_length keeps the buffer big enough, while clamping loop_length to play_end.
        let b = silent_buffer(&i, &buffer(0x1000, 0, 100, 0, 9000, 255), z, 1 << 20);
        assert_eq!({ b.audio_bytes } / 4, 9000);
        assert_eq!(({ b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 100, 0, 100, 255));

        // Non-zero LoopBegin with LoopLength == 0 allocates at least LoopBegin + 1 frames and sets explicit LoopLength.
        let b = silent_buffer(&i, &buffer(0x100, 0, 0, 5000, 0, 1), z, 1 << 20);
        assert!({ b.audio_bytes } / 4 >= 5001);
        assert_eq!({ b.loop_begin }, 5000);
        assert!({ b.loop_length } >= 1);
        assert_eq!({ b.loop_count }, 1);

        // Zero max_bytes clears loop_count as well as all regions.
        let b = silent_buffer(&i, &buffer(0x1000, 0, 100, 0, 100, 255), z, 0);
        assert_eq!(({ b.audio_bytes }, { b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }, { b.loop_count }), (0, 0, 0, 0, 0, 0));

        // Truncated looping buffer with non-zero max_bytes loops the full silence buffer cleanly.
        let b = silent_buffer(&i, &buffer(0x1000, 1000, 1_000_000, 1000, 500_000, 255), z, 4000);
        assert_eq!(({ b.audio_bytes }, { b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }, { b.loop_count }), (4000, 0, 0, 0, 0, 255));
    }

    #[test]
    fn clamped_and_silenced_buffers_accepted_by_windows_xaudio2() {
        struct AudioGuard {
            eng: *mut c_void,
            master: *mut c_void,
            voice: *mut c_void,
        }
        impl Drop for AudioGuard {
            fn drop(&mut self) {
                unsafe {
                    if !self.voice.is_null() {
                        let destroy: DestroyVoiceFn = std::mem::transmute(vtable(self.voice).add(VOICE_DESTROY).read());
                        destroy(self.voice);
                    }
                    if !self.master.is_null() {
                        let destroy: DestroyVoiceFn = std::mem::transmute(vtable(self.master).add(VOICE_DESTROY).read());
                        destroy(self.master);
                    }
                    if !self.eng.is_null() {
                        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vtable(self.eng).add(2).read());
                        release(self.eng);
                    }
                }
            }
        }

        // Buffers declared before the guard so the guard drops first (destroying voices before backing memory frees).
        let pcm = vec![0u8; 960 * 4];
        let sil = vec![0u8; 64 * 1024];

        unsafe {
            let mut guard = AudioGuard { eng: std::ptr::null_mut(), master: std::ptr::null_mut(), voice: std::ptr::null_mut() };
            assert_eq!(XAudio2Create(&mut guard.eng, 0, 1), S_OK);
            let vt_eng = vtable(guard.eng);
            let create_master: CreateMasteringVoiceFn = std::mem::transmute(vt_eng.add(ENGINE_CREATE_MASTERING_VOICE).read());
            if create_master(guard.eng, &mut guard.master, 0, 0, 0, std::ptr::null(), std::ptr::null(), AUDIO_CATEGORY_GAME_EFFECTS) < 0 {
                eprintln!("skipping: no audio device");
                return;
            }
            let create_src: CreateSourceVoiceFn = std::mem::transmute(vt_eng.add(ENGINE_CREATE_SOURCE_VOICE).read());
            let wfx = pcm16_for_rate(2, 48_000);
            assert_eq!(create_src(guard.eng, &mut guard.voice, &wfx, 0, 2.0, std::ptr::null_mut(), std::ptr::null(), std::ptr::null()), S_OK);
            let vt_v = vtable(guard.voice);
            let submit: SubmitSourceBufferFn = std::mem::transmute(vt_v.add(VOICE_SUBMIT_SOURCE_BUFFER).read());
            let flush: FlushSourceBuffersFn = std::mem::transmute(vt_v.add(VOICE_FLUSH_SOURCE_BUFFERS).read());

            let cases = [
                (100, 200, 50, 400, 1),  // loop_end > play_end
                (100, 200, 350, 50, 1),  // loop_begin >= play_end
                (200, 400, 50, 100, 1),  // loop_end <= play_begin
                (200, 400, 50, 150, 1),  // loop_end == play_begin
                (0, 0, 100, 0, 1),       // loop_begin > 0 with loop_length == 0
                (100, 200, 0, 0, 1),     // loop_length == 0 with restricted play_length
                (100, 0, 0, 0, 1),       // play_begin > 0, play_length == 0, whole-buffer loop
                (0, 0, 100, 200, 0),     // loop_count == 0 with stale loop fields
            ];
            for (pb, pl, lb, ll, lc) in cases {
                let mut b = XAudio2Buffer {
                    flags: 0,
                    audio_bytes: pcm.len() as u32,
                    audio_data: pcm.as_ptr(),
                    play_begin: pb,
                    play_length: pl,
                    loop_begin: lb,
                    loop_length: ll,
                    loop_count: lc,
                    context: std::ptr::null_mut(),
                };
                clamp_regions(&mut b, 960);
                let hr = submit(guard.voice, &b, std::ptr::null());
                assert_eq!(hr, S_OK, "SubmitSourceBuffer failed ({hr:#x}) for case ({pb},{pl},{lb},{ll},{lc}) -> ({},{},{},{},{})", { b.play_begin }, { b.play_length }, { b.loop_begin }, { b.loop_length }, { b.loop_count });
                assert_eq!(flush(guard.voice), S_OK);
            }

            let info = XmaInfo { channels: 2, rate: 48_000, ..Default::default() };
            let sb1 = silent_buffer(&info, &buffer(0x1000, 0, 100, 0, 9000, 255), sil.as_ptr(), sil.len());
            assert_eq!(submit(guard.voice, &sb1, std::ptr::null()), S_OK);
            assert_eq!(flush(guard.voice), S_OK);
            let sb2 = silent_buffer(&info, &buffer(0x100, 0, 0, 5000, 0, 1), sil.as_ptr(), sil.len());
            assert_eq!(submit(guard.voice, &sb2, std::ptr::null()), S_OK);
            assert_eq!(flush(guard.voice), S_OK);
        }
    }
}
