//! BCPack: the DirectStorage stage between zlib and swizzle that unpacks console BCn blocks.
//! A stream has no header; 12 or 16 rANS lanes share one byte stream and rebuild into blocks.
//! Decodes BC1, BC3, BC4, BC5, BC6H and BC7 from the format's constant tables (`tables.rs`);
//! BC2 has no lane layout.
//! Port of VIctory's `victory-formats/src/bcpack.rs`; tables generated from its
//! `bcpack-tables.bin` by `scripts/gen_bcpack_victory_tables.py`.

mod tables;
mod victory;

/// Why a stream does not unpack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// no decoder for this mode
    Unsupported,
    /// the output is not a whole number of blocks
    OutputSize,
    /// the stream is truncated, malformed, or ends in a bad lane state
    Decode(&'static str),
}

impl Error {
    /// The kind of failure, without its numbers.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Unsupported => "unsupported",
            Error::OutputSize => "output size",
            Error::Decode(_) => "decode",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::Unsupported => write!(f, "no decoder for this mode"),
            Error::OutputSize => write!(f, "the output size is not a whole number of blocks"),
            Error::Decode(why) => write!(f, "{why}"),
        }
    }
}

/// The bytes of one block of `mode`.
pub fn block_size(mode: u8) -> Option<usize> {
    Some(match mode {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return None,
    })
}

/// Whether [`decode`] unpacks `mode`.
pub fn can_decode(mode: u8) -> bool {
    block_size(mode).is_some()
}

/// Unpack a stream of `mode` to its `out_len` bytes of BCn blocks.
pub fn decode(mode: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    let block = block_size(mode).ok_or(Error::Unsupported)?;
    if out_len == 0 || out_len % block != 0 {
        return Err(Error::OutputSize);
    }
    let mut out = vec![0u8; out_len];
    victory::decode(mode, src, &mut out, victory::Tables::builtin())
        .map_err(|e| Error::Decode(e.0))?;
    Ok(out)
}

#[cfg(test)]
mod tests;
