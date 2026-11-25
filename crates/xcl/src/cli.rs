//! Shared option parser for CLI subcommands (`xcl rpf`, `xcl shaders`).
//!
//! Rejects unknown options, stray positional arguments, missing option values,
//! and malformed numbers/hashes instead of silently falling back to defaults.

pub(crate) struct Parser<'a> {
    args: &'a [String],
    index: usize,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(args: &'a [String]) -> Self {
        Self { args, index: 0 }
    }

    pub(crate) fn next_arg(&mut self) -> Option<&'a str> {
        let arg = self.args.get(self.index)?;
        self.index += 1;
        Some(arg.as_str())
    }

    /// Consume the required value following option `name`.
    /// Rejects end-of-arguments or another `--...` flag in its place.
    pub(crate) fn value(&mut self, name: &str) -> Result<&'a str, String> {
        match self.args.get(self.index) {
            Some(v) if !v.starts_with("--") => {
                self.index += 1;
                Ok(v.as_str())
            }
            _ => Err(format!("{name} needs a value")),
        }
    }

    pub(crate) fn parse_usize(&mut self, name: &str) -> Result<usize, String> {
        let v = self.value(name)?;
        v.parse::<usize>().map_err(|_| format!("invalid value for {name}: {v}"))
    }

    pub(crate) fn parse_u64(&mut self, name: &str) -> Result<u64, String> {
        let v = self.value(name)?;
        v.parse::<u64>().map_err(|_| format!("invalid value for {name}: {v}"))
    }

    pub(crate) fn parse_bool_01(&mut self, name: &str) -> Result<bool, String> {
        let v = self.value(name)?;
        match v {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(format!("invalid value for {name}: {v} (expected 0 or 1)")),
        }
    }

    pub(crate) fn parse_hex_u64(&mut self, name: &str) -> Result<u64, String> {
        let v = self.value(name)?;
        parse_hex_u64_str(v).map_err(|e| format!("{name}: {e}"))
    }

    pub(crate) fn unexpected(arg: &str) -> String {
        if arg.starts_with('-') {
            format!("unknown option: {arg}; see `xcl --help`")
        } else {
            format!("unexpected argument: {arg}; see `xcl --help`")
        }
    }
}

pub(crate) fn parse_hex_u64_str(s: &str) -> Result<u64, String> {
    let hex = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if hex.is_empty() || hex.len() > 16 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("{s:?} is not a 1–16 digit hexadecimal hash"));
    }
    u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn value_and_numeric_helpers_reject_missing_and_malformed_inputs() {
        let a = vecs(&["--top"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--top"));
        assert_eq!(p.parse_usize("--top").unwrap_err(), "--top needs a value");

        let a = vecs(&["--top", "--full"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--top"));
        assert_eq!(p.parse_usize("--top").unwrap_err(), "--top needs a value");

        let a = vecs(&["--top", "abc"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--top"));
        assert_eq!(p.parse_usize("--top").unwrap_err(), "invalid value for --top: abc");

        let a = vecs(&["--top", "-5"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--top"));
        assert_eq!(p.parse_usize("--top").unwrap_err(), "invalid value for --top: -5");

        let a = vecs(&["--split", "2"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--split"));
        assert!(p.parse_bool_01("--split").unwrap_err().contains("invalid value for --split: 2"));

        let a = vecs(&["--hash", "0xZZ"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--hash"));
        assert!(p.parse_hex_u64("--hash").unwrap_err().contains("--hash:"));

        let a = vecs(&["--hash", "0xDeadBeef"]);
        let mut p = Parser::new(&a);
        assert_eq!(p.next_arg(), Some("--hash"));
        assert_eq!(p.parse_hex_u64("--hash").unwrap(), 0xDEAD_BEEF);

        assert_eq!(Parser::unexpected("--fnid"), "unknown option: --fnid; see `xcl --help`");
        assert_eq!(Parser::unexpected("extra"), "unexpected argument: extra; see `xcl --help`");
    }
}
