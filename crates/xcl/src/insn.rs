//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX, which are AMD's). The
//! exception handler (`eh`) emulates the ones decoded here and steps over them.

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a immediate form (`src` None) or register-control form (low source bits 5:0,
    /// 13:8 are the length and index). XMM register numbers are 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a INSERTQ; register controls are in source bits 69:64 and 77:72.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

// SSE4a implementation from upstream 0cd0a53b584a8be0fd779d16e5826d97a0e66344,
// with Zen 5 hardware-verified prefix behavior from af20b4b38e5a8003234f28091fb45e03aca68167.
// Kept separate from the local bounded CLZERO loop emulation.
fn field_len(len: u8) -> u32 {
    match len & 63 { 0 => 64, n => n as u32 }
}

/// Defined EXTRQ fields follow AMD's manual. A field crossing bit 63 is undefined;
/// upstream consistently uses the result of the shift and mask for those inputs.
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 { field } else { field & ((1u64 << len) - 1) }
}

pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// Only SSE4a uses this upstream prefix scan. It does not broaden the existing decoder's
/// handling of address-size prefixes on CLZERO or replace the local loop safety checks.
fn decode_sse4a(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    let mut i = 0;
    let (mut p66, mut rep, mut rex) = (false, 0u8, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => rep = 0xF3,
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67 => {}
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F { return None; }
    let opcode = *code.get(i + 1)?;
    if matches!(opcode, 0x78 | 0x79) && i + (if opcode == 0x78 { 5 } else { 3 }) > 15 { return None; }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

/// The console-only instruction at the start of `code` and its length.
pub fn decode(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    if let Some(sse4a) = decode_sse4a(code) { return Some(sse4a); }
    let mut i = 0;
    let (mut p66, mut pf3, mut rex) = (false, false, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF3 => pf3 = true,
            0xF2 | 0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 => {}
            0x40..=0x4F => {
                rex = b;
                i += 1;
                break; // REX is the last prefix
            }
            _ => break,
        }
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => Some((ConsoleOnly::Clzero, i + 3)),
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}

/// Recognize only `clzero; add rax,64; cmp rax,r64; jb clzero`.
/// The register index uses the CONTEXT order documented on `ConsoleOnly::Rdpid`.
/// Returning to the native CMP after a batch preserves its flags and branch behavior.
pub fn clzero_loop_bound(code: &[u8]) -> Option<usize> {
    let b = code.get(..12)?;
    if b[..7] != [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40]
        || !matches!(b[7], 0x48 | 0x4C) || b[8] != 0x39
        || b[9] & 0xC7 != 0xC0 || b[10..] != [0x72, 0xF4]
    {
        return None;
    }
    Some(((b[9] >> 3) & 7) as usize | ((((b[7] >> 2) & 1) as usize) << 3))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClzeroBatch {
    /// CLZERO rounds its address down to a 64-byte line.
    pub address: usize,
    pub bytes: usize,
    /// RAX at the following native CMP, after all emulated ADDs.
    pub next_rax: usize,
}

/// Plan a bounded number of iterations of the recognized do-while loop. No memory is
/// accessed here; the exception handler must validate the whole writable span first.
pub fn clzero_batch(start: usize, bound: usize) -> Option<ClzeroBatch> {
    const MAX_LINES: usize = (1 << 20) / 64;
    let distance = bound.saturating_sub(start);
    let lines = (distance / 64 + usize::from(distance % 64 != 0)).max(1).min(MAX_LINES);
    let bytes = lines.checked_mul(64)?;
    let next_rax = start.checked_add(bytes)?;
    let address = start & !63;
    address.checked_add(bytes)?;
    Some(ClzeroBatch { address, bytes, next_rax })
}

/// The observed `clwb [rax]; add rax,64; cmp rax,rcx; jbe clwb` loop.
/// Unlike upstream b25582a's CLWB NOP patch, this leaves the title's code intact.
/// Both encodings below compare RAX with RCX; prefixes, operands and backedge
/// must otherwise match exactly before any native arithmetic is skipped.
pub fn clwb_loop(code: &[u8]) -> bool {
    let Some(b) = code.get(..13) else { return false };
    b[..8] == [0x66, 0x0F, 0xAE, 0x30, 0x48, 0x83, 0xC0, 0x40]
        && (b[8..11] == [0x48, 0x39, 0xC8] || b[8..11] == [0x48, 0x3B, 0xC1])
        && b[11..13] == [0x76, 0xF3]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClwbBatch {
    pub iterations: usize,
    /// RAX at the following native CMP after the emulated ADDs, without wrapping.
    pub next_rax: usize,
}

/// CLWB is already a no-op in our coherent host-memory emulation. Skip only
/// bounded repetitions of that same operation and its ADD, then run CMP/JBE
/// natively. JBE includes equality: reaching the bound requires another iteration.
/// Declining a wrapping advance leaves the original ADD/branch to handle it.
pub fn clwb_batch(start: usize, bound: usize) -> Option<ClwbBatch> {
    const MAX_LINES: usize = 1 << 20; // at most 64 MiB of address advancement
    let iterations = (bound.saturating_sub(start) / 64 + 1).min(MAX_LINES);
    let next_rax = start.checked_add(iterations.checked_mul(64)?)?;
    Some(ClwbBatch { iterations, next_rax })
}

/// `code` as hex, for a log line.
pub fn hex(code: &[u8]) -> String {
    code.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod clwb_batch_tests {
    use super::*;

    const LOOP: [u8; 13] = [0x66, 0x0F, 0xAE, 0x30, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x3B, 0xC1, 0x76, 0xF3];

    #[test]
    fn accepts_only_the_observed_loop_and_equivalent_compare_encoding() {
        assert!(clwb_loop(&LOOP));
        let mut other = LOOP;
        other[9..11].copy_from_slice(&[0x39, 0xC8]);
        assert!(clwb_loop(&other));
        for n in 0..LOOP.len() { assert!(!clwb_loop(&LOOP[..n])); }
        for i in 0..LOOP.len() {
            let mut changed = LOOP;
            changed[i] ^= 1;
            assert!(!clwb_loop(&changed), "changed byte {i}");
        }
        // A different effective address, increment, bound, strict branch or prefix
        // must keep using the normal single-instruction path.
        for (i, value) in [(3, 0x31), (7, 0x20), (10, 0xC2), (11, 0x72), (0, 0x67)] {
            let mut changed = LOOP;
            changed[i] = value;
            assert!(!clwb_loop(&changed));
        }
    }

    #[test]
    fn inclusive_bound_and_one_iteration_match_a_scalar_noop_loop() {
        for start in [0usize, 1, 0x1000, 0x1051] {
            for distance in 0..257usize {
                let bound = start + distance;
                let mut scalar = start;
                let mut iterations = 0;
                loop {
                    scalar += 64;
                    iterations += 1;
                    if scalar > bound { break; }
                }
                assert_eq!(clwb_batch(start, bound), Some(ClwbBatch { iterations, next_rax: scalar }));
            }
        }
        assert_eq!(clwb_batch(0x1000, 0x1040), Some(ClwbBatch { iterations: 2, next_rax: 0x1080 }));
        assert_eq!(clwb_batch(0x1000, 0x1000), Some(ClwbBatch { iterations: 1, next_rax: 0x1040 }));
        assert_eq!(clwb_batch(0x1000, 0), Some(ClwbBatch { iterations: 1, next_rax: 0x1040 }));
    }

    #[test]
    fn bounds_each_chunk_and_declines_wrapping_adds() {
        let batch = clwb_batch(0x1000, usize::MAX).unwrap();
        assert_eq!(batch.iterations, 1 << 20);
        assert_eq!(batch.next_rax - 0x1000, 64 << 20);
        assert!(batch.next_rax < usize::MAX, "the native JBE continues at a chunk boundary");
        assert_eq!(clwb_batch(usize::MAX - 63, usize::MAX), None);
        assert_eq!(clwb_batch(usize::MAX, 0), None);
        assert_eq!(clwb_batch(usize::MAX - 64, 0), Some(ClwbBatch { iterations: 1, next_rax: usize::MAX }));
    }
}

#[cfg(test)]
mod sse4a_tests {
    use super::*;

    #[test]
    fn decodes_observed_insertq_and_all_rex_register_forms() {
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 16, 16]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 6)));
        for dst in 0..16u8 {
            for src in 0..16u8 {
                let rex = 0x40 | ((dst >> 3) << 2) | (src >> 3);
                let m = 0xC0 | ((dst & 7) << 3) | (src & 7);
                assert_eq!(decode(&[0xF2, rex, 0x0F, 0x78, m, 0x50, 0x50]), Some((ConsoleOnly::Insertq { dst, src, reg: false, len: 0x50, idx: 0x50 }, 7)));
                assert_eq!(decode(&[0xF2, rex, 0x0F, 0x79, m]), Some((ConsoleOnly::Insertq { dst, src, reg: true, len: 0, idx: 0 }, 5)));
                assert_eq!(decode(&[0x66, rex, 0x0F, 0x79, m]), Some((ConsoleOnly::Extrq { dst, src: Some(src), len: 0, idx: 0 }, 5)));
            }
        }
    }

    #[test]
    fn decodes_extrq_immediate_fixed_extension_and_rex_b() {
        for dst in 0..16u8 {
            assert_eq!(decode(&[0x66, 0x40 | (dst >> 3), 0x0F, 0x78, 0xC0 | (dst & 7), 8, 16]), Some((ConsoleOnly::Extrq { dst, src: None, len: 8, idx: 16 }, 7)));
        }
        // Upstream follows the hardware's /0 check on ModRM; REX.R is irrelevant here.
        assert_eq!(decode(&[0x66, 0x44, 0x0F, 0x78, 0xC0, 8, 16]), Some((ConsoleOnly::Extrq { dst: 0, src: None, len: 8, idx: 16 }, 7)));
    }

    #[test]
    fn follows_upstream_hardware_verified_prefix_order() {
        let expected = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [
            [0x66, 0xF2, 0x0F, 0x78, 0xC1, 16, 16], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 16, 16],
            [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 16, 16], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 16, 16],
            [0x41, 0xF2, 0x0F, 0x78, 0xC1, 16, 16],
        ] { assert_eq!(decode(&code), expected, "{code:02x?}"); }
        for code in [
            [0xF2, 0xF3, 0x0F, 0x78, 0xC1, 16, 16], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 16, 16],
            [0x66, 0xF3, 0x0F, 0x78, 0xC0, 16, 16],
        ] { assert_eq!(decode(&code), None, "#UD prefix order: {code:02x?}"); }
    }

    #[test]
    fn rejects_truncation_memory_operands_and_non_sse4a_opcodes() {
        let code = [0xF2, 0x0F, 0x78, 0xC1, 16, 16];
        for n in 0..code.len() { assert_eq!(decode(&code[..n]), None); }
        for bad in [
            &[0xF2, 0x0F, 0x78, 0x01, 16, 16][..], &[0x66, 0x0F, 0x78, 0xC8, 16, 16][..],
            &[0x0F, 0x78, 0xC1, 16, 16][..], &[0xF3, 0x0F, 0x78, 0xC1, 16, 16][..],
            &[0xF2, 0x0F, 0x79, 0x01][..], &[0x66, 0x0F, 0x79, 0x01][..],
        ] { assert_eq!(decode(bad), None, "{bad:02x?}"); }
        let mut too_long = vec![0x66; 10];
        too_long.extend_from_slice(&code);
        assert_eq!(decode(&too_long), None, "an x86 instruction may not exceed 15 bytes");
    }

    #[test]
    fn defined_fields_match_an_independent_bitwise_reference() {
        let (dst, src) = (0xA5A5_89AB_CDEF_0123u64, 0xF0E1_D2C3_B4A5_9687u64);
        for idx in 0..64u8 {
            for width in 1..=64u8 - idx {
                let mut expected = dst;
                for bit in 0..width {
                    let at = u32::from(idx + bit);
                    expected = (expected & !(1u64 << at)) | (((src >> bit) & 1) << at);
                }
                let len = if width == 64 { 0 } else { width };
                assert_eq!(insertq(dst, src, len, idx), expected);
                let extracted = (0..width).fold(0u64, |v, bit| v | (((src >> (idx + bit)) & 1) << bit));
                assert_eq!(extrq(src, len, idx), extracted);
            }
        }
    }

    #[test]
    fn masks_controls_and_matches_upstream_undefined_field_policy() {
        assert_eq!(insertq(u64::MAX, 0x1234, 0x50, 0x50), 0xFFFF_FFFF_1234_FFFF);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0x50, 0x50), 0x9ABC);
        assert_eq!(insertq(0, u64::MAX, 0, 0), u64::MAX);
        assert_eq!(extrq(u64::MAX, 0, 0), u64::MAX);
        assert_eq!(insertq(0, u64::MAX, 0, 1), u64::MAX - 1);
        assert_eq!(extrq(u64::MAX, 63, 2), u64::MAX >> 2);
    }
}

#[cfg(test)]
mod clzero_batch_tests {
    use super::*;

    const LOOP_R10: [u8; 12] = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4];

    #[test]
    fn recognizes_exact_loop_and_all_bound_registers() {
        assert_eq!(clzero_loop_bound(&LOOP_R10), Some(10));
        for reg in 0..16usize {
            let mut code = LOOP_R10;
            code[7] = if reg < 8 { 0x48 } else { 0x4C };
            code[9] = 0xC0 | ((reg as u8 & 7) << 3);
            assert_eq!(clzero_loop_bound(&code), Some(reg));
        }
    }

    #[test]
    fn rejects_changed_instructions_backedges_and_truncation() {
        for n in 0..LOOP_R10.len() {
            assert_eq!(clzero_loop_bound(&LOOP_R10[..n]), None);
        }
        for (at, byte) in [(0, 0x90), (6, 0x20), (7, 0x44), (8, 0x3B), (9, 0xD1), (10, 0x76), (11, 0xF5)] {
            let mut code = LOOP_R10;
            code[at] = byte;
            assert_eq!(clzero_loop_bound(&code), None, "changed byte {at}");
        }
    }

    #[test]
    fn plans_unaligned_zero_lines_and_final_comparison_register() {
        let p = clzero_batch(0x1051, 0x1102).unwrap();
        assert_eq!(p, ClzeroBatch { address: 0x1040, bytes: 0xC0, next_rax: 0x1111 });
        assert!(p.next_rax >= 0x1102);
        assert!(p.next_rax - 64 < 0x1102);
    }

    #[test]
    fn executes_one_iteration_even_when_bound_is_not_ahead() {
        for bound in [0x1000, 0x1051] {
            assert_eq!(clzero_batch(0x1051, bound), Some(ClzeroBatch { address: 0x1040, bytes: 64, next_rax: 0x1091 }));
        }
    }

    #[test]
    fn caps_work_and_rejects_addition_overflow() {
        let p = clzero_batch(0x1000, usize::MAX).unwrap();
        assert_eq!(p.bytes, 1 << 20);
        assert_eq!(p.next_rax, 0x101000);
        assert!(p.next_rax < usize::MAX); // native JB will request another bounded batch
        assert_eq!(clzero_batch(usize::MAX - 63, usize::MAX), None);
        assert_eq!(clzero_batch(usize::MAX, 0), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // Exact instruction at the retail title's illegal-instruction crash: clwb [r9].
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x31]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // not console-only: clflushopt (/7), xsaveopt (no 66), a plain mov, truncated bytes
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }
}
