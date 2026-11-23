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
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
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

/// `code` as hex, for a log line.
pub fn hex(code: &[u8]) -> String {
    code.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
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
