//! What a console compute program can change outside itself, read from its code alone.
//!
//! A dispatch XCL does not run (its pipeline is still being built) leaves undone what the
//! program would have written. For memory that is dangerous: a culling pass writes the
//! counts and the arguments an `ExecuteIndirect` reads next, and without them the GPU is
//! given unbounded work. A program that writes nothing but images leaves a stale picture
//! and nothing else, so its dispatches can go on without it while the driver compiles it
//! (`hle::d3d12x::pso_async`).

use super::{rdna2, recomp, xbox};

/// The writes a program can make.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    /// memory other programs, the CPU or indirect arguments may read: buffer / flat / global /
    /// scratch stores and atomics, and the global data share (append / consume counters, GWS)
    pub memory: bool,
    /// image stores and image atomics
    pub images: bool,
    /// instructions in the program
    pub instructions: usize,
}

/// The writes of the RDNA2 program `code` (one program, as `xbox::Fssm::programs` cuts it).
/// None when the program cannot be decoded through a terminator, falls through past decoded
/// code, contains an unresolved indirect transfer, or branches outside the decoded instructions.
pub fn of(code: &[u8]) -> Option<Effects> {
    let mut insts = Vec::new();
    let mut off = 0;
    // the whole program: code can follow an early terminator (a branch to it, or past it); what
    // does not decode (or S_CODE_END padding) after one is taken for trailing data, unless a
    // decoded branch still targets past it or the tail falls through into it
    let mut ended = false;
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else {
            if ended {
                break;
            }
            return None;
        };
        if ended && d.inst.name == "S_CODE_END" {
            break;
        }
        off += d.len;
        ended |= d.inst.flags & rdna2::F_TERMINATOR != 0;
        insts.push(d);
    }
    if !ended {
        return None;
    }
    rdna2::resolve_far_jumps(&mut insts);
    let last = insts.last()?;
    if last.inst.flags & rdna2::F_TERMINATOR == 0 && last.inst.name != "S_BRANCH" && last.far_target.is_none() {
        return None;
    }
    let pcs: std::collections::HashSet<u64> = insts.iter().map(|d| d.pc).collect();
    let mut e = Effects::default();
    for d in &insts {
        if d.inst.flags & (rdna2::F_BRANCH | rdna2::F_INDIRECT) != 0 && rdna2::branch_target(d).is_none_or(|t| !pcs.contains(&t)) {
            return None;
        }
        e.instructions += 1;
        let n = d.inst.name;
        let writes = n.contains("_STORE") || n.contains("_ATOMIC");
        match d.enc.name {
            "ENC_MIMG" => e.images |= writes,
            "ENC_MUBUF" | "ENC_MTBUF" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" | "ENC_SMEM" => e.memory |= writes,
            // LDS is the group's own and gone after it; the global data share is not
            "ENC_DS" => {
                let gds = rdna2::field_value(d.enc, &d.words, "GDS").is_some_and(|v| v != 0);
                e.memory |= gds || n == "DS_APPEND" || n == "DS_CONSUME" || n == "DS_ORDERED_COUNT" || n.starts_with("DS_GWS_");
            }
            _ => {}
        }
    }
    Some(e)
}

/// The compute program of a console shader container (the hardware CS variant of an FSSM, or
/// an FSPX driver container), as the translators take it.
fn compute_code(blob: &[u8]) -> Option<Vec<u8>> {
    if let Some(f) = recomp::pso::fspx(blob) {
        return Some(f.code.to_vec());
    }
    let sh = xbox::parse(blob)?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6))?;
    if f.error.is_some() {
        return None;
    }
    let p = f.programs.first()?;
    Some(f.data.get(p.offset..p.offset.checked_add(p.len)?)?.to_vec())
}

/// Console code a compute program needs to be `big`: 16 KiB, about 4 000 instructions. The
/// programs a retail title's world start waits on longest are 73 – 118 KiB each (22 – 172 s
/// apiece in an RDNA2 driver); what it runs every frame from its first one is far smaller.
pub const BIG_CODE: usize = 16 << 10;

/// Whether the compute shader `blob` is a big program whose only writes are to images: one a
/// title can do without for the frames its pipeline takes to build. False for anything that
/// cannot be read.
pub fn big_and_image_only(blob: &[u8]) -> bool {
    let Some(code) = compute_code(blob) else { return false };
    code.len() >= BIG_CODE && of(&code).is_some_and(|e| !e.memory)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(insts: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
        insts.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap_or_else(|e| panic!("{n}: {e}"))).collect()
    }

    const END: (&str, &str, &[(&str, u64)]) = ("S_ENDPGM", "ENC_SOPP", &[]);
    const MOV: (&str, &str, &[(&str, u64)]) = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 129)]);
    const IMAGE_STORE: (&str, &str, &[(&str, u64)]) = ("IMAGE_STORE", "ENC_MIMG", &[("DMASK", 15), ("VADDR", 0), ("VDATA", 4), ("SRSRC", 2)]);
    const IMAGE_LOAD: (&str, &str, &[(&str, u64)]) = ("IMAGE_LOAD", "ENC_MIMG", &[("DMASK", 15), ("VADDR", 0), ("VDATA", 4), ("SRSRC", 2)]);

    #[test]
    fn images_and_memory_are_told_apart() {
        // loads of either kind write nothing
        let load: (&str, &str, &[(&str, u64)]) = ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        assert_eq!(of(&asm(&[MOV, IMAGE_LOAD, load, END])), Some(Effects { memory: false, images: false, instructions: 4 }));
        assert_eq!(of(&asm(&[MOV, IMAGE_STORE, END])), Some(Effects { memory: false, images: true, instructions: 3 }));
        let image_atomic: (&str, &str, &[(&str, u64)]) = ("IMAGE_ATOMIC_ADD", "ENC_MIMG", &[("DMASK", 1), ("VADDR", 0), ("VDATA", 4), ("SRSRC", 2)]);
        assert_eq!(of(&asm(&[image_atomic, END])).map(|e| (e.memory, e.images)), Some((false, true)));
        // every way into memory another pass or an indirect call reads
        let writers: [(&str, &str, &[(&str, u64)]); 7] = [
            ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]),
            ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]),
            ("TBUFFER_STORE_FORMAT_XYZW", "ENC_MTBUF", &[("VDATA", 1), ("SRSRC", 2)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 0), ("DATA", 2)]),
            ("FLAT_ATOMIC_ADD", "ENC_FLAT", &[("ADDR", 0), ("DATA", 2)]),
            ("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 0), ("DATA", 2)]),
            ("DS_APPEND", "ENC_DS", &[("VDST", 1)]),
        ];
        for w in writers {
            assert_eq!(of(&asm(&[IMAGE_STORE, w, END])).map(|e| (e.memory, e.images)), Some((true, true)), "{}", w.0);
        }
        // the group's own LDS is not memory anyone else reads
        let lds: (&str, &str, &[(&str, u64)]) = ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1)]);
        assert_eq!(of(&asm(&[lds, IMAGE_STORE, END])).map(|e| (e.memory, e.images)), Some((false, true)));
    }

    #[test]
    fn code_after_an_early_end_counts() {
        // a branch can jump past an early s_endpgm: a store after it is the program's (were it
        // constant data instead, taking it for a store only makes binds wait, as before)
        let store: (&str, &str, &[(&str, u64)]) = ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        assert_eq!(of(&asm(&[IMAGE_STORE, END, store, END])), Some(Effects { memory: true, images: true, instructions: 4 }));
        // what does not decode after the end is trailing data
        let mut code = asm(&[IMAGE_STORE, END]);
        code.extend_from_slice(&[0xff; 8]);
        if rdna2::decode(&[0xff; 8], 0).is_err() {
            assert_eq!(of(&code), Some(Effects { memory: false, images: true, instructions: 2 }));
        }
    }

    #[test]
    fn the_global_data_share_is_memory() {
        // the same LDS add, once with GDS set: other groups and the CPU can see that one
        let lds: (&str, &str, &[(&str, u64)]) = ("DS_ADD_U32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1)]);
        let gds: (&str, &str, &[(&str, u64)]) = ("DS_ADD_U32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1), ("GDS", 1)]);
        assert_eq!(of(&asm(&[lds, IMAGE_STORE, END])).map(|e| e.memory), Some(false));
        assert_eq!(of(&asm(&[gds, IMAGE_STORE, END])).map(|e| e.memory), Some(true));
    }

    #[test]
    fn code_that_does_not_decode_says_nothing() {
        let mut code = asm(&[IMAGE_STORE]);
        code.extend_from_slice(&[0xff; 8]);
        code.extend(asm(&[END]));
        // (only where the decoder really refuses the dwords: otherwise the test states nothing)
        if rdna2::decode(&[0xff; 8], 0).is_err() {
            assert_eq!(of(&code), None);
        }
    }

    #[test]
    fn branch_past_early_end_gap_and_fallthrough_say_nothing() {
        assert!(rdna2::decode(&[0xff; 8], 0).is_err());
        let store: (&str, &str, &[(&str, u64)]) = ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("SRSRC", 2)]);
        // 0x00: s_cbranch_scc1 +5 (-> 0x18), 0x04: image_store (8B), 0x0c: s_endpgm,
        // 0x10: undecodable gap (8B), 0x18: buffer_store_dword, 0x20: s_endpgm
        let mut jumped = asm(&[("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 5)]), IMAGE_STORE, END]);
        jumped.extend_from_slice(&[0xff; 8]);
        jumped.extend(asm(&[store, END]));
        assert_eq!(of(&jumped), None);

        // 0x00: s_cbranch_scc1 +1 (-> 0x08), 0x04: s_endpgm, 0x08: image_store falls into gap
        let mut fallthrough = asm(&[("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]), END, IMAGE_STORE]);
        fallthrough.extend_from_slice(&[0xff; 8]);
        fallthrough.extend(asm(&[store, END]));
        assert_eq!(of(&fallthrough), None);

        // missing terminator fails closed; non-default F_TERMINATOR + trailing data still passes
        assert_eq!(of(&asm(&[IMAGE_STORE])), None);
        let mut saved = asm(&[IMAGE_STORE, ("S_ENDPGM_SAVED", "ENC_SOPP", &[])]);
        saved.extend_from_slice(&[0xff; 8]);
        assert_eq!(of(&saved), Some(Effects { memory: false, images: true, instructions: 2 }));
    }

    #[test]
    fn unresolved_indirect_and_oob_branches_fail_closed_while_far_jump_passes() {
        let setpc: (&str, &str, &[(&str, u64)]) = ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)]);
        let swappc: (&str, &str, &[(&str, u64)]) = ("S_SWAPPC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]);
        let oob_branch: (&str, &str, &[(&str, u64)]) = ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 8)]);
        let oob_call: (&str, &str, &[(&str, u64)]) = ("S_CALL_B64", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 8)]);
        for bad in [setpc, swappc, oob_branch, oob_call] {
            assert_eq!(of(&asm(&[IMAGE_STORE, bad, END])), None, "{}", bad.0);
        }

        // resolved far jump (base = 0x04, +16 -> 0x14) past early END to IMAGE_STORE; trailing S_CODE_END padding
        let far_ok = asm(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 0)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 16)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)]),
            setpc,
            END,
            IMAGE_STORE,
            END,
            ("S_CODE_END", "ENC_SOPP", &[]),
        ]);
        assert_eq!(of(&far_ok), Some(Effects { memory: false, images: true, instructions: 7 }));

        // resolved far jump whose target lands past the decoded slice still fails closed
        let far_oob = asm(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 0)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 128 + 64)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 128)]),
            setpc,
            END,
        ]);
        assert_eq!(of(&far_oob), None);
    }

    #[test]
    fn errored_fssm_is_rejected_by_big_and_image_only() {
        let mut code = Vec::with_capacity(BIG_CODE + 16);
        let mov = asm(&[MOV]);
        while code.len() < BIG_CODE {
            code.extend_from_slice(&mov);
        }
        code.extend(asm(&[IMAGE_STORE, END]));
        let valid = xbox::tests::sample(5, &[xbox::tests::fssm(&code, &[(6, 0)])]);
        assert!(big_and_image_only(&valid));

        // bump XBCH rsize so parse_xbch records f.error while still populating f.programs
        let mut errored = valid.clone();
        let at = errored.windows(4).position(|w| w == b"XBCH").expect("XBCH header");
        let rsize = u32::from_le_bytes(errored[at + 16..at + 20].try_into().unwrap());
        errored[at + 16..at + 20].copy_from_slice(&(rsize + 64).to_le_bytes());
        let sh = xbox::parse(&errored).expect("container still parses");
        assert!(sh.fssm[0].error.is_some() && !sh.fssm[0].programs.is_empty());
        assert!(!big_and_image_only(&errored));
    }
}
