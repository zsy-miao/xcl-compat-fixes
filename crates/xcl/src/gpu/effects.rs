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
/// None: an instruction that does not decode, so nothing can be said about what follows it.
pub fn of(code: &[u8]) -> Option<Effects> {
    let mut e = Effects::default();
    let mut off = 0;
    // the whole program: code can follow an early S_ENDPGM (a branch to it, or past it); what
    // does not decode after one is taken for trailing data, before one it says nothing
    let mut ended = false;
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else {
            if ended {
                break;
            }
            return None;
        };
        off += d.len;
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
        ended |= n == "S_ENDPGM";
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
}
