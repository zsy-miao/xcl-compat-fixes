//! Determinism of the direct translator (same input → byte-identical containers, in
//! separate translator instances and threads) and builder misuse reported as errors.

use super::kernel::{self, KernelOptions};
use crate::gpu::dxbc::SigElement as S;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::{CsInterface, PsInterface, VsInterface};

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> S {
    S { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// Programs touching many registers (hash-map keyed state), branches, guarded memory,
/// LDS, cross-lane reads, and VS / PS signatures.
fn translate_all() -> Vec<Vec<u8>> {
    const V: u64 = 256;
    let branchy = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)], None),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 8)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 5), ("VSRC1", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 10), ("SRC1", 129)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", 30)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 31)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 32)], None),
    ]);
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let cs = CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 1 };
    let o = |wave64, split| KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
    let mut out = Vec::new();
    for (w, s) in [(false, false), (true, true), (true, false)] {
        out.push(kernel::translate(&branchy, o(w, s)).expect("kernel"));
        out.push(kernel::translate_compute(&branchy, o(w, s), &cs).expect("compute").dxil);
    }
    out.push(kernel::translate_vertex(&vs_code, o(true, false), &vsi).expect("vs").dxil);
    out.push(kernel::translate_pixel(&ps_code, o(true, false), &psi).expect("ps").dxil);
    out
}

#[test]
fn translation_is_byte_identical_across_runs_and_threads() {
    let first = translate_all();
    for _ in 0..3 {
        assert!(translate_all() == first, "a second translation differs");
    }
    let threads: Vec<_> = (0..4).map(|_| std::thread::Builder::new().stack_size(64 << 20).spawn(translate_all).unwrap()).collect();
    for t in threads {
        assert!(t.join().unwrap() == first, "a translation on another thread differs");
    }
}

// ---- builder misuse: recorded as the first error, finish fails instead of panicking

use super::container::Stage;
use super::module::{BinOp, FnAttrs, Module};
use super::shader::Shader;

/// A compute shader with `misuse` applied fails to finish with an error containing `want`.
fn finish_fails(want: &str, misuse: impl FnOnce(&mut Shader)) {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    misuse(&mut s);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.try_finish([1, 1, 1], None)));
    let e = r.expect("try_finish panicked").expect_err("misuse was not reported");
    assert!(e.contains(want), "error {e:?} does not name {want:?}");
}

#[test]
fn forward_reference_outside_a_phi_is_an_error() {
    finish_fails("forward reference", |s| {
        let later = s.m.new_block();
        let entry = s.m.current_block();
        s.m.set_block(later);
        let one = s.m.i32c(1);
        let two = s.m.i32c(2);
        let v = s.m.bin(BinOp::Add, one, two);
        s.m.ret();
        s.m.set_block(entry);
        let _ = s.m.bin(BinOp::Add, v, one);
        s.m.br(later);
    });
}

#[test]
fn instruction_after_a_terminator_is_an_error() {
    finish_fails("after a terminator", |s| {
        s.m.ret();
        let one = s.m.i32c(1);
        let _ = s.m.bin(BinOp::Add, one, one);
    });
}

#[test]
fn unknown_overload_suffix_is_an_error() {
    finish_fails("overload suffix", |s| {
        let i32 = s.m.i32();
        let arr = s.m.array(4, i32);
        let _ = s.type_suffix(arr);
    });
}

#[test]
fn gep_and_extractvalue_into_wrong_types_are_errors() {
    finish_fails("gep into", |s| {
        let i32 = s.m.i32();
        let g = s.m.global("g", i32, 3, None, false, 4);
        let z = s.m.i32c(0);
        let _ = s.m.gep(g, &[z, z]);
    });
    finish_fails("gep: struct field", |s| {
        let i32 = s.m.i32();
        let st = s.m.named_struct("S", &[i32]);
        let g = s.m.global("g", st, 3, None, false, 4);
        let z = s.m.i32c(0);
        let k = s.m.i32c(5);
        let _ = s.m.gep_fields(g, &[z, k]);
    });
    finish_fails("not a pointer", |s| {
        let one = s.m.i32c(1);
        let _ = s.m.load(one, 4);
    });
    finish_fails("extractvalue", |s| {
        let one = s.m.i32c(1);
        let _ = s.m.extract(one, 0);
    });
}

#[test]
fn redeclared_function_and_bad_calls_are_errors() {
    finish_fails("redeclared", |s| {
        let v = s.m.void();
        let i32 = s.m.i32();
        let a = s.m.func_ty(v, &[]);
        let b = s.m.func_ty(v, &[i32]);
        s.m.declare("f", a, FnAttrs::None);
        s.m.declare("f", b, FnAttrs::None);
    });
    finish_fails("arguments", |s| {
        let v = s.m.void();
        let t = s.m.func_ty(v, &[]);
        let f = s.m.declare("f", t, FnAttrs::None);
        let one = s.m.i32c(1);
        s.m.call(f, &[one]);
    });
    finish_fails("second defined function", |s| {
        s.m.define_entry("other");
    });
}

#[test]
fn bad_constants_phis_and_stores_are_errors() {
    finish_fails("const_int", |s| {
        let f = s.m.f32();
        s.m.const_int(f, 1);
    });
    finish_fails("const_float", |s| {
        let i = s.m.i32();
        s.m.const_float(i, 1);
    });
    finish_fails("non-phi", |s| {
        let one = s.m.i32c(1);
        let v = s.m.bin(BinOp::Add, one, one);
        let b = s.m.current_block();
        s.m.add_incoming(v, one, b);
    });
    finish_fails("rawBufferStore of 5", |s| {
        let u = s.raw_uav(0, 0);
        let z = s.m.i32c(0);
        s.raw_store(u, z, &[z; 5]);
    });
}

#[test]
fn use_of_a_void_value_and_wrong_finish_are_errors() {
    finish_fails("void instruction", |s| {
        let u = s.raw_uav(0, 0);
        let z = s.m.i32c(0);
        s.raw_store(u, z, &[z]);
        // the store's (void) result as an operand
        let st = super::module::Value::Inst(s.m.block_insts(s.m.current_block()).last().copied().unwrap());
        let i32 = s.m.i32();
        let _ = s.m.cast(super::module::CastOp::BitCast, st, i32);
    });
    let s = Shader::new(Stage::Pixel, (6, 6));
    assert!(s.try_finish([1, 1, 1], None).is_err());
    let mut m = Module::new();
    m.ret();
    m.ret();
    assert!(m.try_to_bitcode("dxil-ms-dx", "").is_err());
}
