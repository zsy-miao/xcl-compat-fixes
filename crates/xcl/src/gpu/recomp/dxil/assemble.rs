//! A DXIL module as LLVM IR text (or bitcode) to a signed container, through the PC DXC's
//! libraries: `dxcompiler.dll`'s assembler builds the container from the module's `dx.*`
//! metadata, `dxil.dll`'s validator checks and signs it. Neither DXC's HLSL front end nor its
//! optimiser runs: a retail compute program's module (8.6 MB of IR, 1 MB of DXIL) takes 0.75 s,
//! where its HLSL takes 25 s through `dxc::compile`.
//!
//! What a module needs on top of what the validator checks (RX 7800 XT, Windows driver):
//! an `!llvm.ident`. Without one the container validates and signs, but
//! `CreateComputePipelineState` faults inside the AMD driver.

use std::ffi::c_void;

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D::Dxc::{
    DxcValidatorFlags_InPlaceEdit, IDxcAssembler, IDxcOperationResult, IDxcUtils, IDxcValidator, CLSID_DxcAssembler, CLSID_DxcUtils, CLSID_DxcValidator, DXC_CP_UTF8,
};

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

fn errors(r: &IDxcOperationResult) -> String {
    unsafe { r.GetErrorBuffer().map(|b| String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize())).into_owned()).unwrap_or_default() }
}

unsafe fn make<T: Interface>(create: CreateInstance, clsid: &GUID) -> Result<T, String> {
    let mut p = std::ptr::null_mut();
    create(clsid, &T::IID, &mut p).ok().map_err(|e| format!("DxcCreateInstance: {e}"))?;
    Ok(T::from_raw(p))
}

/// The module `ir` as a validated, signed DXIL container; the assembler's or the validator's
/// messages as the error.
pub fn assemble(ir: &str) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    let create = super::super::dxc::create_instance().ok_or("dxcompiler.dll not found")?;
    unsafe {
        let create = std::mem::transmute::<usize, CreateInstance>(create);
        // (`dxc` loaded dxil.dll by its full path, before dxcompiler.dll)
        let signer = GetModuleHandleW(crate::util::wide("dxil.dll").as_ptr());
        let sign = (!signer.is_null()).then(|| GetProcAddress(signer, b"DxcCreateInstance\0".as_ptr())).flatten().ok_or("assemble: no dxil.dll to sign with")?;
        let utils: IDxcUtils = make(create, &CLSID_DxcUtils)?;
        let blob = utils.CreateBlob(ir.as_ptr() as *const c_void, ir.len() as u32, DXC_CP_UTF8).map_err(|e| format!("assemble: {e}"))?;
        let asm: IDxcAssembler = make(create, &CLSID_DxcAssembler)?;
        let r = asm.AssembleToContainer(&blob).map_err(|e| format!("assemble: {e}"))?;
        if r.GetStatus().map_err(|e| format!("assemble: {e}"))?.is_err() {
            return Err(format!("assemble: {}", errors(&r)));
        }
        let container = r.GetResult().map_err(|e| format!("assemble: {e}"))?;
        let v: IDxcValidator = make(std::mem::transmute::<usize, CreateInstance>(sign as usize), &CLSID_DxcValidator)?;
        let r = v.Validate(&container, DxcValidatorFlags_InPlaceEdit).map_err(|e| format!("validate: {e}"))?;
        if r.GetStatus().map_err(|e| format!("validate: {e}"))?.is_err() {
            return Err(format!("validate: {}", errors(&r)));
        }
        Ok(std::slice::from_raw_parts(container.GetBufferPointer() as *const u8, container.GetBufferSize()).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::runner;

    /// A CS of the ABI root signature, written by hand: `mem[i] = 3 i + 1` through `u2`.
    const IR: &str = r#"target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"
%dx.types.Handle = type { i8* }
%dx.types.ResBind = type { i32, i32, i32, i8 }
%dx.types.ResourceProperties = type { i32, i32 }
%struct.RWByteAddressBuffer = type { i32 }
define void @main() {
  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 2, i32 2, i32 0, i8 1 }, i32 2, i1 false)
  %2 = call i32 @dx.op.threadId.i32(i32 93, i32 0)
  %3 = mul i32 %2, 3
  %4 = add i32 %3, 1
  %5 = shl i32 %2, 2
  %6 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 4107, i32 0 })
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %6, i32 %5, i32 undef, i32 %4, i32 undef, i32 undef, i32 undef, i8 1, i32 4)
  ret void
}
declare i32 @dx.op.threadId.i32(i32, i32) #0
declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32) #1
declare %dx.types.Handle @dx.op.annotateHandle(i32, %dx.types.Handle, %dx.types.ResourceProperties) #0
declare %dx.types.Handle @dx.op.createHandleFromBinding(i32, %dx.types.ResBind, i32, i1) #0
attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!7}
!0 = !{!"xcl"}
!1 = !{i32 1, i32 6}
!2 = !{i32 1, i32 6}
!3 = !{!"cs", i32 6, i32 6}
!4 = !{null, !5, null, null}
!5 = !{!6}
!6 = !{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 2, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!7 = !{void ()* @main, !"main", null, !4, !8}
!8 = !{i32 0, i64 16, i32 4, !9}
!9 = !{i32 64, i32 1, i32 1}
"#;

    /// The text assembles, signs and runs on the GPU; a module the validator rejects (a use
    /// before its definition) is an error, not a container.
    #[test]
    fn ir_text_assembles_signs_and_runs() {
        if super::super::super::dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        // a Windows SDK's `bin\<ver>\x64` has dxcompiler.dll without dxil.dll: nothing to sign with
        let dxil = match super::assemble(IR) {
            Err(e) if e.contains("no dxil.dll") => {
                runner::skip_test("no dxil.dll");
                return;
            }
            r => r.unwrap(),
        };
        assert_eq!(&dxil[..4], b"DXBC");
        let bad = super::assemble(&IR.replace("%3 = mul i32 %2, 3", "%3 = mul i32 %4, 3")).unwrap_err();
        assert!(bad.contains("dominate"), "{bad}");
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return;
        }
        let out = runner::dispatch(&dxil, [1, 1, 1], &[], &[0u32; 64]).unwrap();
        let want: Vec<u32> = (0..64).map(|i| 3 * i + 1).collect();
        assert_eq!(&out[..64], &want[..]);
    }
}
