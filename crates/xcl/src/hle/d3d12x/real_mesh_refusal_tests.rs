use super::*;

/// A small Xbox DXBC/DXIL container: the program kind and XDAT are sufficient for the
/// stream's stage/layout decision; no native program is compiled by these CPU tests.
fn console_shader(kind: u32) -> Vec<u8> {
    let mut dxil = vec![0u8; 0x18];
    dxil[0..4].copy_from_slice(&(kind << 16 | 0x66).to_le_bytes());
    dxil[8..12].copy_from_slice(b"DXIL");
    dxil.extend_from_slice(b"XDAT");
    dxil.extend_from_slice(&[0; 8]);
    let mut blob = vec![0u8; 36];
    blob[0..4].copy_from_slice(b"DXBC");
    blob[20..24].copy_from_slice(&1u32.to_le_bytes());
    blob[28..32].copy_from_slice(&1u32.to_le_bytes());
    blob[32..36].copy_from_slice(&36u32.to_le_bytes());
    blob.extend_from_slice(b"DXIL");
    blob.extend_from_slice(&(dxil.len() as u32).to_le_bytes());
    blob.extend_from_slice(&dxil);
    let bytes = blob.len() as u32;
    blob[24..28].copy_from_slice(&bytes.to_le_bytes());
    blob
}

fn desc(stream: &StreamOut) -> [usize; 2] {
    [stream.0.len(), stream.0.as_ptr() as usize]
}

#[test]
fn console_ms_in_as_slot_is_refused_before_renumbering() {
    let ms = console_shader(13);
    assert!(crate::gpu::recomp::pso::is_console(&ms));
    assert_eq!(crate::gpu::xbox::parse(&ms).unwrap().stage, "MS");
    let mut stream = StreamOut::default();
    stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, &ms);
    let descriptor = desc(&stream);
    let raw = descriptor.as_ptr() as *const c_void;
    unsafe {
        assert!(unsupported_console_mesh_stream(raw).is_some());
        let mut output = 1usize as *mut c_void;
        // No device exists: the actual entry point must reject before accessing a device
        // or translating/queuing the shader, and must not return a fake pipeline object.
        assert_eq!(dev_create_pipeline_state(std::ptr::null_mut(), raw, &GUID::from_u128(0), &mut output), E_INVALIDARG);
        assert!(output.is_null());
    }
}

#[test]
fn console_ms_in_ms_slot_remains_supported() {
    let ms = console_shader(13);
    let mut stream = StreamOut::default();
    stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
    let descriptor = desc(&stream);
    unsafe {
        assert!(unsupported_console_mesh_stream(descriptor.as_ptr() as *const c_void).is_none());
    }
}

#[test]
fn console_combined_as_and_ms_is_refused_instead_of_dropping_as() {
    let ms = console_shader(13);
    let amplification = console_shader(14);
    let mut stream = StreamOut::default();
    stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, &amplification);
    stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
    let descriptor = desc(&stream);
    unsafe {
        let mut output = 1usize as *mut c_void;
        assert_eq!(dev_create_pipeline_state(std::ptr::null_mut(), descriptor.as_ptr() as *const c_void, &GUID::from_u128(0), &mut output), E_INVALIDARG);
        assert!(output.is_null());
    }
}

#[test]
fn pc_amplification_and_mesh_stream_is_not_refused() {
    let mut ms = console_shader(13);
    let mut amplification = console_shader(14);
    // PC shader metadata has no XDAT. Keep its genuine MS / AS program kinds.
    ms[44 + 0x18..44 + 0x1C].copy_from_slice(b"PCBC");
    amplification[44 + 0x18..44 + 0x1C].copy_from_slice(b"PCBC");
    assert!(!crate::gpu::recomp::pso::is_console(&ms));
    let mut stream = StreamOut::default();
    stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, &amplification);
    stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
    let descriptor = desc(&stream);
    unsafe {
        assert!(unsupported_console_mesh_stream(descriptor.as_ptr() as *const c_void).is_none());
    }
}
