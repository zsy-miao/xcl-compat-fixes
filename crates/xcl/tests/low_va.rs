//! xcl.exe opts out of high-entropy ASLR, so the title's memory stays below 2^40 (#585).

/// IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA
const HIGH_ENTROPY_VA: u16 = 0x20;

#[test]
fn xcl_exe_is_linked_without_high_entropy_va() {
    let image = std::fs::read(env!("CARGO_BIN_EXE_xcl")).unwrap();
    let pe = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    assert_eq!(&image[pe..pe + 4], b"PE\0\0");
    let optional = pe + 24;
    assert_eq!(u16::from_le_bytes(image[optional..optional + 2].try_into().unwrap()), 0x20b, "PE32+");
    // DllCharacteristics is at +70 of the PE32+ optional header
    let flags = u16::from_le_bytes(image[optional + 70..optional + 72].try_into().unwrap());
    assert_eq!(flags & HIGH_ENTROPY_VA, 0, "DllCharacteristics {flags:#x}: high-entropy VA is on");
}
