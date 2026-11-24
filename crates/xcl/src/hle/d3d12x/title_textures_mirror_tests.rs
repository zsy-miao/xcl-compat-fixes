use super::*;
use crate::hle::kernelx::{self, mirrors};
use windows_sys::Win32::System::Memory as mem;

static TEST_LOCK: Mutex<()> = Mutex::new(());
// Larger than all four 4 GiB UMA windows together: the public allocation hook MUST
// take its Windows fallback. This reserves address space, not 24 GiB of RAM.
const RESERVATION: usize = 24usize << 30;
const OFFSET: usize = 0x2_0000;
const PAGE: usize = 0x1_0000;

struct Reservation(usize);
impl Drop for Reservation {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe { kernelx::VirtualFree(self.0 as *mut c_void, 0, mem::MEM_RELEASE); }
        }
    }
}

unsafe fn reserve(flags: u32, protect: u32) -> Reservation {
    let at = kernelx::VirtualAlloc(std::ptr::null(), RESERVATION, flags, protect) as usize;
    assert_ne!(at, 0, "reserve-only fallback must succeed");
    let r = Reservation(at);
    let u = crate::gpu::uma::get().expect("UMA initialized");
    assert!(!u.contains(at), "test must use the Windows fallback, not a GPU window");
    assert!(kernelx::in_graphics_reservation(at), "fallback must be write-watched");
    r
}

#[test]
fn gpu_visible_fallback_is_forced_and_reports_cpu_writes() {
    assert_eq!(super::super::backend(), super::super::Backend::Real, "real backend required");
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        for (flags, protect) in [(0x2040_2000, mem::PAGE_READWRITE), (0x2000_2000, 0x2404), (kernelx::MEM_GRAPHICS | mem::MEM_RESERVE, mem::PAGE_READWRITE)] {
            let r = reserve(flags, protect);
            let at = r.0 + OFFSET;
            assert_eq!(kernelx::VirtualAlloc(at as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, at);
            let mut pages = [std::ptr::null_mut(); 32];
            let (mut count, mut granularity) = (pages.len(), 0);
            assert_eq!(mem::GetWriteWatch(1, at as *const c_void, PAGE, pages.as_mut_ptr(), &mut count, &mut granularity), 0);
            ((at + 0x1234) as *mut u8).write_volatile(7);
            count = pages.len();
            assert_eq!(mem::GetWriteWatch(1, at as *const c_void, PAGE, pages.as_mut_ptr(), &mut count, &mut granularity), 0);
            assert_eq!(&pages[..count], &[(at + 0x1000) as *mut c_void]);
            let base = r.0;
            drop(r);
            assert!(!kernelx::in_graphics_reservation(base), "release removes the watched range");
        }
    }
}

#[test]
fn gpu_mirror_pairing_preserves_offsets_and_release() {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (cpu, gpu, other, size) = (0x6A00_0000_0000, 0x6A10_0000_0000, 0x6A20_0000_0000, 0x80_0000);
    mirrors::reserved(cpu, size, 0x2040_2000, mem::PAGE_READWRITE);
    mirrors::reserved(other, size / 2, 0x2040_2000, mem::PAGE_READWRITE);
    mirrors::reserved(gpu, size, 0x2040_2000, 0x801);
    assert_eq!(mirrors::source(gpu), Some(cpu));
    assert_eq!(mirrors::source(gpu + OFFSET), Some(cpu + OFFSET));
    assert_eq!(mirrors::source(gpu + size), None);
    assert_eq!(mirrors::source(cpu), None);
    mirrors::released(gpu);
    assert_eq!(mirrors::source(gpu), None);
    mirrors::reserved(gpu, size, 0x2040_2000, 0x801);
    assert_eq!(mirrors::source(gpu), Some(cpu));
    mirrors::released(cpu);
    assert_eq!(mirrors::source(gpu), None, "releasing either view removes the pair");
    mirrors::released(other);
}
