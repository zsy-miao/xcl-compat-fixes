use super::*;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT,
    MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE,
};

const PAGE: usize = crate::hle::kernelx::PHYS_PAGE_SIZE;

struct Region(usize);

impl Region {
    unsafe fn reserve(pages: usize, deferred: bool) -> Self {
        let at = VirtualAlloc(std::ptr::null(), pages * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
        assert_ne!(at, 0);
        if deferred {
            for k in 0..pages {
                crate::hle::kernelx::lazy::mark(at + k * PAGE);
            }
        }
        Self(at)
    }

    unsafe fn info(&self) -> MEMORY_BASIC_INFORMATION {
        let mut info = std::mem::zeroed();
        assert_ne!(
            VirtualQuery(
                self.0 as *const c_void,
                &mut info,
                std::mem::size_of_val(&info)
            ),
            0
        );
        info
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe {
            assert_ne!(VirtualFree(self.0 as *mut c_void, 0, MEM_RELEASE), 0);
            crate::hle::kernelx::lazy::forget_released(self.0);
        }
    }
}

unsafe fn copy_from(at: usize, destination: &mut [u8]) -> Result<(), HRESULT> {
    let request = Request {
        options: SOURCE_MEMORY_BIT,
        destination: destination.as_mut_ptr() as u64,
        destination_size: destination.len() as u32,
        source: at as u64,
        source_size: destination.len() as u32,
        ..Default::default()
    };
    assert_eq!(request.check(true), Ok(()));
    execute(&request, None, None, false, None)
}

#[test]
fn a_deferred_zero_memory_source_is_readable_to_directstorage() {
    unsafe {
        let region = Region::reserve(1, true);
        assert_eq!(region.info().State, MEM_RESERVE);
        let mut destination = [0xCCu8; 64];
        assert_eq!(copy_from(region.0 + 8, &mut destination), Ok(()));
        assert_eq!(destination, [0; 64]);
        assert_eq!(region.info().State, MEM_COMMIT);
    }
}

#[test]
fn a_deferred_page_array_snapshot_owns_the_initial_zero_entries() {
    unsafe {
        let region = Region::reserve(2, true);
        let table = region.0 + PAGE - std::mem::size_of::<usize>();
        assert_eq!(region.info().State, MEM_RESERVE);
        let snapshot = snapshot_page_array(table as u64, (PAGE - 4) as u16, 8)
            .expect("a mapped lazy page array logically holds zeros");
        assert_eq!(snapshot, [0, 0]);
        (table as *mut usize).write(16);
        ((table + std::mem::size_of::<usize>()) as *mut usize).write(32);
        assert_eq!(snapshot, [0, 0], "the enqueue snapshot remains owned");
        assert_eq!(
            snapshot_page_array(table as u64, 0, PAGE as u32 * 2),
            Ok(vec![16, 32])
        );
    }
}

#[test]
fn an_untracked_reservation_stays_unreadable_and_unmodified() {
    unsafe {
        let region = Region::reserve(1, false);
        let mut destination = [0xCCu8; 64];
        assert_eq!(
            copy_from(region.0, &mut destination),
            Err(E_ACCESS_VIOLATION)
        );
        assert_eq!(destination, [0xCC; 64]);
        assert_eq!(
            snapshot_page_array(region.0 as u64, 0, 8),
            Err(E_ACCESS_VIOLATION)
        );
        assert_eq!(region.info().State, MEM_RESERVE);
        assert_eq!(
            snapshot_page_array((usize::MAX - 3) as u64, 0, 8),
            Err(E_ACCESS_VIOLATION)
        );
    }
}

#[test]
fn deferred_read_preparation_does_not_widen_a_partly_committed_page() {
    unsafe {
        let region = Region::reserve(1, true);
        assert!(
            !VirtualAlloc(region.0 as *const c_void, 4096, MEM_COMMIT, PAGE_READONLY).is_null()
        );
        let mut destination = [0xCCu8; 64];
        assert_eq!(
            copy_from(region.0 + 4096, &mut destination),
            Err(E_ACCESS_VIOLATION)
        );
        assert_eq!(
            snapshot_page_array((region.0 + 4096) as u64, 0, 8),
            Err(E_ACCESS_VIOLATION)
        );
        assert_eq!(destination, [0xCC; 64]);
        let first = region.info();
        assert_eq!(
            (first.State, first.Protect, first.RegionSize),
            (MEM_COMMIT, PAGE_READONLY, 4096)
        );
        let mut rest = std::mem::zeroed();
        assert_ne!(
            VirtualQuery(
                (region.0 + 4096) as *const c_void,
                &mut rest,
                std::mem::size_of_val(&rest)
            ),
            0
        );
        assert_eq!(rest.State, MEM_RESERVE);
    }
}

#[test]
fn deferred_read_preparation_preserves_explicit_page_protections() {
    unsafe {
        for protect in [PAGE_READONLY, PAGE_NOACCESS, PAGE_READWRITE | PAGE_GUARD] {
            let region = Region::reserve(1, true);
            assert!(
                !VirtualAlloc(region.0 as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE)
                    .is_null()
            );
            let mut old = 0;
            assert_ne!(
                VirtualProtect(region.0 as *const c_void, PAGE, protect, &mut old),
                0
            );
            let mut destination = [0xCCu8; 64];
            let expected = if protect == PAGE_READONLY {
                Ok(())
            } else {
                Err(E_ACCESS_VIOLATION)
            };
            assert_eq!(copy_from(region.0, &mut destination), expected);
            if expected.is_ok() {
                assert_eq!(destination, [0; 64]);
                assert_eq!(snapshot_page_array(region.0 as u64, 0, 8), Ok(vec![0]));
            } else {
                assert_eq!(destination, [0xCC; 64]);
                assert_eq!(
                    snapshot_page_array(region.0 as u64, 0, 8),
                    Err(E_ACCESS_VIOLATION)
                );
            }
            assert_eq!(region.info().Protect, protect);
        }
    }
}

#[test]
fn a_read_retries_when_another_thread_materializes_the_deferred_page() {
    use std::sync::{Arc, Barrier};
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    unsafe {
        for native in [false, true] {
            let region = Region::reserve(1, true);
            let at = region.0;
            let begin = Arc::new(Barrier::new(2));
            let done = Arc::new(Barrier::new(2));
            let other = {
                let begin = Arc::clone(&begin);
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    begin.wait();
                    assert_eq!((at as *const u8).read_volatile(), 0);
                    done.wait();
                })
            };
            let mut attempts = 0;
            let mut bytes = [0xCCu8; 16];
            let ok = read_prepared(
                || {
                    attempts += 1;
                    if native {
                        let mut copied = 0;
                        ReadProcessMemory(
                            GetCurrentProcess(),
                            at as *const c_void,
                            bytes.as_mut_ptr() as *mut c_void,
                            bytes.len(),
                            &mut copied,
                        ) != 0
                            && copied == bytes.len()
                    } else {
                        crate::gpu::shader::readable(at, bytes.len())
                    }
                },
                || {
                    begin.wait();
                    done.wait();
                    let materialized = crate::hle::kernelx::lazy::prepare_read(at, 16);
                    assert!(
                        !materialized,
                        "the other thread already removed the lazy mark"
                    );
                    materialized
                },
            );
            other.join().unwrap();
            assert!(
                ok,
                "readable/native reads must recheck after a competing first touch"
            );
            assert_eq!(attempts, 2);
            if native {
                assert_eq!(bytes, [0; 16]);
            }
            assert_eq!(copy_from(at, &mut bytes), Ok(()));
            assert_eq!(bytes, [0; 16]);
        }
    }
}
