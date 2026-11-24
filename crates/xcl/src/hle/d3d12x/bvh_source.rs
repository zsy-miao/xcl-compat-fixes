//! Address provenance for a refused acceleration-structure view. This queries
//! region metadata only: it never reads nodes, commits pages or chooses an alias.

use std::ffi::c_void;
use std::fmt;
use windows_sys::Win32::System::Memory::{
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RESERVE,
};

pub(super) struct Probe {
    address: u64,
    in_uma: bool,
    physical_page_mapped: bool,
    region: Option<MEMORY_BASIC_INFORMATION>,
}

pub(super) fn probe(address: u64, in_uma: bool, physical_page_mapped: bool) -> Probe {
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let region = usize::try_from(address)
        .ok()
        .filter(|&at| at != 0)
        .and_then(|at| {
            let got = unsafe {
                VirtualQuery(at as *const c_void, &mut info, std::mem::size_of_val(&info))
            };
            (got == std::mem::size_of_val(&info)).then_some(info)
        });
    Probe {
        address,
        in_uma: address != 0 && in_uma,
        physical_page_mapped: address != 0 && physical_page_mapped,
        region,
    }
}

impl fmt::Display for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "address={:#x}, UMA={}, physical_page_mapped={}",
            self.address, self.in_uma, self.physical_page_mapped
        )?;
        if let Some(r) = &self.region {
            let state = match r.State {
                MEM_COMMIT => "committed",
                MEM_RESERVE => "reserved",
                MEM_FREE => "free",
                _ => "unknown",
            };
            write!(f, ", region={:#x}+{:#x}, allocation={:#x}, state={state}, protect={:#x}, allocation_protect={:#x}",
                r.BaseAddress as usize, r.RegionSize, r.AllocationBase as usize, r.Protect, r.AllocationProtect)
        } else {
            write!(f, ", region=unavailable")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, MEM_RELEASE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE,
    };

    struct Reservation(*mut c_void);
    impl Drop for Reservation {
        fn drop(&mut self) {
            unsafe {
                VirtualFree(self.0, 0, MEM_RELEASE);
            }
        }
    }

    #[test]
    fn bvh_source_probe_does_not_touch_reserved_or_guarded_memory() {
        unsafe {
            let allocation = Reservation(VirtualAlloc(
                std::ptr::null(),
                0x1_0000,
                MEM_RESERVE,
                PAGE_NOACCESS,
            ));
            assert!(!allocation.0.is_null());
            let at = allocation.0 as u64;
            assert_eq!(probe(at, false, false).region.unwrap().State, MEM_RESERVE);
            assert!(
                !VirtualAlloc(allocation.0, 4096, MEM_COMMIT, PAGE_READWRITE | PAGE_GUARD)
                    .is_null()
            );
            let report = probe(at, false, true);
            assert_eq!(report.region.as_ref().unwrap().State, MEM_COMMIT);
            assert_ne!(report.region.as_ref().unwrap().Protect & PAGE_GUARD, 0);
            // A read would consume the guard; repeated metadata queries must not.
            assert_ne!(
                probe(at, false, true).region.unwrap().Protect & PAGE_GUARD,
                0
            );
            assert!(report.to_string().contains("physical_page_mapped=true"));
        }
    }

    #[test]
    fn bvh_source_probe_preserves_the_offset_and_full_address_of_each_view() {
        const SIZE: usize = (8usize << 30) + 0x1_0000;
        const OFFSET: usize = 0x1_b2cc_0000; // observed AS offset within its GPU reservation
        unsafe {
            let cpu = Reservation(VirtualAlloc(
                std::ptr::null(),
                SIZE,
                MEM_RESERVE,
                PAGE_READWRITE,
            ));
            let gpu = Reservation(VirtualAlloc(
                std::ptr::null(),
                SIZE,
                MEM_RESERVE,
                PAGE_NOACCESS,
            ));
            assert!(!cpu.0.is_null() && !gpu.0.is_null());
            let cpu_at = cpu.0 as usize + OFFSET;
            let gpu_at = gpu.0 as usize + OFFSET;
            assert!(
                !VirtualAlloc(cpu_at as *const c_void, 4096, MEM_COMMIT, PAGE_READWRITE).is_null()
            );
            let source = probe(gpu_at as u64, false, false);
            let candidate = probe(cpu_at as u64, false, true);
            assert_eq!(source.address, gpu_at as u64);
            assert_eq!(candidate.address, cpu_at as u64);
            assert_eq!(source.region.as_ref().unwrap().AllocationBase, gpu.0);
            assert_eq!(candidate.region.as_ref().unwrap().AllocationBase, cpu.0);
            assert_eq!(source.region.as_ref().unwrap().State, MEM_RESERVE);
            assert_eq!(candidate.region.as_ref().unwrap().State, MEM_COMMIT);
            assert!(
                !source.in_uma && !candidate.in_uma,
                "a CPU candidate is not proof of GPU residency"
            );
            assert!(source.to_string().contains(&format!("address={gpu_at:#x}")));
            assert!(candidate
                .to_string()
                .contains(&format!("address={cpu_at:#x}")));
        }
    }

    #[test]
    fn bvh_source_probe_null_has_no_region_or_mapping_claim() {
        let report = probe(0, true, true);
        assert!(report.region.is_none());
        assert!(!report.in_uma && !report.physical_page_mapped);
        assert!(report.to_string().contains("region=unavailable"));
    }
}
