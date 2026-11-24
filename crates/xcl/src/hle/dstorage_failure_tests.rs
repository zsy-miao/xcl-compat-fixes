//! CPU-only controls for the actual request and queue diagnostic paths.
use super::*;

#[test]
fn failure_diagnostics_distinguish_zero_and_nonzero_page_snapshots() {
    unsafe {
        let source = [0x5Au8; 16];
        for page in [0usize, 1] { // neither is a live physical allocation
            let table = [page];
            let req = Request { options: SOURCE_MEMORY_BIT | DEST_PHYS_PAGES_BIT, destination: table.as_ptr() as u64,
                destination_size: 16, source: source.as_ptr() as u64, source_size: 16, ..Default::default() };
            let mut trace = Trace::new(Some(&table), None);
            assert_eq!(execute_traced(&req, None, Some(&table), None, Some(&mut trace)), Err(E_ACCESS_VIOLATION));
            assert_eq!(execute(&req, None, Some(&table), None), Err(E_ACCESS_VIOLATION), "diagnostics preserve the failure");
            assert_eq!(trace.stage, Stage::DestinationPages);
            assert_eq!(trace.destination, Some(Counts { total: 1, zero: usize::from(page == 0) }));
            assert_eq!(trace.used, trace.destination);
            assert_eq!(trace.reread, if page == 0 { Reread::Read } else { Reread::NotNeeded });
            assert_eq!(table, [page]);
        }
    }
}

#[test]
fn failure_diagnostics_keep_unreadable_sources_and_destinations_unchanged() {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS};
    unsafe {
        let unreadable = VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
        assert!(!unreadable.is_null());
        let mut dst = [0xCDu8; 16];
        let req = Request { options: SOURCE_MEMORY_BIT, destination: dst.as_mut_ptr() as u64, destination_size: 16,
            source: unreadable as u64, source_size: 16, ..Default::default() };
        let mut trace = Trace::default();
        assert_eq!(execute_traced(&req, None, None, None, Some(&mut trace)), Err(E_ACCESS_VIOLATION));
        assert_eq!(trace.stage, Stage::SourceReadable);
        assert_eq!(dst, [0xCD; 16]);
        assert_eq!(execute(&req, None, None, None), Err(E_ACCESS_VIOLATION));
        assert_eq!(dst, [0xCD; 16]);
        let blocked = Request { source: dst.as_ptr() as u64, destination: unreadable as u64, ..req };
        assert_eq!(execute_traced(&blocked, None, None, None, Some(&mut trace)), Err(E_ACCESS_VIOLATION));
        assert_eq!(trace.stage, Stage::DestinationWritable);
        assert_eq!(dst, [0xCD; 16]);
        VirtualFree(unreadable, 0, MEM_RELEASE);
    }
}

#[test]
fn failure_diagnostics_follow_the_actual_batch_submission_origin() {
    unsafe {
        for (capacity, expected) in [(2u16, Origin::HalfFull), (32, Origin::TitleSubmit)] {
            let mut desc = [0u8; 0x18];
            desc[..4].copy_from_slice(&1u32.to_le_bytes());
            desc[8..10].copy_from_slice(&capacity.to_le_bytes());
            desc[10] = 2;
            let mut q = std::ptr::null_mut();
            assert_eq!(factory_create_queue(std::ptr::null_mut(), desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
            let diagnostics = queue(q).diagnostics.get_or_init(Diagnostics::new);
            let status = Arc::new(StatusInner { slots: vec![AtomicI32::new(E_PENDING)].into_boxed_slice() });
            // A page array still unready when the worker executes: the #549 distinction.
            let table = [0usize];
            let source = [0x5Au8; 16];
            let mut req = [0u8; REQUEST_BYTES];
            req[..8].copy_from_slice(&(SOURCE_MEMORY_BIT | DEST_PHYS_PAGES_BIT).to_le_bytes());
            req[8..16].copy_from_slice(&(table.as_ptr() as u64).to_le_bytes());
            req[0x18..0x1C].copy_from_slice(&16u32.to_le_bytes());
            req[0x20..0x28].copy_from_slice(&(source.as_ptr() as u64).to_le_bytes());
            req[0x30..0x34].copy_from_slice(&16u32.to_le_bytes());
            queue_enqueue_request(q, req.as_ptr());
            queue(q).enqueue(Entry::Status(status.clone(), 0));
            if expected == Origin::TitleSubmit { queue_submit(q); }
            let started = std::time::Instant::now();
            while status.slots[0].load(Ordering::Acquire) == E_PENDING && started.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            assert_eq!(status.slots[0].load(Ordering::Acquire), E_ACCESS_VIOLATION);
            let events = diagnostics.events.lock().unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!((events[0].origin, events[0].trace.stage), (expected, Stage::DestinationPages));
            assert_eq!(events[0].trace.destination, Some(Counts { total: 1, zero: 1 }));
            assert_eq!(events[0].trace.used, events[0].trace.destination);
            assert_eq!(events[0].trace.reread, Reread::Read);
            drop(events);
            let mut record = [0u8; ERROR_RECORD_BYTES];
            queue_retrieve_error_record(q, record.as_mut_ptr());
            assert_eq!(u32::from_le_bytes(record[..4].try_into().unwrap()), 1);
            assert_eq!(i32::from_le_bytes(record[8..12].try_into().unwrap()), E_ACCESS_VIOLATION);
            queue_close(q);
            queue_release(q);
        }
    }
}
