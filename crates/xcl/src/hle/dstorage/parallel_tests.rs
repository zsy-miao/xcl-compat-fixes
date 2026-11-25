use super::super::*;
use super::*;
use std::io::Write;
use std::os::windows::ffi::OsStrExt;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr::null_mut;
use std::thread;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(5);
const TEST_BUDGET: usize = 1024 * 1024;

fn with_deadline(test: impl FnOnce() + Send + 'static) {
    // Leave a deadlocked fixture's resources alive so a failing test cannot free
    // a destination still owned by a worker or hang libtest while closing it.
    let (done, result) = mpsc::sync_channel(1);
    let thread = thread::spawn(move || {
        let _ = done.send(panic::catch_unwind(AssertUnwindSafe(test)));
    });
    let result = result
        .recv_timeout(TIMEOUT * 3)
        .expect("DirectStorage fixture did not quiesce");
    thread.join().unwrap();
    if let Err(error) = result {
        panic::resume_unwind(error);
    }
}

struct TestFile {
    raw: *mut c_void,
    path: PathBuf,
    chunks: Vec<(u64, u32, u32)>,
}

impl TestFile {
    fn new(chunks: impl IntoIterator<Item = (Vec<u8>, usize)>) -> Self {
        static NEXT_FILE: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "xcl-dstorage-parallel-{}-{}.zlib",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut test_file = Self {
            raw: null_mut(),
            path,
            chunks: Vec::new(),
        };
        let mut file = std::fs::File::create(&test_file.path).unwrap();
        let mut offset = 0;
        for (bytes, decoded) in chunks {
            assert!(bytes.len() < decoded);
            file.write_all(&bytes).unwrap();
            test_file
                .chunks
                .push((offset, bytes.len() as u32, decoded as u32));
            offset += bytes.len() as u64;
        }
        drop(file);
        let path: Vec<_> = test_file
            .path
            .as_os_str()
            .encode_wide()
            .chain([0])
            .collect();
        assert_eq!(
            unsafe { factory_open_file(null_mut(), path.as_ptr(), &IID_FILE, &mut test_file.raw) },
            S_OK
        );
        test_file
    }

    fn compressed(payloads: &[&[u8]]) -> Self {
        Self::new(payloads.iter().map(|bytes| {
            (
                miniz_oxide::deflate::compress_to_vec_zlib(bytes, 6),
                bytes.len(),
            )
        }))
    }

    fn request(&self, chunk: usize, destination: usize, tag: u64) -> [u8; REQUEST_BYTES] {
        let (offset, compressed, size) = self.chunks[chunk];
        let mut bytes = [0; REQUEST_BYTES];
        bytes[0..8].copy_from_slice(&ZLIB_BIT.to_le_bytes());
        bytes[8..16].copy_from_slice(&(destination as u64).to_le_bytes());
        bytes[0x18..0x1c].copy_from_slice(&size.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&(self.raw as u64).to_le_bytes());
        bytes[0x28..0x30].copy_from_slice(&offset.to_le_bytes());
        bytes[0x30..0x34].copy_from_slice(&compressed.to_le_bytes());
        bytes[0x38..0x40].copy_from_slice(&tag.to_le_bytes());
        bytes
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                file_close(self.raw);
                file_release(self.raw);
            }
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

struct TestQueue(*mut c_void);

impl TestQueue {
    fn new(pool: Option<Arc<DecompressPool>>, hook: Option<TestHook>) -> Self {
        let mut desc = [0; 0x18];
        desc[8..10].copy_from_slice(&64u16.to_le_bytes());
        let mut raw = null_mut();
        assert_eq!(
            unsafe { factory_create_queue(null_mut(), desc.as_ptr(), &IID_QUEUE, &mut raw) },
            S_OK
        );
        let queue = Self(raw);
        assert!(queue.inner().pool_override.set(pool).is_ok());
        if let Some(hook) = hook {
            assert!(queue.inner().test_hook.set(hook).is_ok());
        }
        queue
    }

    fn inner(&self) -> &Arc<QueueInner> {
        unsafe { queue(self.0) }
    }

    /// The destination must remain live until the request retires or the queue closes.
    unsafe fn enqueue(&self, request: &[u8; REQUEST_BYTES]) {
        unsafe { queue_enqueue_request(self.0, request.as_ptr()) };
    }

    fn status(&self, status: &TestStatus, slot: u32) {
        unsafe { queue_enqueue_status(self.0, status.0, slot) };
    }

    fn submit(&self) {
        unsafe { queue_submit(self.0) };
    }

    fn cancel(&self, tag: u64) {
        unsafe { queue_cancel(self.0, u64::MAX, tag) };
    }
}

impl Drop for TestQueue {
    fn drop(&mut self) {
        unsafe {
            queue_close(self.0);
            queue_release(self.0);
        }
    }
}

struct TestStatus(*mut c_void);

impl TestStatus {
    fn new(slots: u32) -> Self {
        let mut raw = null_mut();
        assert_eq!(
            unsafe {
                factory_create_status_array(
                    null_mut(),
                    slots,
                    std::ptr::null(),
                    &IID_STATUS_ARRAY,
                    &mut raw,
                )
            },
            S_OK
        );
        Self(raw)
    }

    fn result(&self, slot: u32) -> HRESULT {
        unsafe { status_get_hresult(self.0, slot) }
    }

    fn wait(&self, slot: u32) -> HRESULT {
        wait_until(|| unsafe { status_is_complete(self.0, slot) });
        self.result(slot)
    }
}

impl Drop for TestStatus {
    fn drop(&mut self) {
        unsafe { status_release(self.0) };
    }
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + TIMEOUT;
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "DirectStorage milestone timed out"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn test_pool(workers: usize, budget: usize) -> Arc<DecompressPool> {
    DecompressPool::new(workers, budget).expect("test decompression workers failed to start")
}

fn receive<T>(channel: &mpsc::Receiver<T>) -> T {
    channel
        .recv_timeout(TIMEOUT)
        .expect("DirectStorage event timed out")
}

fn next_event(
    events: &mpsc::Receiver<TestEvent>,
    mut wanted: impl FnMut(&TestEvent) -> bool,
) -> TestEvent {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let event = events
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("expected DirectStorage event did not arrive");
        if wanted(&event) {
            return event;
        }
    }
}

fn event_hook() -> (TestHook, mpsc::Receiver<TestEvent>) {
    let (tx, rx) = mpsc::channel();
    (
        Arc::new(move |event| {
            let _ = tx.send(event);
        }),
        rx,
    )
}

fn gated_hook(milestone: TestEvent) -> (TestHook, mpsc::Receiver<TestEvent>, mpsc::Sender<()>) {
    let (tx, rx) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let blocked = Mutex::new(blocked);
    let hook = Arc::new(move |event| {
        let _ = tx.send(event);
        if event == milestone {
            let _ = blocked.lock().unwrap().recv_timeout(TIMEOUT);
        }
    });
    (hook, rx, release)
}

fn await_later_preparation(events: &mpsc::Receiver<TestEvent>) {
    loop {
        match receive(events) {
            TestEvent::Prepared(2) => return,
            TestEvent::BeforeRetire(_) | TestEvent::Retired(_) => {
                panic!("a request retired while the FIFO head was still blocked");
            }
            _ => {}
        }
    }
}

#[test]
fn parallel_out_of_order_completion_retires_fifo() {
    with_deadline(|| {
        let first = vec![0x15; 48 * 1024];
        let second = vec![0x67; 48 * 1024];
        let file = TestFile::compressed(&[&first, &second]);
        let (mut out_first, mut out_second) = (vec![0; first.len()], vec![0; second.len()]);
        let pool = test_pool(2, TEST_BUDGET);
        let (hook, events, release) = gated_hook(TestEvent::BeforeDecode(1));
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        let status = TestStatus::new(1);
        unsafe {
            queue.enqueue(&file.request(0, out_first.as_mut_ptr() as usize, 7));
            queue.enqueue(&file.request(1, out_second.as_mut_ptr() as usize, 8));
        }
        queue.status(&status, 0);
        queue.submit();

        await_later_preparation(&events);
        assert_eq!(status.result(0), E_PENDING);
        release.send(()).unwrap();
        for seq in 1..=2 {
            assert_eq!(
                next_event(&events, |e| matches!(e, TestEvent::Retired(_))),
                TestEvent::Retired(seq)
            );
        }
        assert_eq!(status.wait(0), S_OK);
        assert_eq!(out_first, first);
        assert_eq!(out_second, second);
        let metrics = pool.metrics();
        assert_eq!((metrics.jobs_started, metrics.jobs_completed), (2, 2));
        assert_eq!(metrics.current_staging_bytes, 0);
    });
}

#[test]
fn submitted_batch_retires_without_notifications_or_followup_submits() {
    with_deadline(|| {
        let payload = vec![0x81; 64 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let mut output = vec![0; payload.len()];
        let pool = test_pool(2, TEST_BUDGET);
        let (hook, events) = event_hook();
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        unsafe { queue.enqueue(&file.request(0, output.as_mut_ptr() as usize, 9)) };
        queue.submit();

        assert_eq!(
            next_event(&events, |e| matches!(e, TestEvent::Retired(_))),
            TestEvent::Retired(1)
        );
        assert_eq!(output, payload);
        assert_eq!(pool.metrics().jobs_completed, 1);
    });
}

#[test]
fn budget_retires_ready_error_and_payload_without_worker_notification() {
    with_deadline(|| {
        let cap = 96 * 1024;
        let payload = vec![0x2a; cap];
        let file = TestFile::compressed(&[&payload]);
        let mut output = vec![0; cap];
        let pool = test_pool(2, cap);
        let (hook, events) = event_hook();
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        let mut pipeline = Pipeline::new(queue.inner(), pool.clone());
        let request = Request::parse(&file.request(0, output.as_mut_ptr() as usize, 1));
        let reservation = pool.shared.budget.try_reserve(cap).unwrap();
        let mut staged = StagingBuffer::new(reservation).unwrap();
        staged.bytes.copy_from_slice(&payload);
        pipeline.in_flight.extend([
            InFlightTask {
                seq: 0,
                req: request,
                file: None,
            },
            InFlightTask {
                seq: 1,
                req: request,
                file: None,
            },
        ]);
        pipeline.completed.insert(0, Err(E_FILE_HANDLE));
        pipeline.completed.insert(1, Ok(staged));
        assert_eq!(pipeline.outstanding_workers, 0);
        assert_eq!(pool.metrics().current_staging_bytes, cap);

        let mut first_error = S_OK;
        let next_reservation = pipeline.reserve(cap, &mut first_error).unwrap();
        assert_eq!(first_error, E_FILE_HANDLE);
        assert!(pipeline.in_flight.is_empty());
        assert!(pipeline.completed.is_empty());
        assert_eq!(pool.metrics().current_staging_bytes, 0);
        for seq in 0..2 {
            assert_eq!(
                next_event(&events, |e| matches!(e, TestEvent::Retired(_))),
                TestEvent::Retired(seq)
            );
        }
        assert_eq!(output, payload);
        drop(next_reservation);
        assert_eq!(pool.metrics().peak_staging_bytes, cap);
    });
}

#[test]
fn tight_budget_makes_progress_after_immediate_file_failure() {
    with_deadline(|| {
        let payload = vec![0x37; 96 * 1024];
        let broken = TestFile::compressed(&[&payload]);
        let available = TestFile::compressed(&[&payload]);
        let (mut discarded, mut output) = (vec![0; payload.len()], vec![0; payload.len()]);
        let pool = test_pool(2, 144 * 1024);
        let (hook, events) = event_hook();
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        let status = TestStatus::new(1);
        unsafe {
            file_close(broken.raw);
            queue.enqueue(&broken.request(0, discarded.as_mut_ptr() as usize, 22));
            queue.enqueue(&available.request(0, output.as_mut_ptr() as usize, 23));
        }
        queue.status(&status, 0);
        queue.submit();
        assert_eq!(status.wait(0), E_FILE_HANDLE);
        assert_eq!(
            next_event(&events, |e| matches!(e, TestEvent::BeforeDecode(_))),
            TestEvent::BeforeDecode(2)
        );
        assert!(discarded.iter().all(|b| *b == 0));
        assert_eq!(output, payload);
        assert_eq!(pool.metrics().jobs_started, 1);
    });
}

#[test]
fn cancellation_of_started_or_prepared_work_keeps_existing_results() {
    with_deadline(|| {
        let payload = vec![0x52; 48 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let (mut first, mut second, mut later) = (
            vec![0; payload.len()],
            vec![0; payload.len()],
            vec![0; payload.len()],
        );
        let pool = test_pool(2, TEST_BUDGET);
        let (hook, events, release) = gated_hook(TestEvent::BeforeDecode(1));
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        let status = TestStatus::new(2);
        let tag = 0x42;
        unsafe {
            queue.enqueue(&file.request(0, first.as_mut_ptr() as usize, tag));
            queue.enqueue(&file.request(0, second.as_mut_ptr() as usize, tag));
        }
        queue.status(&status, 0);
        queue.submit();
        await_later_preparation(&events);
        queue.cancel(tag);
        assert_eq!(status.result(0), E_PENDING);
        unsafe { queue.enqueue(&file.request(0, later.as_mut_ptr() as usize, tag)) };
        queue.status(&status, 1);
        queue.submit();

        release.send(()).unwrap();
        assert_eq!(status.wait(0), S_OK);
        assert_eq!(status.wait(1), S_OK);
        assert_eq!(first, payload);
        assert_eq!(second, payload);
        assert_eq!(later, payload);
        assert_eq!(pool.metrics().jobs_completed, 3);
    });
}

#[test]
fn staging_stays_charged_until_the_other_queue_finishes_retirement() {
    with_deadline(|| {
        let payload = vec![0x91; 96 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let (mut first, mut second) = (vec![0; payload.len()], vec![0; payload.len()]);
        let pool = test_pool(2, 144 * 1024);
        let (hook1, events1, release) = gated_hook(TestEvent::BeforeRetire(1));
        let (hook2, events2) = event_hook();
        let q1 = TestQueue::new(Some(pool.clone()), Some(hook1));
        let q2 = TestQueue::new(Some(pool.clone()), Some(hook2));
        let status = TestStatus::new(2);
        unsafe { q1.enqueue(&file.request(0, first.as_mut_ptr() as usize, 1)) };
        q1.status(&status, 0);
        q1.submit();
        next_event(&events1, |e| *e == TestEvent::BeforeRetire(1));
        assert_eq!(pool.metrics().current_staging_bytes, payload.len());
        unsafe { q2.enqueue(&file.request(0, second.as_mut_ptr() as usize, 2)) };
        q2.status(&status, 1);
        q2.submit();
        assert_eq!(receive(&events2), TestEvent::BudgetWait);
        assert_eq!((status.result(0), status.result(1)), (E_PENDING, E_PENDING));

        release.send(()).unwrap();
        assert_eq!((status.wait(0), status.wait(1)), (S_OK, S_OK));
        assert_eq!(first, payload);
        assert_eq!(second, payload);
        let stats = pool.metrics();
        assert_eq!(stats.current_staging_bytes, 0);
        assert!(stats.peak_staging_bytes <= stats.budget_bytes);
        assert!(stats.budget_waits > 0);
    });
}

#[test]
fn failed_worker_task_reports_error_and_preserves_the_next_result() {
    with_deadline(|| {
        let payload = vec![0x46; 48 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let (mut discarded, mut output) = (vec![0; payload.len()], vec![0; payload.len()]);
        let pool = test_pool(1, TEST_BUDGET);
        let hook = Arc::new(|event| {
            if event == TestEvent::BeforeDecode(1) {
                panic!("test decoder task failure");
            }
        });
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        let status = TestStatus::new(1);
        unsafe {
            queue.enqueue(&file.request(0, discarded.as_mut_ptr() as usize, 1));
            queue.enqueue(&file.request(0, output.as_mut_ptr() as usize, 2));
        }
        queue.status(&status, 0);
        queue.submit();
        assert_eq!(status.wait(0), E_ABORT);
        assert!(discarded.iter().all(|b| *b == 0));
        assert_eq!(output, payload);
        assert_eq!(pool.metrics().current_staging_bytes, 0);
    });
}

#[test]
fn parallel_decode_reports_corruption_and_zero_pads_the_next_request() {
    with_deadline(|| {
        let payload = vec![0x2b; 48 * 1024];
        let file = TestFile::new([
            (vec![0; 64], payload.len()),
            (
                miniz_oxide::deflate::compress_to_vec_zlib(&payload, 6),
                payload.len(),
            ),
        ]);
        let mut failed = vec![0x1d; payload.len()];
        let mut padded = vec![0x1d; payload.len() + 4096];
        let pool = test_pool(2, TEST_BUDGET);
        let queue = TestQueue::new(Some(pool.clone()), None);
        let status = TestStatus::new(1);
        let mut request = file.request(1, padded.as_mut_ptr() as usize, 2);
        request[0x18..0x1c].copy_from_slice(&(padded.len() as u32).to_le_bytes());
        request[0x34..0x38].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        unsafe {
            queue.enqueue(&file.request(0, failed.as_mut_ptr() as usize, 1));
            queue.enqueue(&request);
        }
        queue.status(&status, 0);
        queue.submit();
        assert_eq!(status.wait(0), E_ZLIB_DATA);
        assert!(failed.iter().all(|byte| *byte == 0x1d));
        assert_eq!(&padded[..payload.len()], payload);
        assert!(padded[payload.len()..].iter().all(|byte| *byte == 0));
        assert_eq!(pool.metrics().jobs_completed, 2);
    });
}

#[test]
fn request_larger_than_staging_budget_uses_sequential_fallback() {
    with_deadline(|| {
        let payload = vec![0x54; 48 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let mut output = vec![0; payload.len()];
        let pool = test_pool(2, 1024);
        let queue = TestQueue::new(Some(pool.clone()), None);
        let status = TestStatus::new(1);
        unsafe { queue.enqueue(&file.request(0, output.as_mut_ptr() as usize, 1)) };
        queue.status(&status, 0);
        queue.submit();
        assert_eq!(status.wait(0), S_OK);
        assert_eq!(output, payload);
        assert_eq!(pool.metrics().jobs_started, 0);
        assert_eq!(pool.metrics().peak_staging_bytes, 0);
    });
}

#[test]
fn queue_close_waits_for_started_preparation() {
    with_deadline(|| {
        let payload = vec![0x3a; 48 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let mut output = vec![0; payload.len()];
        let pool = test_pool(2, TEST_BUDGET);
        let (hook, events, release) = gated_hook(TestEvent::BeforeDecode(1));
        let queue = TestQueue::new(Some(pool.clone()), Some(hook));
        unsafe { queue.enqueue(&file.request(0, output.as_mut_ptr() as usize, 13)) };
        queue.submit();
        assert_eq!(receive(&events), TestEvent::BeforeDecode(1));

        let raw_queue = queue.0 as usize;
        thread::scope(|scope| {
            let (send_closed, closed) = mpsc::channel();
            let close_worker = scope.spawn(move || {
                unsafe { queue_close(raw_queue as *mut c_void) };
                send_closed.send(()).unwrap();
            });
            wait_until(|| queue.inner().closed.load(Ordering::Acquire));
            assert!(matches!(closed.try_recv(), Err(mpsc::TryRecvError::Empty)));
            release.send(()).unwrap();
            receive(&closed);
            close_worker.join().unwrap();
        });

        assert!(events
            .try_iter()
            .all(|e| !matches!(e, TestEvent::Retired(_))));
        assert!(output.iter().all(|b| *b == 0));
        assert_eq!(pool.metrics().current_staging_bytes, 0);
    });
}

#[test]
fn a_cancelled_refusal_after_parallel_work_keeps_status_successful() {
    with_deadline(|| {
        let payload = vec![0x36; 64 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let mut output = vec![0; payload.len()];
        let mut refused_output = vec![0xCD; payload.len()];
        let status = TestStatus::new(1);
        let (hook, events, release) = gated_hook(TestEvent::BeforeDecode(1));
        let queue = TestQueue::new(Some(test_pool(2, TEST_BUDGET)), Some(hook));
        let mut refused = file.request(0, refused_output.as_mut_ptr() as usize, 2);
        // BC2 is refused by the upstream decoder.
        refused[..8].copy_from_slice(&(2_u64 << BCPACK_SHIFT).to_le_bytes());
        unsafe {
            queue.enqueue(&file.request(0, output.as_mut_ptr() as usize, 1));
            queue.enqueue(&refused);
        }
        queue.status(&status, 0);
        queue.submit();
        assert_eq!(receive(&events), TestEvent::BeforeDecode(1));
        queue.cancel(2);
        release.send(()).unwrap();
        assert_eq!(status.wait(0), S_OK);
        assert_eq!(queue.inner().errors.lock().unwrap().0, 0);
        assert_eq!(output, payload);
        assert!(refused_output.iter().all(|&byte| byte == 0xCD));
    });
}

#[test]
fn a_destination_decommitted_during_decode_fails_in_fifo_order() {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
    };

    with_deadline(|| {
        let payload = vec![0x51; 64 * 1024];
        let file = TestFile::compressed(&[&payload]);
        let destination = unsafe {
            VirtualAlloc(std::ptr::null(), payload.len(), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        };
        assert!(!destination.is_null());
        let mut next = vec![0; payload.len()];
        let status = TestStatus::new(2);
        let (hook, events, release) = gated_hook(TestEvent::BeforeDecode(1));
        let queue = TestQueue::new(Some(test_pool(2, TEST_BUDGET)), Some(hook));
        unsafe { queue.enqueue(&file.request(0, destination as usize, 1)) };
        queue.status(&status, 0);
        unsafe { queue.enqueue(&file.request(0, next.as_mut_ptr() as usize, 2)) };
        queue.status(&status, 1);
        queue.submit();
        assert_eq!(receive(&events), TestEvent::BeforeDecode(1));
        let decommitted = unsafe { VirtualFree(destination, payload.len(), MEM_DECOMMIT) };
        release.send(()).unwrap();
        assert_ne!(decommitted, 0);
        assert_eq!(status.wait(0), E_ACCESS_VIOLATION);
        assert_eq!(status.wait(1), S_OK);
        assert_eq!(next, payload);
        assert_ne!(unsafe { VirtualFree(destination, 0, MEM_RELEASE) }, 0);
    });
}
