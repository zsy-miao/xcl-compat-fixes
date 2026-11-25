//! Bounded preparation of file-backed zlib requests. The queue thread owns
//! file reads, cancellation checks, and retirement; decoders own private buffers.

use super::{note_request, read_exact_at, writable, Entry, FileInner, QueueInner, Request, E_ABORT, E_ACCESS_VIOLATION, E_FILE_HANDLE, E_ZLIB_DATA};
use crate::util::{HRESULT, S_OK};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use windows_sys::Win32::Foundation::{E_OUTOFMEMORY, HANDLE};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Decode a zlib stream into a caller-provided contiguous buffer with no output growth.
/// Miniz validates the zlib checksum and uses an inline DecompressorOxide as its codec state.
fn decompress_zlib_into(compressed: &[u8], output: &mut [u8]) -> Result<(), HRESULT> {
    match miniz_oxide::inflate::decompress_slice_iter_to_slice(output, std::iter::once(compressed), true, false) {
        Ok(written) if written == output.len() => Ok(()),
        _ => Err(E_ZLIB_DATA),
    }
}

const CODEC_STATE_BYTES: usize = std::mem::size_of::<miniz_oxide::inflate::core::DecompressorOxide>();

struct StagingBudget {
    max: usize,
    used: Mutex<usize>,
    capacity: Condvar,
    #[cfg(test)]
    metrics: TestCounters,
}

impl StagingBudget {
    fn try_reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        let mut used = self.used.lock().unwrap();
        if bytes > self.max - *used {
            return None;
        }
        *used += bytes;
        Some(Reservation { budget: Arc::clone(self), bytes })
    }

    fn wait_for_capacity(&self, bytes: usize, closed: &AtomicBool) {
        let used = self.used.lock().unwrap();
        drop(self.capacity.wait_while(used, |used| bytes > self.max - *used && !closed.load(Ordering::Acquire)).unwrap());
    }

    fn wake_waiters(&self) {
        let _used = self.used.lock().unwrap();
        self.capacity.notify_all();
    }
}

/// Reserved bytes travel with their owners through the decode queue and FIFO retirement.
struct Reservation {
    budget: Arc<StagingBudget>,
    bytes: usize,
}

impl Reservation {
    fn split(&mut self, bytes: usize) -> Self {
        self.bytes -= bytes;
        Self { budget: Arc::clone(&self.budget), bytes }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut used = self.budget.used.lock().unwrap();
        *used -= self.bytes;
        self.budget.capacity.notify_all();
    }
}

struct StagingBuffer {
    bytes: Box<[u8]>,
    // Drop the box before releasing its charge to other queues.
    _allocation: Reservation,
}

#[cfg(test)]
impl Drop for StagingBuffer {
    fn drop(&mut self) {
        self._allocation.budget.metrics.staged.fetch_sub(self.bytes.len(), Ordering::Relaxed);
    }
}

impl StagingBuffer {
    fn new(reservation: Reservation) -> Result<Self, HRESULT> {
        let len = reservation.bytes;
        let bytes = if len == 0 {
            Box::default()
        } else {
            let layout = std::alloc::Layout::array::<u8>(len).map_err(|_| E_OUTOFMEMORY)?;
            // SAFETY: global allocator and the u8 slice's box use the same nonempty
            // layout. alloc_zeroed initializes every byte, and ownership moves to Box.
            let raw = unsafe { std::alloc::alloc_zeroed(layout) };
            if raw.is_null() {
                return Err(E_OUTOFMEMORY);
            }
            unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw, len)) }
        };
        #[cfg(test)]
        {
            let metrics = &reservation.budget.metrics;
            let staged = metrics.staged.fetch_add(len, Ordering::Relaxed) + len;
            metrics.peak_staged.fetch_max(staged, Ordering::Relaxed);
        }
        Ok(Self { bytes, _allocation: reservation })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TestEvent {
    BeforeDecode(u64),
    Prepared(u64),
    BeforeRetire(u64),
    Retired(u64),
    BudgetWait,
}

#[cfg(test)]
pub(super) type TestHook = Arc<dyn Fn(TestEvent) + Send + Sync>;

struct DecompressTask {
    seq: u64,
    compressed: StagingBuffer,
    output: Reservation,
    codec_state: Reservation,
    target_size: usize,
    tx: mpsc::Sender<DecompressResult>,
    #[cfg(test)]
    hook: Option<TestHook>,
}

struct DecompressResult {
    seq: u64,
    result: Result<StagingBuffer, HRESULT>,
}

struct PoolQueue {
    tasks: VecDeque<DecompressTask>,
    shutdown: bool,
}

struct SharedPool {
    queue: Mutex<PoolQueue>,
    work_available: Condvar,
    budget: Arc<StagingBudget>,
}

pub(super) struct DecompressPool {
    shared: Arc<SharedPool>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl DecompressPool {
    fn new(workers: usize, budget_bytes: usize) -> Option<Arc<Self>> {
        if workers == 0 {
            return None;
        }
        let mut pool = Self {
            shared: Arc::new(SharedPool {
                queue: Mutex::new(PoolQueue { tasks: VecDeque::new(), shutdown: false }),
                work_available: Condvar::new(),
                budget: Arc::new(StagingBudget {
                    max: budget_bytes,
                    used: Mutex::new(0),
                    capacity: Condvar::new(),
                    #[cfg(test)]
                    metrics: TestCounters::default(),
                }),
            }),
            workers: Vec::with_capacity(workers),
        };
        for id in 0..workers {
            let shared = Arc::clone(&pool.shared);
            match std::thread::Builder::new().name(format!("xcl-ds-decomp-{id}")).spawn(move || Self::worker_loop(shared)) {
                Ok(worker) => pool.workers.push(worker),
                Err(err) => {
                    crate::xlog_warn!("DirectStorage: could not start decompression workers ({err}); using sequential requests");
                    return None;
                }
            }
        }
        Some(Arc::new(pool))
    }

    fn worker_loop(shared: Arc<SharedPool>) {
        loop {
            let task = {
                let queue = shared.queue.lock().unwrap();
                let mut queue = shared.work_available.wait_while(queue, |q| q.tasks.is_empty() && !q.shutdown).unwrap();
                let Some(task) = queue.tasks.pop_front() else { break };
                task
            };
            #[cfg(test)]
            let counters = &shared.budget.metrics;
            let DecompressTask { seq, compressed, output, codec_state, target_size, tx, #[cfg(test)] hook } = task;
            // A worker owns each reservation until the associated allocation goes away.
            // Convert a panicking task into a completion so its queue cannot wait forever.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                #[cfg(test)]
                if let Some(hook) = hook {
                    hook(TestEvent::BeforeDecode(seq));
                }
                let mut output = StagingBuffer::new(output)?;
                let _codec_state = codec_state;
                #[cfg(test)]
                counters.started.fetch_add(1, Ordering::Relaxed);
                let result = decompress_zlib_into(&compressed.bytes, &mut output.bytes[..target_size]);
                #[cfg(test)]
                counters.completed.fetch_add(1, Ordering::Relaxed);
                result?;
                Ok(output)
            }))
            .unwrap_or(Err(E_ABORT));
            let _ = tx.send(DecompressResult { seq, result });
        }
    }

    fn dispatch(&self, task: DecompressTask) {
        let mut queue = self.shared.queue.lock().unwrap();
        queue.tasks.push_back(task);
        self.shared.work_available.notify_one();
    }

    #[cfg(test)]
    fn metrics(&self) -> TestMetrics {
        let budget = &self.shared.budget;
        let counters = &budget.metrics;
        TestMetrics {
            budget_bytes: budget.max,
            current_staging_bytes: counters.staged.load(Ordering::Relaxed),
            peak_staging_bytes: counters.peak_staged.load(Ordering::Relaxed),
            jobs_started: counters.started.load(Ordering::Relaxed),
            jobs_completed: counters.completed.load(Ordering::Relaxed),
            budget_waits: counters.budget_waits.load(Ordering::Relaxed),
        }
    }
}

impl Drop for DecompressPool {
    fn drop(&mut self) {
        self.shared.queue.lock().unwrap().shutdown = true;
        self.shared.work_available.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// The process owns the global pool after its first use. A scoped owner, such as
// an isolated test pool, shuts down and joins its workers on the last drop.
static DECOMPRESS_POOL: OnceLock<Option<Arc<DecompressPool>>> = OnceLock::new();

pub(super) fn pool_for(queue: &QueueInner) -> Option<Arc<DecompressPool>> {
    if queue.source_memory {
        return None;
    }
    #[cfg(test)]
    if let Some(pool) = queue.pool_override.get() {
        return pool.clone();
    }
    DECOMPRESS_POOL.get_or_init(|| {
        if !std::env::var("XCL_DSTORAGE_PARALLEL_DECOMPRESS").is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")) {
            return None;
        }
        let workers = std::env::var("XCL_DSTORAGE_WORKERS").ok().and_then(|value| value.parse().ok()).unwrap_or(4).clamp(1, 16);
        let budget_mib = std::env::var("XCL_DSTORAGE_MEMORY_BUDGET_MIB").ok().and_then(|value| value.parse().ok()).unwrap_or(32).clamp(1, 1024);
        DecompressPool::new(workers, budget_mib * 1024 * 1024)
    }).clone()
}

pub(super) fn wake_waiters(_queue: &QueueInner) {
    #[cfg(test)]
    let pool = _queue.pool_override.get().or_else(|| DECOMPRESS_POOL.get());
    #[cfg(not(test))]
    let pool = DECOMPRESS_POOL.get();
    if let Some(Some(pool)) = pool {
        pool.shared.budget.wake_waiters();
    }
}

#[cfg(test)]
#[derive(Default)]
struct TestCounters {
    staged: AtomicUsize,
    peak_staged: AtomicUsize,
    started: AtomicUsize,
    completed: AtomicUsize,
    budget_waits: AtomicUsize,
}

#[cfg(test)]
struct TestMetrics {
    budget_bytes: usize,
    current_staging_bytes: usize,
    peak_staging_bytes: usize,
    jobs_started: usize,
    jobs_completed: usize,
    budget_waits: usize,
}

struct InFlightTask {
    seq: u64,
    req: Request,
    file: Option<Arc<FileInner>>,
}

struct Pipeline<'a> {
    queue: &'a QueueInner,
    pool: Arc<DecompressPool>,
    tx: mpsc::Sender<DecompressResult>,
    rx: mpsc::Receiver<DecompressResult>,
    in_flight: VecDeque<InFlightTask>,
    completed: HashMap<u64, Result<StagingBuffer, HRESULT>>,
    outstanding_workers: usize,
}

impl<'a> Pipeline<'a> {
    fn new(queue: &'a QueueInner, pool: Arc<DecompressPool>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self { queue, pool, tx, rx, in_flight: VecDeque::new(), completed: HashMap::new(), outstanding_workers: 0 }
    }

    fn charge(&self, req: &Request, stage1_len: usize) -> Option<usize> {
        let cost = (req.source_size as usize)
            .checked_add(stage1_len.max(req.destination_size as usize))?
            .checked_add(CODEC_STATE_BYTES)?;
        (cost <= self.pool.shared.budget.max).then_some(cost)
    }

    fn retire_ready(&mut self, since_status: &mut HRESULT) {
        while let Some(front) = self.in_flight.front() {
            let Some(result) = self.completed.remove(&front.seq) else { break };
            let task = self.in_flight.pop_front().unwrap();
            if !self.queue.closed.load(Ordering::Acquire) {
                self.retire_one(task, result, since_status);
            }
        }
    }

    fn accept(&mut self, result: DecompressResult, since_status: &mut HRESULT) {
        let seq = result.seq;
        self.outstanding_workers -= 1;
        self.completed.insert(seq, result.result);
        self.retire_ready(since_status);
        #[cfg(test)]
        notify_test(self.queue, TestEvent::Prepared(seq));
    }

    fn collect_ready(&mut self, since_status: &mut HRESULT) {
        self.retire_ready(since_status);
        while let Ok(result) = self.rx.try_recv() {
            self.accept(result, since_status);
        }
    }

    fn wait_for_result(&mut self, since_status: &mut HRESULT) {
        // All dispatched workers send a completion even if an individual decoder panics.
        // In particular, do not block on this channel for a queue containing only ready errors.
        if self.outstanding_workers != 0 {
            if let Ok(result) = self.rx.recv() {
                self.accept(result, since_status);
                return;
            }
            self.outstanding_workers = 0;
        }
        for task in &self.in_flight {
            self.completed.entry(task.seq).or_insert(Err(E_ABORT));
        }
        self.retire_ready(since_status);
    }

    fn reserve(&mut self, cost: usize, since_status: &mut HRESULT) -> Option<Reservation> {
        loop {
            self.collect_ready(since_status);
            if self.queue.closed.load(Ordering::Acquire) {
                return None;
            }
            let budget = &self.pool.shared.budget;
            if let Some(reservation) = budget.try_reserve(cost) {
                return Some(reservation);
            }
            #[cfg(test)]
            {
                budget.metrics.budget_waits.fetch_add(1, Ordering::Relaxed);
                notify_test(self.queue, TestEvent::BudgetWait);
            }
            if self.in_flight.is_empty() {
                // With no local reservations left, another queue owns the bytes.
                budget.wait_for_capacity(cost, &self.queue.closed);
            } else {
                self.wait_for_result(since_status);
            }
        }
    }

    fn drain(&mut self, since_status: &mut HRESULT) {
        if self.in_flight.is_empty() {
            return;
        }
        self.collect_ready(since_status);
        while !self.in_flight.is_empty() {
            self.wait_for_result(since_status);
        }
    }

    fn start(&mut self, task: InFlightTask, stage1_len: usize, cost: usize, since_status: &mut HRESULT) {
        let seq = task.seq;
        let req = task.req;
        if self.queue.is_cancelled(seq, req.tag) {
            return;
        }
        let Some(mut reservation) = self.reserve(cost, since_status) else { return };
        // A request waiting for budget has not started. Check once more before preparing it,
        // then never reconsider cancellation during decode or retirement.
        if self.queue.closed.load(Ordering::Acquire) || self.queue.is_cancelled(seq, req.tag) {
            return;
        }
        note_request(&req);
        let loaded = (|| {
            if !writable(req.destination as usize, req.destination_size as usize) {
                return Err(E_ACCESS_VIOLATION);
            }
            let handle = task.file.as_ref().and_then(|file| file.handle()).ok_or(E_FILE_HANDLE)?;
            let mut input = StagingBuffer::new(reservation.split(req.source_size as usize))?;
            read_exact_at(&handle, req.offset, &mut input.bytes)?;
            Ok(input)
        })();
        self.in_flight.push_back(task);
        match loaded {
            Ok(compressed) => {
                let codec_state = reservation.split(CODEC_STATE_BYTES);
                self.outstanding_workers += 1;
                self.pool.dispatch(DecompressTask {
                    seq,
                    compressed,
                    output: reservation,
                    codec_state,
                    target_size: stage1_len,
                    tx: self.tx.clone(),
                    #[cfg(test)]
                    hook: self.queue.test_hook.get().cloned(),
                });
            }
            Err(hr) => {
                self.completed.insert(seq, Err(hr));
                drop(reservation);
            }
        }
    }

    fn retire_one(&self, task: InFlightTask, result: Result<StagingBuffer, HRESULT>, since_status: &mut HRESULT) {
        #[cfg(test)]
        notify_test(self.queue, TestEvent::BeforeRetire(task.seq));
        let result = result.and_then(|payload| {
            let n = task.req.destination_size as usize;
            let at = task.req.destination as usize;
            // The title can unmap the destination while a decoder runs. The late
            // check resolves lazy pages, and the kernel copy handles a subsequent unmap.
            if writable(at, n) && (n == 0 || write_destination(at, &payload.bytes[..n])) {
                Ok(())
            } else {
                Err(E_ACCESS_VIOLATION)
            }
        });
        if let Err(hr) = result {
            self.queue.update_error(hr, task.req, task.file.as_deref(), since_status);
        }
        #[cfg(test)]
        notify_test(self.queue, TestEvent::Retired(task.seq));
    }
}

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn NtWriteVirtualMemory(process: HANDLE, at: *mut c_void, data: *const c_void, len: usize, done: *mut usize) -> i32;
}

fn write_destination(at: usize, bytes: &[u8]) -> bool {
    let mut done = 0;
    // Use the current-process handle so an access fault is returned as a status.
    let status = unsafe {
        NtWriteVirtualMemory(GetCurrentProcess(), at as *mut c_void, bytes.as_ptr().cast(), bytes.len(), &mut done)
    };
    status >= 0 && done == bytes.len()
}

#[cfg(test)]
fn notify_test(queue: &QueueInner, event: TestEvent) {
    if let Some(hook) = queue.test_hook.get() {
        hook(event);
    }
}

/// Only file-backed zlib with a virtual destination uses the pool. Drain before
/// handling any other entry or ending a batch, so retirement, statuses and fences
/// retain upstream FIFO semantics even if the producer never submits again.
pub(super) fn run(queue: &QueueInner, pool: Arc<DecompressPool>, rx: mpsc::Receiver<Vec<Entry>>) {
    let mut pipeline = Pipeline::new(queue, pool);
    let mut since_status = S_OK;
    for batch in rx {
        for entry in batch {
            if let Entry::Request { seq, req, ref file, .. } = entry {
                if req.zlib() && !req.memory_source() && !req.dest_physical_pages() && req.swizzle() == 0 && req.bcpack() == 0 {
                    let stage1_len = if req.intermediate_size == 0 { req.destination_size } else { req.intermediate_size } as usize;
                    if let Some(cost) = pipeline.charge(&req, stage1_len) {
                        if !queue.closed.load(Ordering::Acquire) {
                            pipeline.start(InFlightTask { seq, req, file: file.clone() }, stage1_len, cost, &mut since_status);
                        }
                        continue;
                    }
                }
            }
            pipeline.drain(&mut since_status);
            queue.run_entry(entry, &mut since_status);
        }
        pipeline.drain(&mut since_status);
    }
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod tests;
