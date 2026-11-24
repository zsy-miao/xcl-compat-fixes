use super::*;
use std::sync::atomic::AtomicUsize;
use std::sync::{mpsc, Barrier};
use std::time::Duration;

static NEXT_STANDIN: AtomicUsize = AtomicUsize::new(0xF8ED_0000_0000);

struct Standin(usize);

impl Standin {
    fn new() -> Self {
        Self(NEXT_STANDIN.fetch_add(16, Ordering::Relaxed))
    }
}

impl Drop for Standin {
    fn drop(&mut self) {
        release(released(self.0));
    }
}

#[test]
fn lazy_pipeline_waits_for_its_first_bind() {
    let key = Standin::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let job_calls = calls.clone();
    let (finished, completion) = mpsc::channel();
    submit_lazy(key.0, false, Box::new(move || {
        job_calls.fetch_add(1, Ordering::SeqCst);
        finished.send(()).unwrap();
        0
    }), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "creation must not run the build");
    assert!(is_pending(key.0));
    assert!(matches!(completion.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert_eq!(resolve_for_bind(key.0), key.0);
    completion.recv_timeout(Duration::from_secs(5)).expect("first bind must schedule the lazy build");
    assert_eq!(resolve_wait(key.0), key.0, "a failed build retains its stand-in");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!is_pending(key.0));
}

#[test]
fn concurrent_waiting_binds_build_a_lazy_pipeline_once() {
    let key = Standin::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let job_calls = calls.clone();
    submit_lazy(key.0, false, Box::new(move || {
        job_calls.fetch_add(1, Ordering::SeqCst);
        0
    }), 0);
    const CALLERS: usize = 8;
    let ready = Arc::new(Barrier::new(CALLERS + 1));
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for _ in 0..CALLERS {
            let ready = ready.clone();
            let standin = key.0;
            joins.push(scope.spawn(move || {
                ready.wait();
                resolve_wait(standin)
            }));
        }
        ready.wait();
        for join in joins {
            assert_eq!(join.join().unwrap(), key.0);
        }
    });
    assert_eq!(calls.load(Ordering::SeqCst), 1, "concurrent binds must share one build");
    assert!(!is_pending(key.0));
}

#[test]
fn releasing_an_unused_lazy_pipeline_drops_its_job_without_building() {
    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let key = Standin::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let captured = Dropped(drops.clone());
    let job_calls = calls.clone();
    submit_lazy(key.0, false, Box::new(move || {
        job_calls.fetch_add(1, Ordering::SeqCst);
        drop(captured);
        0
    }), 0);
    assert_eq!(released(key.0), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1, "releasing before use must release captured build data");
    assert!(!is_pending(key.0));
    assert_eq!(resolve_wait(key.0), key.0, "a released entry must not be resurrected");
}

#[test]
fn eager_submission_still_builds_without_a_bind() {
    let key = Standin::new();
    let (finished, completion) = mpsc::channel();
    submit(key.0, true, Box::new(move || {
        finished.send(std::thread::current().name().unwrap_or("").to_owned()).unwrap();
        0
    }));
    let thread = completion.recv_timeout(Duration::from_secs(5)).expect("eager submission must run without a bind");
    assert!(thread.starts_with("xcl-pso-"), "eager build should use a worker: {thread}");
    assert_eq!(resolve_wait(key.0), key.0);
    assert!(!is_pending(key.0));
}
