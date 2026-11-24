//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// sent to the workers (a lazy pipeline is sent at its first use)
    queued: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
}

impl Pending {
    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// Sample jobs already taken by a worker/bind. Never wait for a pipeline's mutex in a stall report.
pub(super) fn diagnostic_builds() -> Option<(Vec<(usize, u64, bool)>, usize)> {
    diagnostic_builds_in(&PENDING)
}

fn diagnostic_builds_in(pending: &Mutex<Option<HashMap<usize, Arc<Pending>>>>) -> Option<(Vec<(usize, u64, bool)>, usize)> {
    let g = pending.try_lock().ok()?;
    let mut busy = 0;
    let builds = g.as_ref().map(|m| m.iter().filter_map(|(&id, p)| {
        let Ok(done) = p.done.try_lock() else { busy += 1; return None };
        if *done { return None; }
        let Ok(job) = p.job.try_lock() else { busy += 1; return None };
        if job.is_some() { return None; }
        Some((id, p.key, p.compute))
    }).take(16).collect()).unwrap_or_default();
    Some((builds, busy))
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn stall_snapshot_distinguishes_queued_building_complete_and_busy_jobs() {
        let p = Arc::new(Pending {
            job: Mutex::new(Some(Box::new(|| 0))), queued: Default::default(), real: AtomicUsize::new(0),
            done: Mutex::new(false), cv: Condvar::new(), compute: true, key: 123,
            used: Default::default(), orphan: Default::default(), late: Default::default(),
        });
        let pending = Mutex::new(Some(HashMap::from([(42, p.clone())])));
        assert_eq!(diagnostic_builds_in(&pending), Some((vec![], 0)), "queued is not building");
        p.job.lock().unwrap().take();
        assert_eq!(diagnostic_builds_in(&pending), Some((vec![(42, 123, true)], 0)));
        {
            let _held = p.job.lock().unwrap();
            assert_eq!(diagnostic_builds_in(&pending), Some((vec![], 1)));
        }
        *p.done.lock().unwrap() = true;
        assert_eq!(diagnostic_builds_in(&pending), Some((vec![], 0)));
        let _held = pending.lock().unwrap();
        assert!(diagnostic_builds_in(&pending).is_none(), "busy map is not waited for");
    }
}

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

/// Stack of the threads that build pipelines: translation recurses deeply on big programs.
const BUILD_STACK: usize = 64 << 20;

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Arc<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        // DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(BUILD_STACK).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    complete(&p, job());
                }
            });
        }
        Mutex::new(tx)
    })
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    let p = pending(standin, compute, job, 0);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    send(p);
}

/// Pipelines built at their first use (`submit_lazy`), `XCL_LAZY_PSO=0` to build each when
/// the title creates it. A title can create far more pipelines than it draws with: the retail
/// GTA VI creates ~51 000 different ones in its first 30 s; each a pipeline in AMD's driver
/// (~400 KB of its heap) took 20 GB of commit and the system ran out.
pub fn lazy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// Compute first-use builds are opt-in: other titles retain eager compute creation.
/// The first bind waits for the real pipeline, so a dispatch cannot use its stand-in.
pub fn lazy_compute() -> bool {
    let configured = crate::state::config().lazy_compute_pipelines;
    let override_value = std::env::var("XCL_LAZY_COMPUTE").ok().and_then(|v| match v.trim() {
        "1" => Some(true),
        "0" => Some(false),
        value => value.parse::<bool>().ok(),
    });
    lazy() && override_value.unwrap_or(configured)
}

/// `submit`, sent to the workers only on its first use – or now, at a low priority,
/// when an earlier run used these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    let p = pending(standin, compute, job, key);
    if used::known(key) {
        if let Some(q) = preload(compute) {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
    let n = LAZY.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n % 10_000 == 0 {
        crate::xlog_info!("D3D12.X: pipelines are built at their first use ({n} created so far)");
    }
}

static LAZY: AtomicUsize = AtomicUsize::new(0);

fn pending(standin: usize, compute: bool, job: Job, key: u64) -> Arc<Pending> {
    let p = Arc::new(Pending {
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        job: Mutex::new(Some(job)),
        queued: Default::default(),
        late: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    p
}

/// To the workers, once.
fn send(p: Arc<Pending>) {
    if !p.queued.swap(true, Ordering::AcqRel) {
        let _ = queue().lock().unwrap().send(p);
    }
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off; `graphics`: graphics only). Apart from the ones a bind sends to
/// (`queue`) and at the lowest priority: a pipeline that is bound is not held up by the ones
/// made ahead (whoever takes `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

#[derive(Clone, Copy)]
enum PreloadMode {
    All,
    Graphics,
    Off,
}

fn preload(compute: bool) -> Option<&'static Preload> {
    static MODE: OnceLock<PreloadMode> = OnceLock::new();
    let mode = MODE.get_or_init(|| match std::env::var("XCL_PSO_PRELOAD").ok().as_deref().map(str::trim) {
        Some("0") => PreloadMode::Off,
        Some("graphics") => PreloadMode::Graphics,
        _ => PreloadMode::All,
    });
    if matches!(mode, PreloadMode::Off) || (compute && matches!(mode, PreloadMode::Graphics)) {
        return None;
    }
    static Q: OnceLock<Preload> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| Preload { list: Mutex::new(Default::default()), cv: Condvar::new() });
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        let real = job();
                        complete(&p, real);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// Each policy gets a fresh process: the production queue reads its environment once.
    /// Jobs return distinct sentinels, as in the other scheduling test. Only completed entries
    /// are removed, and their returned pointers are never passed to native COM Release.
    #[test]
    fn preload_policy_preserves_required_first_use() {
        const CHILD: &str = "XCL_TEST_PRELOAD_POLICY";
        if let Ok(expected) = std::env::var(CHILD) {
            let (graphics, compute) = match expected.as_str() {
                "all" => (true, true),
                "graphics" => (true, false),
                "off" => (false, false),
                other => panic!("unknown test policy {other}"),
            };
            assert_eq!(preload(false).is_some(), graphics);
            assert_eq!(preload(true).is_some(), compute);
            for (standin, key, is_compute, ahead) in [(0x7E57_7101, 0x5EED_0101, false, graphics), (0x7E57_7102, 0x5EED_0102, true, compute)] {
                let calls = Arc::new(AtomicUsize::new(0));
                let count = calls.clone();
                let real = standin + 0x100;
                used::earlier_run_used(key);
                submit_lazy(standin, is_compute, Box::new(move || { count.fetch_add(1, Ordering::SeqCst); real }), key);
                let p = entry(standin).unwrap();
                if ahead {
                    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), std::time::Duration::from_secs(10), |done| !*done).unwrap();
                    assert!(*done, "a remembered pipeline admitted by the policy builds before binding");
                    assert_eq!(calls.load(Ordering::SeqCst), 1);
                } else {
                    assert!(is_pending(standin));
                    assert!(p.job.lock().unwrap().is_some(), "the first bind still owns the build");
                    assert_eq!(calls.load(Ordering::SeqCst), 0);
                }
                let first = resolve_for_bind(standin);
                if is_compute || ahead {
                    assert_eq!(first, real, "compute binds and completed graphics binds select the real pipeline");
                } else {
                    assert!(first == standin || first == real, "a graphics bind may race its worker");
                }
                // A graphics bind schedules rather than waits. A compute bind completes here.
                let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), std::time::Duration::from_secs(10), |done| !*done).unwrap();
                assert!(*done, "first use completes even when preloading is disabled for this type");
                drop(done);
                assert_eq!(resolve_for_bind(standin), real);
                assert_eq!(calls.load(Ordering::SeqCst), 1, "never build the same job twice");
                assert_eq!(released(standin), real);
            }
            return;
        }
        for (value, expected) in [(None, "all"), (Some("1"), "all"), (Some(" 0 "), "off"), (Some(" graphics "), "graphics"), (Some("legacy-value"), "all")] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args(["--exact", "hle::d3d12x::pso_async::used_tests::preload_policy_preserves_required_first_use", "--nocapture"]);
            child.env(CHILD, expected);
            if let Some(value) = value { child.env("XCL_PSO_PRELOAD", value); } else { child.env_remove("XCL_PSO_PRELOAD"); }
            let output = child.output().unwrap();
            assert!(output.status.success(), "policy {value:?}: {}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        }
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// Record a first use before its urgent build: a run ending during that build must
    /// still teach the next run to build these shaders ahead of their first use.
    #[test]
    fn a_compute_first_bind_records_history_before_its_urgent_build() {
        let (standin, key) = (0x7E57_7010usize, 0x5EED_0010u64);
        let (observed, received) = std::sync::mpsc::channel();
        submit_lazy(standin, true, Box::new(move || {
            observed.send((
                std::thread::current().name().unwrap_or("").to_owned(),
                used::noted(key),
            )).unwrap();
            0
        }), key);
        assert!(!used::noted(key), "creation alone is not a use");
        assert_eq!(resolve_for_bind(standin), standin, "a failed build keeps its stand-in");
        let (worker, noted_before_build) = received.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert_eq!(worker, "xcl-pso-urgent", "the retained compute first-use path must build urgently");
        assert!(noted_before_build, "history must be recorded before waiting for the build");
        assert!(used::noted(key) && !is_pending(standin));
        assert_eq!(released(standin), 0);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy pipeline's first use: build it now (the stand-in draws nothing meanwhile)
            send(p);
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if let Some(budget) = budget {
        return resolve_within(&p, pso, budget);
    }
    // not started yet: build it now instead of waiting behind the whole queue (thousands of
    // pipelines while a title loads – GTA VI's watchdog killed the game while a title thread
    // waited here), on a thread with the translator's stack, its compiles first
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        let real = std::thread::scope(|s| {
            std::thread::Builder::new()
                .name("xcl-pso-urgent".into())
                .stack_size(BUILD_STACK)
                .spawn_scoped(s, || crate::gpu::recomp::dxc::urgent(job))
                .ok()
                .and_then(|t| t.join().ok())
                .unwrap_or(0)
        });
        complete(&p, real);
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    send(p.clone());
    if p.late.load(Ordering::Acquire) {
        return pso;
    }
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    if *done {
        return built(p);
    }
    drop(done);
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    pso
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
#[path = "pso_async_tests.rs"]
mod scheduling_tests;
