//! Opt-in snapshots before a title's watchdog: absence of presents is not proof of a GPU hang.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

fn timeout() -> Option<Duration> {
    static VALUE: OnceLock<Option<Duration>> = OnceLock::new();
    *VALUE.get_or_init(|| parse_timeout(std::env::var("XCL_STALL_SECONDS").ok().as_deref()))
}

fn parse_timeout(value: Option<&str>) -> Option<Duration> {
    value?.trim().parse::<u64>().ok().filter(|&n| (1..=120).contains(&n)).map(Duration::from_secs)
}

#[derive(Default)]
struct Detector {
    frames: u64,
    progress: Duration,
    reported: Duration,
    reports: u8,
}

impl Detector {
    fn sample(&mut self, now: Duration, frames: u64, limit: Duration) -> bool {
        if frames != self.frames {
            self.frames = frames;
            self.progress = now;
            self.reported = now;
            self.reports = 0;
        }
        if frames == 0 || self.reports == 3 || now.saturating_sub(self.reported) < limit {
            return false;
        }
        self.reported = now;
        self.reports += 1;
        true
    }
}

#[derive(Clone)]
struct Submission {
    stage: &'static str,
    since: Instant,
    lists: Option<Vec<(usize, usize)>>,
    returned: bool,
}

// (queue, submitting thread) -> its latest stage and last recorded PSOs, never guest pointers
// to dereference. Bound the number of records and lists, even for a title creating many queues.
static SUBMISSIONS: Mutex<Option<HashMap<(usize, u32), Submission>>> = Mutex::new(None);

#[derive(Default)]
pub(super) struct Submit {
    key: Option<(usize, u32)>,
    previous: Option<Submission>,
}

pub(super) fn enter(queue: usize) -> Submit {
    if timeout().is_none() { return Submit::default(); }
    let key = (queue, unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() });
    enter_key(key)
}

fn enter_key(key: (usize, u32)) -> Submit {
    if let Ok(mut g) = SUBMISSIONS.try_lock() {
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() < 32 || m.contains_key(&key) {
            let previous = m.insert(key, Submission { stage: "texture refresh", since: Instant::now(), lists: None, returned: false }).filter(|s| !s.returned);
            return Submit { key: Some(key), previous };
        }
    }
    Submit::default()
}

impl Submit {
    fn update(&self, f: impl FnOnce(&mut Submission)) {
        if let Some(key) = self.key {
            if let Ok(mut g) = SUBMISSIONS.try_lock() {
                if let Some(s) = g.as_mut().and_then(|m| m.get_mut(&key)) {
                    f(s);
                }
            }
        }
    }

    pub(super) fn stage(&self, stage: &'static str) {
        self.update(|s| { s.stage = stage; s.since = Instant::now(); });
    }

    pub(super) fn lists(&self, lists: &[*mut std::ffi::c_void]) {
        if self.key.is_some() {
            let recorded = super::bind::diagnostic_pipelines(lists);
            self.update(|s| s.lists = recorded);
        }
    }
}

impl Drop for Submit {
    fn drop(&mut self) {
        let previous = self.previous.take();
        self.update(|s| match previous {
            Some(previous) => *s = previous,
            None => { s.stage = "returned from submit"; s.since = Instant::now(); s.returned = true; }
        });
    }
}

pub(crate) fn start() {
    let Some(limit) = timeout() else { return };
    let _ = std::thread::Builder::new().name("xcl-stall".into()).spawn(move || {
        let start = Instant::now();
        let mut detector = Detector::default();
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let frames = super::present::counters().0;
            let now = start.elapsed();
            if detector.sample(now, frames, limit) {
                snapshot(now.saturating_sub(detector.progress), frames);
            }
        }
    });
}

fn snapshot(idle: Duration, frames: u64) {
    crate::xlog_warn!("stall snapshot: no shown frame for {} ms ({frames} shown); this alone does not establish a GPU hang", idle.as_millis());
    let records = SUBMISSIONS.try_lock().ok().map(|g| g.as_ref().map(|m| m.iter().map(|(&k, v)| (k, v.clone())).collect::<Vec<_>>()).unwrap_or_default());
    match records {
        Some(records) => for ((q, thread), s) in records {
            crate::xlog_warn!("stall submit: queue {q:#x}, thread {thread}, stage '{}', age {} ms, last recorded (list, PSO) {:x?}", s.stage, s.since.elapsed().as_millis(), s.lists);
        },
        None => crate::xlog_warn!("stall submit: tracker busy; unavailable"),
    }
    match super::alloc_pool::diagnostic_progress() {
        Some(queues) => for (q, submitted, completed) in queues {
            crate::xlog_warn!("stall fence: queue {q:#x}, submitted {submitted}, completed {completed}, removed {}", completed == u64::MAX);
        },
        None => crate::xlog_warn!("stall fence: allocator tracker busy; unavailable"),
    }
    match super::pso_async::diagnostic_builds() {
        Some((builds, busy)) => {
            crate::xlog_warn!("stall build: {} active entries sampled (limit 16), {busy} busy entries unavailable", builds.len());
            for (pso, key, compute) in builds {
                crate::xlog_warn!("stall build: PSO {pso:#x}, shader key {key:016x}, compute {compute}; job taken and not complete");
            }
        },
        None => crate::xlog_warn!("stall build: pipeline tracker busy; unavailable"),
    }
    crate::log::flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_submissions_restore_the_outer_stage_and_busy_tracking_is_skipped() {
        let key = (0x7e57_1200, 0);
        let outer = enter_key(key);
        outer.stage("outer");
        {
            let inner = enter_key(key);
            inner.stage("inner");
        }
        {
            let g = SUBMISSIONS.lock().unwrap();
            assert_eq!(g.as_ref().unwrap()[&key].stage, "outer");
            let missed = enter_key((key.0 + 1, 0));
            assert!(missed.key.is_none(), "a busy tracker must not block a submit");
        }
        drop(outer);
        let mut g = SUBMISSIONS.lock().unwrap();
        assert!(g.as_ref().unwrap()[&key].returned);
        g.as_mut().unwrap().remove(&key);
    }

    #[test]
    fn snapshots_wait_for_the_first_frame_and_are_bounded_until_progress() {
        let mut d = Detector::default();
        let seconds = Duration::from_secs;
        assert!(!d.sample(seconds(100), 0, seconds(8)));
        assert!(!d.sample(seconds(101), 1, seconds(8)));
        assert!(!d.sample(seconds(108), 1, seconds(8)));
        assert!(d.sample(seconds(109), 1, seconds(8)));
        assert!(!d.sample(seconds(110), 1, seconds(8)));
        assert!(d.sample(seconds(117), 1, seconds(8)));
        assert!(d.sample(seconds(125), 1, seconds(8)));
        assert!(!d.sample(seconds(200), 1, seconds(8)));
        assert!(!d.sample(seconds(201), 2, seconds(8)));
        assert!(d.sample(seconds(209), 2, seconds(8)));
        assert_eq!(d.progress, seconds(201));
    }

    #[test]
    fn stall_timeout_is_opt_in_and_bounded() {
        for v in [None, Some("0"), Some("-1"), Some("121"), Some("bad")] {
            assert_eq!(parse_timeout(v), None);
        }
        assert_eq!(parse_timeout(Some(" 8 ")), Some(Duration::from_secs(8)));
    }
}
