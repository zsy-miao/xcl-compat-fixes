//! Opt-in access-failure metadata. Counts come only from arrays already read by DirectStorage.

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Stage {
    #[default]
    Execute,
    EnqueueValidation,
    DestinationSnapshot,
    SourceSnapshot,
    DestinationWritable,
    SourcePages,
    SourceReadable,
    DestinationPages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Origin {
    HalfFull,
    TitleSubmit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Counts {
    pub total: usize,
    pub zero: usize,
}

impl Counts {
    pub fn of(pages: &[usize]) -> Self {
        Self { total: pages.len(), zero: pages.iter().filter(|&&p| p == 0).count() }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Reread {
    #[default]
    NotNeeded,
    Read,
    Unreadable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Trace {
    pub stage: Stage,
    pub destination: Option<Counts>,
    pub source: Option<Counts>,
    pub used: Option<Counts>,
    pub reread: Reread,
}

impl Trace {
    pub fn new(destination: Option<&[usize]>, source: Option<&[usize]>) -> Self {
        Self { destination: destination.map(Counts::of), source: source.map(Counts::of), ..Default::default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Event {
    pub queue: u64,
    pub origin: Origin,
    pub trace: Trace,
    pub source_memory: bool,
    pub source_pages: bool,
    pub destination_pages: bool,
    pub bytes: u32,
    pub bcpack: u8,
    pub zlib: bool,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct PageCounts(Option<Counts>);
        impl fmt::Display for PageCounts {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Some(n) => write!(f, "{} total/{} zero", n.total, n.zero),
                    None => f.write_str("unavailable"),
                }
            }
        }
        write!(f, "queue={} submit={:?} stage={:?} source={} destination={} bytes={} bcpack={} zlib={} dst-snapshot=[{}] src-snapshot=[{}] dst-used=[{}] late-reread={:?}",
            self.queue, self.origin, self.trace.stage,
            if self.source_pages { "memory-pages" } else if self.source_memory { "memory" } else { "file" },
            if self.destination_pages { "pages" } else { "virtual" },
            self.bytes, self.bcpack, self.zlib, PageCounts(self.trace.destination),
            PageCounts(self.trace.source), PageCounts(self.trace.used), self.trace.reread)
    }
}

pub(super) struct Diagnostics {
    pub id: u64,
    logged: AtomicUsize,
    #[cfg(test)]
    pub events: std::sync::Mutex<Vec<Event>>,
}

impl Diagnostics {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self { id: NEXT.fetch_add(1, Ordering::Relaxed), logged: AtomicUsize::new(0), #[cfg(test)] events: Default::default() }
    }

    pub fn admit(&self, event: Event) -> bool {
        static ALL: AtomicUsize = AtomicUsize::new(0);
        // Diagnostic bursts stay bounded even when the title never retrieves its errors.
        if !claim(&self.logged, 8) || !claim(&ALL, 64) {
            return false;
        }
        #[cfg(test)]
        self.events.lock().unwrap().push(event);
        let _ = event;
        true
    }
}

fn claim(counter: &AtomicUsize, limit: usize) -> bool {
    let mut n = counter.load(Ordering::Relaxed);
    while n < limit {
        match counter.compare_exchange_weak(n, n + 1, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(now) => n = now,
        }
    }
    false
}

#[cfg(test)]
mod failure_diagnostics_tests {
    use super::*;

    #[test]
    fn diagnostic_limit_is_shared_and_never_exceeded() {
        let count = AtomicUsize::new(0);
        let admitted = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4).map(|_| s.spawn(|| (0..100).filter(|_| claim(&count, 7)).count())).collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum::<usize>()
        });
        assert_eq!(admitted, 7);
        assert_eq!(count.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn diagnostic_counts_do_not_format_page_numbers() {
        let pages = [0, 0x7fff_dead_beef, 0];
        let trace = Trace::new(Some(&pages), None);
        assert_eq!(trace.destination, Some(Counts { total: 3, zero: 2 }));
        let event = Event { queue: 1, origin: Origin::HalfFull, trace, source_memory: false, source_pages: false, destination_pages: true, bytes: 16, bcpack: 0, zlib: false };
        let text = event.to_string();
        assert!(text.contains("dst-snapshot=[3 total/2 zero]"));
        assert!(text.contains("src-snapshot=[unavailable]"));
        assert!(!text.contains("dead") && !text.contains(&pages[1].to_string()));
    }
}
