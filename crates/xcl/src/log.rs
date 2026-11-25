//! Process-wide logger. The file handle is opened once at startup (before any file
//! hooks are installed) so logging never re-enters the VFS.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl Level {
    pub fn parse(s: &str) -> Level {
        match s.to_ascii_lowercase().as_str() {
            "error" => Level::Error,
            "warn" | "warning" => Level::Warn,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => Level::Info,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN ",
            Level::Info => "INFO ",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }
}

static FILE: Mutex<Option<File>> = Mutex::new(None);
static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);
static ECHO: AtomicBool = AtomicBool::new(true);
static START: OnceLock<Instant> = OnceLock::new();

/// Seconds since the log started (the host's start) – the times log lines carry.
pub fn uptime() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub fn init(path: Option<&Path>, level: Level, echo_stderr: bool) {
    START.get_or_init(Instant::now);
    LEVEL.store(level as u8, Ordering::Relaxed);
    ECHO.store(echo_stderr, Ordering::Relaxed);
    if let Some(p) = path {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(f) = File::create(p) {
            if let Ok(mut g) = FILE.lock() {
                *g = Some(f);
            }
        }
    }
}

pub fn enabled(level: Level) -> bool {
    level as u8 <= LEVEL.load(Ordering::Relaxed)
}

pub fn write(level: Level, args: std::fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }
    let t = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let line = format!("[{t:9.3}][{tid:5}][{}] {args}\n", level.tag());
    if let Ok(mut g) = FILE.try_lock().or_else(|_| FILE.lock()) {
        if let Some(f) = g.as_mut() {
            let _ = f.write_all(line.as_bytes());
            if level <= Level::Warn {
                let _ = f.flush();
            }
        }
    }
    if ECHO.load(Ordering::Relaxed) && level <= Level::Info {
        let _ = std::io::stderr().write_all(format!("xcl: {args}\n").as_bytes());
    }
}

pub fn flush() {
    if let Ok(mut g) = FILE.lock() {
        if let Some(f) = g.as_mut() {
            let _ = f.flush();
        }
    }
}

/// A panic goes to the log as well as to stderr.
pub fn log_panics() {
    let print = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write(Level::Error, format_args!("thread '{}' {}", std::thread::current().name().unwrap_or(""), info.to_string().replace('\n', " ")));
        print(info);
    }));
}

#[macro_export]
macro_rules! xlog_error { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Error, format_args!($($a)*)) } }
#[macro_export]
macro_rules! xlog_warn { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Warn, format_args!($($a)*)) } }
#[macro_export]
macro_rules! xlog_info { ($($a:tt)*) => { $crate::log::write($crate::log::Level::Info, format_args!($($a)*)) } }
#[macro_export]
macro_rules! xlog_debug { ($($a:tt)*) => { if $crate::log::enabled($crate::log::Level::Debug) { $crate::log::write($crate::log::Level::Debug, format_args!($($a)*)) } } }
#[macro_export]
macro_rules! xlog_trace { ($($a:tt)*) => { if $crate::log::enabled($crate::log::Level::Trace) { $crate::log::write($crate::log::Level::Trace, format_args!($($a)*)) } } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_in_the_log() {
        let path = std::env::temp_dir().join(format!("xcl-log-panics-{}.log", std::process::id()));
        *FILE.lock().unwrap() = Some(File::create(&path).unwrap());
        log_panics();
        let died = std::thread::Builder::new().name("xcl-dies".into()).spawn(|| panic!("gone for the run")).unwrap().join();
        *FILE.lock().unwrap() = None;
        let log = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(died.is_err());
        assert!(log.contains("thread 'xcl-dies' panicked at ") && log.contains("gone for the run"), "the log: {log:?}");
    }
}
