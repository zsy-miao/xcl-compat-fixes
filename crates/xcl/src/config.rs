//! Per-title configuration (`%LOCALAPPDATA%\XCL\<title>\xcl.toml`) and the boot record
//! the launcher hands to the host process.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// "anaconda" (Series X) or "lockhart" (Series S).
    pub device_type: String,
    /// Raw value returned by kernelx `GetConsoleType`. Series values are not public;
    /// this is a guess and can be overridden.
    pub console_type_raw: u32,
    pub gamertag: String,
    pub user_id: u64,
    pub console_id: String,
    pub sandbox_id: String,
    /// Title memory budget reported by TitleMemoryStatus (MiB); 0 = the standard budget of
    /// `device_type` (13 824 Series X, 8 192 Series S).
    pub title_memory_mb: u64,
    /// "auto" (MicrosoftGame.Config), "standard" or "advanced" title memory mode.
    pub title_memory: String,
    /// Persistent local storage quota (MiB); 0 = MicrosoftGame.Config, else 4096.
    pub pls_mb: u64,
    /// Unified (GPU-visible) memory (MiB, up to 4 × 4095: above 4095 it is split into 4 GiB
    /// windows, see `gpu::uma`); 0 = 4095 for "advanced" title memory,
    /// else 2048. `XCL_UMA_MB` overrides.
    pub uma_mb: u64,
    pub log_level: String,
    pub log_to_stderr: bool,
    /// Executable inside the game folder (auto-detected when empty).
    pub exe: String,
    pub package_identifier: String,
    /// Xbox title id (hex) for XGameGetXboxTitleId; empty = `<TitleId>` of MicrosoftGame.Config.
    pub title_id: String,
    /// Console path prefix → target. `{game}` and `{data}` are expanded.
    pub mounts: BTreeMap<String, String>,
    /// Import DLL pattern → route: an HLE module ("kernelx.dll"), "@system" or "@stub".
    pub aliases: BTreeMap<String, String>,
    /// "dll!function" (or "*!function") → return value for auto-stubs.
    pub stub_returns: BTreeMap<String, String>,
    /// "d3d12" (real GPU, default) or "null" (no rendering).
    pub graphics: String,
    /// "auto" (high-performance GPU, then WARP), "warp", a number (that hardware adapter in
    /// high-performance order) or a part of the adapter's name ("radeon"). `XCL_ADAPTER` overrides.
    pub d3d12_adapter: String,
    /// Enable the D3D12 debug layer (needs Graphics Tools).
    pub d3d12_debug: bool,
    /// Build console compute pipelines at their first dispatch instead of while loading.
    /// Opt-in per title; `XCL_LAZY_COMPUTE=1` / `0` overrides this setting.
    pub lazy_compute_pipelines: bool,
    /// "auto" (show the title's frames in an XCL window) or "off".
    pub present: String,
    /// Raytracing tier reported by `CheckFeatureSupport(D3D12_OPTIONS5)`:
    /// "auto" (host GPU tier), "tier1_1" (11), "tier1_0" (10), or "off" (0).
    /// `XCL_RAYTRACING` overrides.
    pub raytracing: String,
    pub window_width: u32,
    pub window_height: u32,
    /// Answer `XUserGetTokenAndSignature*` with XCL's local placeholder token (for a backend of
    /// your own that accepts it). Off: the request fails as on a console that cannot reach Xbox
    /// Live (0x80072EFD), and titles go their offline way – Xbox Live's and the titles' own
    /// services would reject the placeholder anyway (Minecraft retried its sign-in forever).
    pub xbox_live_tokens: bool,
    /// Keyboard and mouse play as the gamepad while the XCL window has the focus: WASD the
    /// left stick, the mouse the right stick (the cursor is held in the window), the mouse
    /// buttons the triggers, the wheel the bumpers (see `hle::kbm`).
    pub keyboard_mouse: bool,
    /// How far the right stick goes for a mouse speed: 1.0 = full deflection at 4000 mouse
    /// counts a second.
    pub mouse_sensitivity: f32,
}

impl Config {
    /// Fill the "auto" settings from the game's MicrosoftGame.Config (if any); afterwards every
    /// setting has a concrete value. Returns what came from the manifest (for the log).
    pub fn resolve(&mut self, manifest: Option<&crate::manifest::Manifest>) -> Vec<String> {
        let mut from = Vec::new();
        let series_s = self.is_series_s();
        if self.title_memory == "auto" {
            let m = manifest.and_then(|m| if series_s { m.lockhart_memory.clone() } else { m.anaconda_memory.clone() });
            if let Some(m) = &m {
                from.push(format!("title memory {m}"));
            }
            self.title_memory = m.unwrap_or_else(|| "standard".into()).to_ascii_lowercase();
        }
        // the standard budgets; "advanced" adds memory the console docs do not give publicly –
        // the standard figure stays (xcl.toml can raise it)
        self.title_memory_mb = self.title_memory_budget_mb();
        if self.pls_mb == 0 {
            self.pls_mb = match manifest.and_then(|m| m.pls_mb) {
                Some(v) => {
                    from.push(format!("PLS {v} MB"));
                    v
                }
                None => 4096,
            };
        }
        if self.title_id.is_empty() {
            if let Some(t) = manifest.and_then(|m| m.title_id.clone()) {
                from.push(format!("TitleId {t}"));
                self.title_id = t;
            }
        }
        if self.uma_mb == 0 {
            self.uma_mb = if self.title_memory == "advanced" { 4095 } else { 2048 };
        }
        from
    }

    /// TitleMemoryStatus budget (MiB): `title_memory_mb`, 0 = the device's standard budget.
    pub fn title_memory_budget_mb(&self) -> u64 {
        match self.title_memory_mb {
            0 if self.is_series_s() => 8_192,
            0 => 13_824,
            v => v,
        }
    }
}

/// Mount target of the persistent local storage drive (`XPersistentLocalStorageGetPath`).
pub const PLS_MOUNT_TARGET: &str = "{data}\\pls";

impl Config {
    /// The console drive the title is given as its persistent local storage root ("S:\"):
    /// the drive-letter mount onto `{data}\pls`. None when the mounts have no such drive.
    pub fn pls_drive(&self) -> Option<String> {
        self.mounts.iter().find_map(|(prefix, target)| {
            let b = prefix.as_bytes();
            let target = target.replace('/', "\\");
            let is_drive = b.len() == 2 && b[0].is_ascii_alphabetic() && b[1] == b':';
            (is_drive && target.trim_end_matches('\\').eq_ignore_ascii_case(PLS_MOUNT_TARGET)).then(|| format!("{}\\", prefix.to_ascii_uppercase()))
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        let mut mounts = BTreeMap::new();
        mounts.insert("GAME:".into(), "{game}".into());
        mounts.insert("G:".into(), "{game}".into());
        // the package drive of the XDK days, still used by GDK titles (XDefiant: d:\hunter\...)
        mounts.insert("D:".into(), "{game}".into());
        mounts.insert("T:".into(), "{data}\\temp".into());
        // persistent local storage: on the console a folder mounted at the root of a drive
        mounts.insert("S:".into(), PLS_MOUNT_TARGET.into());
        Config {
            device_type: "anaconda".into(),
            console_type_raw: 6,
            gamertag: "Player1".into(),
            user_id: 0x0009_0000_0000_0001,
            console_id: CONSOLE_ID.into(),
            sandbox_id: "RETAIL".into(),
            title_memory_mb: 0,
            title_memory: "auto".into(),
            pls_mb: 0,
            uma_mb: 0,
            log_level: "info".into(),
            log_to_stderr: true,
            exe: String::new(),
            package_identifier: String::new(),
            title_id: String::new(),
            mounts,
            aliases: BTreeMap::new(),
            stub_returns: BTreeMap::new(),
            graphics: "d3d12".into(),
            d3d12_adapter: "auto".into(),
            d3d12_debug: false,
            lazy_compute_pipelines: false,
            present: "auto".into(),
            raytracing: "auto".into(),
            window_width: 1280,
            window_height: 720,
            xbox_live_tokens: false,
            keyboard_mouse: true,
            mouse_sensitivity: 1.0,
        }
    }
}

const HEADER: &str = "# XCL – title configuration. Created automatically; you can edit it.\n\
# device_type: \"anaconda\" (Series X) or \"lockhart\" (Series S)\n\
# mounts: console path prefix -> folder ({game} = game folder, read-only; {data} = XCL data)\n\
# aliases: DLL pattern -> \"kernelx.dll\" / \"xgameruntime.dll\" / \"d3d12_xs.dll\" / \"@system\" / \"@stub\"\n\
# stub_returns: \"dll!Function\" -> value the stub returns (e.g. \"E_NOTIMPL\", \"0\", \"0x1\")\n\n";

/// What XSystemGetConsoleId returns on a PC and in RETAIL (the default sandbox): the console id
/// format, 38 characters, which titles size their buffer for (XSystemConsoleIdBytes).
pub const CONSOLE_ID: &str = "00000000.00000000.00000000.00000000.00";
/// The default before: 41 characters, too long for that buffer.
const OLD_CONSOLE_ID: &str = "XCL00000000000000000000000000000000000001";

impl Config {
    pub fn load_or_create(path: &Path) -> Result<Config, String> {
        match std::fs::read_to_string(path) {
            Ok(s) => {
                let mut c: Config = toml::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
                // default mounts added since the file was written
                for (k, v) in Config::default().mounts {
                    c.mounts.entry(k).or_insert(v);
                }
                // the old default console id, written into every xcl.toml created before
                if c.console_id == OLD_CONSOLE_ID {
                    c.console_id = CONSOLE_ID.into();
                }
                Ok(c)
            }
            // only a missing file gets the defaults written: an unreadable one (locked, not UTF-8)
            // must not be replaced by them (issue #106)
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("{}: {e}", path.display())),
            Err(_) => {
                let c = Config::default();
                let body = toml::to_string_pretty(&c).map_err(|e| e.to_string())?;
                if let Some(d) = path.parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                let _ = std::fs::write(path, format!("{HEADER}{body}"));
                Ok(c)
            }
        }
    }

    pub fn is_series_s(&self) -> bool {
        self.device_type.eq_ignore_ascii_case("lockhart") || self.device_type.eq_ignore_ascii_case("series s")
    }

    /// Configured `D3D12_RAYTRACING_TIER` override for `CheckFeatureSupport(D3D12_OPTIONS5)`:
    /// `None` keeps the host GPU tier ("auto"), `Some(0 / 10 / 11)` forces NOT_SUPPORTED /
    /// TIER_1_0 / TIER_1_1. `XCL_RAYTRACING` overrides `xcl.toml`.
    pub fn raytracing_tier_override(&self) -> Option<u32> {
        let env = std::env::var("XCL_RAYTRACING").ok();
        let mode = env.as_deref().unwrap_or(self.raytracing.as_str()).trim();
        parse_raytracing_tier(mode)
    }
}

pub fn parse_raytracing_tier(mode: &str) -> Option<u32> {
    if mode.eq_ignore_ascii_case("off") || mode.eq_ignore_ascii_case("none") || mode == "0" {
        Some(0)
    } else if mode.eq_ignore_ascii_case("tier1_0") || mode.eq_ignore_ascii_case("1.0") || mode == "10" {
        Some(10)
    } else if mode.eq_ignore_ascii_case("tier1_1") || mode.eq_ignore_ascii_case("1.1") || mode == "11" {
        Some(11)
    } else {
        None
    }
}

/// Everything the host process needs; written by the launcher as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boot {
    pub title: String,
    pub exe_path: PathBuf,
    pub game_root: PathBuf,
    pub data_dir: PathBuf,
    pub config: Config,
}

impl Boot {
    pub fn logs_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }
    pub fn cache_dir(&self) -> PathBuf {
        self.data_dir.join("cache")
    }
    pub fn report_path(&self) -> PathBuf {
        self.logs_dir().join("report.json")
    }
    pub fn expand(&self, s: &str) -> String {
        s.replace("{game}", &self.game_root.to_string_lossy())
            .replace("{data}", &self.data_dir.to_string_lossy())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roundtrips_through_toml() {
        let c = Config::default();
        let s = toml::to_string_pretty(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(back.mounts.get("G:").map(String::as_str), Some("{game}"));
        assert_eq!(back.title_memory_mb, 0);
    }

    #[test]
    fn partial_file_uses_defaults() {
        let c: Config = toml::from_str("device_type = \"lockhart\"\n").unwrap();
        assert!(c.is_series_s());
        assert_eq!(c.gamertag, "Player1");
        assert_eq!(c.raytracing, "auto");
        assert!(!c.lazy_compute_pipelines, "existing title configs keep eager compute creation");
    }

    #[test]
    fn compute_first_use_is_a_persistent_per_title_opt_in() {
        assert!(!Config::default().lazy_compute_pipelines);
        let c: Config = toml::from_str("lazy_compute_pipelines = true\n").unwrap();
        assert!(c.lazy_compute_pipelines);
        let saved = toml::to_string_pretty(&c).unwrap();
        let loaded: Config = toml::from_str(&saved).unwrap();
        assert!(loaded.lazy_compute_pipelines);
    }

    #[test]
    fn the_console_id_fits_the_console_id_buffer() {
        // XSystemConsoleIdBytes: the 38-character console id format plus its terminator
        assert_eq!(Config::default().console_id.len() + 1, 39);
        let dir = std::env::temp_dir().join(format!("xcl_cfg_console_id_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("xcl.toml");
        // an xcl.toml written with the old default gets the new one; a user's own id is kept
        std::fs::write(&path, format!("console_id = \"{OLD_CONSOLE_ID}\"\n")).unwrap();
        assert_eq!(Config::load_or_create(&path).unwrap().console_id, CONSOLE_ID);
        std::fs::write(&path, "console_id = \"12345678.12345678.12345678.12345678.12\"\n").unwrap();
        assert_eq!(Config::load_or_create(&path).unwrap().console_id, "12345678.12345678.12345678.12345678.12");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pls_drive_is_the_mount_onto_the_pls_folder() {
        let mut c = Config::default();
        assert_eq!(c.pls_drive().as_deref(), Some("S:\\"));
        // a user-moved drive (any spelling of the target) is followed
        c.mounts.remove("S:");
        assert_eq!(c.pls_drive(), None);
        c.mounts.insert("p:".into(), "{data}/PLS\\".into());
        assert_eq!(c.pls_drive().as_deref(), Some("P:\\"));
        // directory-style prefixes are not drives
        c.mounts.remove("p:");
        c.mounts.insert("PLS:".into(), PLS_MOUNT_TARGET.into());
        assert_eq!(c.pls_drive(), None);
    }

    #[test]
    fn every_series_s_spelling_gets_the_series_s_memory() {
        let manifest = crate::manifest::Manifest {
            anaconda_memory: Some("Standard".into()),
            lockhart_memory: Some("Advanced".into()),
            ..Default::default()
        };
        for device in ["lockhart", "Series S"] {
            let mut c = Config { device_type: device.into(), ..Config::default() };
            c.resolve(Some(&manifest));
            assert_eq!((c.title_memory.as_str(), c.title_memory_mb), ("advanced", 8_192), "{device}");
        }
        let mut c = Config::default();
        c.resolve(Some(&manifest));
        assert_eq!((c.title_memory.as_str(), c.title_memory_mb), ("standard", 13_824));
    }

    #[test]
    fn raytracing_tier_parsing() {
        assert_eq!(parse_raytracing_tier("auto"), None);
        assert_eq!(parse_raytracing_tier(""), None);
        assert_eq!(parse_raytracing_tier("off"), Some(0));
        assert_eq!(parse_raytracing_tier("none"), Some(0));
        assert_eq!(parse_raytracing_tier("0"), Some(0));
        assert_eq!(parse_raytracing_tier("tier1_0"), Some(10));
        assert_eq!(parse_raytracing_tier("1.0"), Some(10));
        assert_eq!(parse_raytracing_tier("10"), Some(10));
        assert_eq!(parse_raytracing_tier("tier1_1"), Some(11));
        assert_eq!(parse_raytracing_tier("1.1"), Some(11));
        assert_eq!(parse_raytracing_tier("11"), Some(11));
    }
}
