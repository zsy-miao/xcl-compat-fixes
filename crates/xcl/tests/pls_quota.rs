//! Exercise the native A/W hooks without loading a title or initializing graphics.

use std::fs;
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::time::{SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Foundation::{GetLastError, SetLastError, ERROR_PATH_NOT_FOUND};
use windows_sys::Win32::Globalization::GetOEMCP;
use windows_sys::Win32::Storage::FileSystem::{
    AreFileApisANSI, GetDiskFreeSpaceExA, GetDiskFreeSpaceExW, SetFileApisToANSI, SetFileApisToOEM,
};
use xcl::config::{Boot, Config};
use xcl::util::wide;

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn disk_free_space_ex_reports_pls_quota() {
    const MIB: u64 = 1 << 20;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl_pls_quota_{}_{nonce}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let root = TempDir(dir);
    let game = root.0.join("game");
    let data = root.0.join("data");
    let pls = data.join("pls");
    fs::create_dir_all(&game).unwrap();
    fs::create_dir_all(pls.join("世界")).unwrap();
    fs::create_dir(pls.join("café")).unwrap();
    fs::create_dir(data.join("pls-other")).unwrap();
    fs::write(pls.join("existing.bin"), [0; 123]).unwrap();
    fs::write(pls.join("世界").join("nested.bin"), [0; 456]).unwrap();
    fs::write(
        game.join("MicrosoftGame.Config"),
        "<Game><PersistentLocalStorage><SizeMB>1</SizeMB></PersistentLocalStorage></Game>",
    )
    .unwrap();

    let mut config = Config::default();
    config.resolve(xcl::manifest::load(&game).as_ref());
    assert_eq!(config.pls_mb, 1);
    config.mounts.insert("P:".into(), "{data}\\pls".into());
    config
        .mounts
        .insert("T:".into(), "{data}\\pls-other".into());
    let boot = Boot {
        title: "PLS test".into(),
        exe_path: game.join("unused.exe"),
        game_root: game,
        data_dir: data,
        config,
    };
    let mounts = boot
        .config
        .mounts
        .iter()
        .map(|(k, v)| (k.clone(), boot.expand(v)))
        .collect();
    xcl::vfs::init(xcl::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts));
    xcl::state::set_boot(boot);
    xcl::vfs_hooks::install().unwrap();

    let host = wide(&pls.to_string_lossy());
    let (mut available, mut total, mut free) = (0, 0, 0);
    unsafe {
        SetLastError(9123);
        let result = GetDiskFreeSpaceExW(host.as_ptr(), &mut available, &mut total, &mut free);
        let host_last_error = GetLastError();
        assert_ne!(result, 0);
        let host_total = total;
        assert!(
            available >= MIB && free >= MIB,
            "test requires 1 MiB of host space"
        );

        let world = wide(r"s:/世界");
        SetLastError(9123);
        let result = GetDiskFreeSpaceExW(world.as_ptr(), &mut available, &mut total, &mut free);
        let error = GetLastError();
        assert_ne!(result, 0);
        assert_eq!(error, host_last_error);
        assert_eq!((available, total, free), (MIB - 579, MIB, MIB - 579));

        let drive = b"S:\\\0";
        let (mut ansi_available, mut ansi_free) = (0, 0);
        assert_ne!(
            GetDiskFreeSpaceExA(
                drive.as_ptr(),
                &mut ansi_available,
                null_mut(),
                &mut ansi_free
            ),
            0
        );
        assert_eq!((ansi_available, ansi_free), (available, free));

        if GetOEMCP() == 437 {
            let was_ansi = AreFileApisANSI() != 0;
            SetFileApisToOEM();
            let result = GetDiskFreeSpaceExA(
                c"S:\\caf\x82".as_ptr().cast(),
                &mut ansi_available,
                null_mut(),
                null_mut(),
            );
            if was_ansi {
                SetFileApisToANSI();
            }
            assert_ne!(result, 0);
            assert_eq!(ansi_available, available);
        }

        let alias = wide(r"P:\");
        assert_ne!(
            GetDiskFreeSpaceExW(alias.as_ptr(), null_mut(), &mut total, null_mut()),
            0
        );
        assert_eq!(total, MIB);
        for path in [&host, &wide(r"T:\"), &wide(r"S:\..\pls-other")] {
            assert_ne!(
                GetDiskFreeSpaceExW(path.as_ptr(), null_mut(), &mut total, null_mut()),
                0
            );
            assert_eq!(total, host_total);
        }
        assert_ne!(
            GetDiskFreeSpaceExW(null(), null_mut(), null_mut(), null_mut()),
            0
        );

        let missing = wide(r"S:\missing");
        SetLastError(0);
        let result = GetDiskFreeSpaceExW(missing.as_ptr(), &mut available, &mut total, &mut free);
        let error = GetLastError();
        assert_eq!((result, error), (0, ERROR_PATH_NOT_FOUND));
        assert_eq!((available, total, free), (MIB - 579, host_total, MIB - 579));

        fs::File::create(pls.join("fills-quota.bin"))
            .unwrap()
            .set_len(MIB)
            .unwrap();
        assert_ne!(
            GetDiskFreeSpaceExA(drive.as_ptr(), &mut available, &mut total, &mut free),
            0
        );
        assert_eq!((available, total, free), (0, MIB, 0));
        assert_ne!(
            GetDiskFreeSpaceExA(drive.as_ptr(), null_mut(), null_mut(), null_mut()),
            0
        );
    }
}
