//! XGameRuntime – offline implementation of the GDK system APIs a title needs to boot.
//! Signatures: public GDK reference (docs.xbox.com).

#![allow(non_snake_case)]

pub mod accessibility;
pub mod api;
pub mod gamesave;
pub mod gdk_table;
pub mod networking;
pub mod taskqueue;
pub mod xasync;
pub mod xsign;

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::util::{write_cstr_out, E_INVALIDARG, E_NOT_FOUND, E_NOT_SUFFICIENT_BUFFER, E_NOT_SUPPORTED, HRESULT, S_OK};
use accessibility::*;
use gamesave::*;
use networking::*;
use taskqueue::*;
use api::{InitializeApiImpl, InitializeApiImplEx, InitializeApiImplEx2, QueryApiImpl, UninitializeApiImpl, XErrorReport};
use xasync::*;

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        // the GDK thunk protocol (api.rs) – what titles actually call
        InitializeApiImpl,
        InitializeApiImplEx,
        InitializeApiImplEx2,
        UninitializeApiImpl,
        QueryApiImpl,
        XErrorReport,
        // runtime
        XGameRuntimeInitialize,
        XGameRuntimeInitializeWithOptions,
        XGameRuntimeUninitialize,
        XGameRuntimeIsFeatureAvailable,
        // task queue
        XTaskQueueCreate,
        XTaskQueueCreateComposite,
        XTaskQueueGetPort,
        XTaskQueueDuplicateHandle,
        XTaskQueueCloseHandle,
        XTaskQueueSubmitCallback,
        XTaskQueueSubmitDelayedCallback,
        XTaskQueueDispatch,
        XTaskQueueIsEmpty,
        XTaskQueueTerminate,
        XTaskQueueGetCurrentProcessTaskQueue,
        XTaskQueueSetCurrentProcessTaskQueue,
        XTaskQueueRegisterWaiter,
        XTaskQueueUnregisterWaiter,
        XTaskQueueRegisterMonitor,
        XTaskQueueUnregisterMonitor,
        // async
        XAsyncBegin,
        XAsyncSchedule,
        XAsyncComplete,
        XAsyncGetStatus,
        XAsyncGetResultSize,
        XAsyncGetResult,
        XAsyncCancel,
        XAsyncRun,
        // threads
        XThreadSetTimeSensitive,
        XThreadVerifyNotTimeSensitive,
        XThreadAssertNotTimeSensitive,
        XThreadIsTimeSensitive,
        // networking
        XNetworkingQueryPreferredLocalUdpMultiplayerPort,
        XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync,
        XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult,
        XNetworkingRegisterPreferredLocalUdpMultiplayerPortChanged,
        XNetworkingUnregisterPreferredLocalUdpMultiplayerPortChanged,
        XNetworkingQuerySecurityInformationForUrlAsync,
        XNetworkingQuerySecurityInformationForUrlAsyncResultSize,
        XNetworkingQuerySecurityInformationForUrlAsyncResult,
        XNetworkingQuerySecurityInformationForUrlUtf16Async,
        XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize,
        XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult,
        XNetworkingVerifyServerCertificate,
        XNetworkingGetConnectivityHint,
        XNetworkingRegisterConnectivityHintChanged,
        XNetworkingUnregisterConnectivityHintChanged,
        // system
        XSystemGetDeviceType,
        XSystemGetConsoleId,
        XSystemGetAnalyticsInfo,
        XSystemGetXboxLiveSandboxId,
        // package
        XPackageIsPackagedProcess,
        XPackageGetCurrentProcessPackageIdentifier,
        XPackageGetIdentifier,
        XPackageMount,
        XPackageGetMountPathSize,
        XPackageGetMountPath,
        XPackageCloseMountHandle,
        XPackageEnumeratePackages,
        // storage
        XPersistentLocalStorageGetPathSize,
        XPersistentLocalStorageGetPath,
        XPersistentLocalStorageGetSpaceInfo,
        XGameSaveFilesGetFolderWithUiAsync,
        XGameSaveFilesGetFolderWithUiResult,
        // game save containers
        XGameSaveInitializeProvider,
        XGameSaveInitializeProviderAsync,
        XGameSaveInitializeProviderResult,
        XGameSaveCloseProvider,
        XGameSaveGetRemainingQuota,
        XGameSaveGetRemainingQuotaAsync,
        XGameSaveGetRemainingQuotaResult,
        XGameSaveDeleteContainer,
        XGameSaveDeleteContainerAsync,
        XGameSaveDeleteContainerResult,
        XGameSaveCreateContainer,
        XGameSaveCloseContainer,
        XGameSaveGetContainerInfo,
        XGameSaveEnumerateContainerInfo,
        XGameSaveEnumerateContainerInfoByName,
        XGameSaveEnumerateBlobInfo,
        XGameSaveEnumerateBlobInfoByName,
        XGameSaveReadBlobData,
        XGameSaveReadBlobDataAsync,
        XGameSaveReadBlobDataResult,
        XGameSaveCreateUpdate,
        XGameSaveCloseUpdate,
        XGameSaveSubmitBlobWrite,
        XGameSaveSubmitBlobDelete,
        XGameSaveSubmitUpdate,
        XGameSaveSubmitUpdateAsync,
        XGameSaveSubmitUpdateResult,
        // users
        XUserAddAsync,
        XUserAddResult,
        XUserDuplicateHandle,
        XUserCloseHandle,
        XUserCompare,
        XUserGetMaxUsers,
        XUserGetId,
        XUserGetLocalId,
        XUserFindUserByLocalId,
        XUserFindUserById,
        XUserGetIsGuest,
        XUserGetState,
        XUserGetGamertag,
        XUserGetAgeGroup,
        XUserRegisterForChangeEvent,
        XUserUnregisterForChangeEvent,
        XUserGetTokenAndSignatureAsync,
        XUserGetTokenAndSignatureResultSize,
        XUserGetTokenAndSignatureResult,
        XUserGetTokenAndSignatureUtf16Async,
        XUserGetTokenAndSignatureUtf16ResultSize,
        XUserGetTokenAndSignatureUtf16Result,
        XUserFindForDevice,
        XUserFindControllerForUserWithUiAsync,
        XUserFindControllerForUserWithUiResult,
        // accessibility
        XClosedCaptionGetProperties,
        XClosedCaptionSetEnabled,
        XHighContrastGetMode,
        // store (offline)
        XStoreCreateContext,
        XStoreCloseContextHandle,
    ]
}

// ---------------------------------------------------------------------------------------
// runtime
// ---------------------------------------------------------------------------------------

static INIT_COUNT: AtomicU32 = AtomicU32::new(0);

pub unsafe extern "system" fn XGameRuntimeInitialize() -> HRESULT {
    if INIT_COUNT.fetch_add(1, Ordering::AcqRel) == 0 {
        crate::xlog_info!("XGameRuntimeInitialize (XCL offline runtime)");
    }
    S_OK
}

pub unsafe extern "system" fn XGameRuntimeInitializeWithOptions(_options: *const c_void) -> HRESULT {
    XGameRuntimeInitialize()
}

pub unsafe extern "system" fn XGameRuntimeUninitialize() {
    INIT_COUNT.fetch_sub(1, Ordering::AcqRel);
}

pub unsafe extern "system" fn XGameRuntimeIsFeatureAvailable(feature: u32) -> bool {
    matches!(feature, 0 | 2..=13 | 15..=21)
}

// ---------------------------------------------------------------------------------------
// XThread
// ---------------------------------------------------------------------------------------

thread_local! {
    static TIME_SENSITIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub unsafe extern "system" fn XThreadSetTimeSensitive(v: bool) -> HRESULT {
    TIME_SENSITIVE.with(|c| c.set(v));
    S_OK
}

pub unsafe extern "system" fn XThreadIsTimeSensitive() -> bool {
    TIME_SENSITIVE.with(|c| c.get())
}

pub unsafe extern "system" fn XThreadVerifyNotTimeSensitive() -> HRESULT {
    if TIME_SENSITIVE.with(|c| c.get()) {
        crate::xlog_warn!("XThreadVerifyNotTimeSensitive called on a time-sensitive thread");
    }
    S_OK
}

pub unsafe extern "system" fn XThreadAssertNotTimeSensitive() {
    if TIME_SENSITIVE.with(|c| c.get()) {
        crate::xlog_warn!("XThreadAssertNotTimeSensitive called on a time-sensitive thread");
    }
}

// ---------------------------------------------------------------------------------------
// XSystem
// ---------------------------------------------------------------------------------------

pub const DEVICE_XBOX_SCARLETT_LOCKHART: u32 = 0x06;
pub const DEVICE_XBOX_SCARLETT_ANACONDA: u32 = 0x07;

pub unsafe extern "system" fn XSystemGetDeviceType() -> u32 {
    if crate::state::config().is_series_s() {
        DEVICE_XBOX_SCARLETT_LOCKHART
    } else {
        DEVICE_XBOX_SCARLETT_ANACONDA
    }
}

pub unsafe extern "system" fn XSystemGetConsoleId(size: usize, buf: *mut u8, used: *mut usize) -> HRESULT {
    write_cstr_out(&crate::state::config().console_id, buf, size, used)
}

pub unsafe extern "system" fn XSystemGetXboxLiveSandboxId(size: usize, buf: *mut u8, used: *mut usize) -> HRESULT {
    write_cstr_out(&crate::state::config().sandbox_id, buf, size, used)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XVersion {
    pub major: u16,
    pub minor: u16,
    pub build: u16,
    pub revision: u16,
}

#[repr(C)]
pub struct XSystemAnalyticsInfo {
    pub os_version: XVersion,
    pub hosting_os_version: XVersion,
    pub family: [u8; 64],
    pub form: [u8; 64],
}

fn fixed64(s: &str) -> [u8; 64] {
    let mut a = [0u8; 64];
    let b = s.as_bytes();
    a[..b.len().min(63)].copy_from_slice(&b[..b.len().min(63)]);
    a
}

/// What a console reports: the Xbox OS build, family "Windows.Xbox".
pub unsafe extern "system" fn XSystemGetAnalyticsInfo() -> XSystemAnalyticsInfo {
    let v = XVersion { major: 10, minor: 0, build: 25398, revision: 0 };
    let form = if crate::state::config().is_series_s() { "Xbox Series S" } else { "Xbox Series X" };
    XSystemAnalyticsInfo { os_version: v, hosting_os_version: v, family: fixed64("Windows.Xbox"), form: fixed64(form) }
}

// ---------------------------------------------------------------------------------------
// XPackage
// ---------------------------------------------------------------------------------------

pub fn package_identifier() -> String {
    let c = crate::state::config();
    if !c.package_identifier.is_empty() {
        return c.package_identifier.clone();
    }
    let title = crate::state::boot().map(|b| b.title.clone()).unwrap_or_else(|| "XCL.Title".into());
    format!("{title}_1.0.0.0_x64__xcl")
}

fn game_root() -> String {
    crate::state::boot()
        .map(|b| b.game_root.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".into())
}

pub unsafe extern "system" fn XPackageIsPackagedProcess() -> bool {
    true
}

pub unsafe extern "system" fn XPackageGetCurrentProcessPackageIdentifier(size: usize, buf: *mut u8) -> HRESULT {
    write_cstr_out(&package_identifier(), buf, size, std::ptr::null_mut())
}

/// `XPackageGetIdentifier(pathOrName, bufferSize, buffer)` — slot 27 on `XPackage` takes
/// three arguments after `this` (unlike 2-arg `XPackageGetCurrentProcessPackageIdentifier`).
pub unsafe extern "system" fn XPackageGetIdentifier(_path_or_name: *const u8, size: usize, buf: *mut u8) -> HRESULT {
    write_cstr_out(&package_identifier(), buf, size, std::ptr::null_mut())
}

struct Mount {
    path: String,
}

pub unsafe extern "system" fn XPackageMount(id: *const u8, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let id = crate::util::from_cstr_ptr(id).unwrap_or_default();
    if !id.is_empty() && !id.eq_ignore_ascii_case(&package_identifier()) {
        crate::xlog_warn!("XPackageMount({id}) – unknown package (DLC?)");
        return E_NOT_FOUND;
    }
    *out = Box::into_raw(Box::new(Mount { path: game_root() })) as *mut c_void;
    S_OK
}

pub unsafe extern "system" fn XPackageGetMountPathSize(m: *mut c_void, size: *mut usize) -> HRESULT {
    if m.is_null() || size.is_null() {
        return E_INVALIDARG;
    }
    let mount = &*(m as *const Mount);
    *size = mount.path.len() + 1;
    S_OK
}

pub unsafe extern "system" fn XPackageGetMountPath(m: *mut c_void, size: usize, buf: *mut u8) -> HRESULT {
    if m.is_null() {
        return E_INVALIDARG;
    }
    write_cstr_out(&(*(m as *const Mount)).path, buf, size, std::ptr::null_mut())
}

pub unsafe extern "system" fn XPackageCloseMountHandle(m: *mut c_void) {
    if !m.is_null() {
        drop(Box::from_raw(m as *mut Mount));
    }
}

pub unsafe extern "system" fn XPackageEnumeratePackages(_kind: u32, _scope: u32, _ctx: *mut c_void, _cb: usize) -> HRESULT {
    // No DLC/related packages offline.
    S_OK
}

// ---------------------------------------------------------------------------------------
// Persistent local storage + save files
// ---------------------------------------------------------------------------------------

fn data_subdir(name: &str) -> PathBuf {
    let p = crate::state::data_dir().join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Host folder behind the title's persistent local storage.
fn pls_dir() -> PathBuf {
    data_subdir("pls")
}

/// The PLS path the title is given. On the console PLS is a folder mounted at the root of a
/// drive, and titles rely on it: Minecraft keeps only the drive name and builds
/// "<drive>:/Users/<xuid>/games/com.mojang" from it. So the title gets the VFS drive that
/// mounts `{data}\pls` ("S:\"); without such a mount, the host folder itself.
fn pls_path() -> String {
    crate::state::config().pls_drive().unwrap_or_else(|| pls_dir().to_string_lossy().into_owned())
}

pub unsafe extern "system" fn XPersistentLocalStorageGetPathSize(size: *mut usize) -> HRESULT {
    if size.is_null() {
        return E_INVALIDARG;
    }
    *size = pls_path().len() + 1;
    S_OK
}

pub unsafe extern "system" fn XPersistentLocalStorageGetPath(size: usize, buf: *mut u8, used: *mut usize) -> HRESULT {
    write_cstr_out(&pls_path(), buf, size, used)
}

#[repr(C)]
pub struct XPersistentLocalStorageSpaceInfo {
    pub available_free_bytes: u64,
    pub total_free_bytes: u64,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

/// Bytes the title occupies under `root`: the sizes of regular files, recursively.
/// Entries whose metadata can't be read are skipped; the total saturates instead of
/// overflowing. Links and other reparse points are never followed and never counted —
/// a link could point outside `root`, and only in-tree bytes belong to the quota.
/// (`DirEntry::file_type` on Windows reports every name-surrogate reparse point —
/// symlinks, junctions and mount points — as `is_symlink`, without an extra syscall
/// and without dereferencing it.)
pub(crate) fn pls_used_bytes(root: &Path) -> u64 {
    let mut used = 0u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_symlink() {
                continue;
            } else if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                used = used.saturating_add(entry.metadata().map(|m| m.len()).unwrap_or(0));
            }
        }
    }
    used
}

/// `total_bytes` is the title's quota; both free figures are the smaller of the host's
/// and the quota still unused (`quota − used`, saturating at zero free once the title
/// sits over quota). `used_bytes` stays truthful even when it exceeds the quota.
fn pls_space_info(quota: u64, used: u64, avail: u64, free: u64) -> XPersistentLocalStorageSpaceInfo {
    let remaining = quota.saturating_sub(used);
    XPersistentLocalStorageSpaceInfo {
        available_free_bytes: avail.min(remaining),
        total_free_bytes: free.min(remaining),
        used_bytes: used,
        total_bytes: quota,
    }
}

pub unsafe extern "system" fn XPersistentLocalStorageGetSpaceInfo(info: *mut XPersistentLocalStorageSpaceInfo) -> HRESULT {
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    if info.is_null() {
        return E_INVALIDARG;
    }
    let pls = pls_dir().to_string_lossy().into_owned();
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    let ok = GetDiskFreeSpaceExW(crate::util::wide(&pls).as_ptr(), &mut avail, &mut total, &mut free);
    // the title's quota (MicrosoftGame.Config <PersistentLocalStorage><SizeMB>, else 4 GiB)
    let quota = crate::state::config().pls_quota_bytes();
    let used = pls_used_bytes(Path::new(&pls));
    if ok == 0 {
        // host figures unavailable – let the remaining quota bound free space alone
        avail = quota;
        free = quota;
    }
    let out = pls_space_info(quota, used, avail, free);
    // titles judge "storage low/full" from these figures: log each new answer
    static LAST: std::sync::Mutex<Option<(u64, u64, u64, u64)>> = std::sync::Mutex::new(None);
    let now = (out.available_free_bytes, out.total_free_bytes, out.used_bytes, out.total_bytes);
    if let Ok(mut last) = LAST.lock() {
        if *last != Some(now) {
            *last = Some(now);
            crate::xlog_info!(
                "XPersistentLocalStorageGetSpaceInfo: {} MiB available, {} MiB free, {} MiB used of {} MiB (host {})",
                now.0 >> 20,
                now.1 >> 20,
                now.2 >> 20,
                now.3 >> 20,
                if ok == 0 { "unavailable" } else { "ok" }
            );
        }
    }
    *info = out;
    S_OK
}

static SAVE_FILES_IDENTITY: u8 = 0;

pub unsafe extern "system" fn XGameSaveFilesGetFolderWithUiAsync(_user: *mut c_void, scid: *const u8, b: *mut XAsyncBlock) -> HRESULT {
    let scid = crate::util::from_cstr_ptr(scid).unwrap_or_else(|| "default".into());
    let safe: String = scid.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    begin_with(b, &SAVE_FILES_IDENTITY as *const u8 as usize, move || {
        let p = data_subdir("saves").join(if safe.is_empty() { "default".into() } else { safe });
        std::fs::create_dir_all(&p).map_err(|_| crate::util::E_FAIL)?;
        let mut v = p.to_string_lossy().into_owned().into_bytes();
        v.push(0);
        Ok(v)
    })
}

pub unsafe extern "system" fn XGameSaveFilesGetFolderWithUiResult(b: *mut XAsyncBlock, size: usize, buf: *mut u8) -> HRESULT {
    XAsyncGetResult(b, &SAVE_FILES_IDENTITY as *const u8 as *const c_void, size, buf as *mut c_void, std::ptr::null_mut())
}

// ---------------------------------------------------------------------------------------
// XUser – one signed-in offline user
// ---------------------------------------------------------------------------------------

#[repr(C)]
pub struct User {
    magic: u32,
    refs: AtomicU32,
}

static USER: User = User { magic: 0x5553_4552, refs: AtomicU32::new(1) };
const LOCAL_ID: u64 = 1;
static USER_ADD_IDENTITY: u8 = 0;

fn user_ptr() -> *mut c_void {
    &USER as *const User as *mut c_void
}

unsafe fn valid_user(h: *mut c_void) -> bool {
    h == user_ptr()
}

pub unsafe extern "system" fn XUserAddAsync(options: u32, b: *mut XAsyncBlock) -> HRESULT {
    crate::xlog_info!("XUserAddAsync(options={options:#x}) → local offline user '{}'", crate::state::config().gamertag);
    begin_with(b, &USER_ADD_IDENTITY as *const u8 as usize, || {
        USER.refs.fetch_add(1, Ordering::AcqRel);
        Ok((user_ptr() as usize).to_le_bytes().to_vec())
    })
}

pub unsafe extern "system" fn XUserAddResult(b: *mut XAsyncBlock, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let hr = XAsyncGetResult(b, &USER_ADD_IDENTITY as *const u8 as *const c_void, 8, out as *mut c_void, std::ptr::null_mut());
    if hr >= 0 {
        user_signed_in();
    }
    hr
}

pub unsafe extern "system" fn XUserDuplicateHandle(h: *mut c_void, out: *mut *mut c_void) -> HRESULT {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    USER.refs.fetch_add(1, Ordering::AcqRel);
    *out = h;
    S_OK
}

pub unsafe extern "system" fn XUserCloseHandle(h: *mut c_void) {
    if valid_user(h) {
        let _ = USER.refs.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some(v.saturating_sub(1).max(1)));
    }
}

pub unsafe extern "system" fn XUserCompare(a: *mut c_void, b: *mut c_void) -> i32 {
    (a as isize - b as isize).signum() as i32
}

pub unsafe extern "system" fn XUserGetMaxUsers(out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = 1;
    S_OK
}

pub unsafe extern "system" fn XUserGetId(h: *mut c_void, out: *mut u64) -> HRESULT {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    *out = crate::state::config().user_id;
    S_OK
}

pub unsafe extern "system" fn XUserGetLocalId(h: *mut c_void, out: *mut u64) -> HRESULT {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    *out = LOCAL_ID;
    S_OK
}

pub unsafe extern "system" fn XUserFindUserByLocalId(id: u64, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    if id != LOCAL_ID {
        return E_NOT_FOUND;
    }
    USER.refs.fetch_add(1, Ordering::AcqRel);
    *out = user_ptr();
    S_OK
}

pub unsafe extern "system" fn XUserFindUserById(id: u64, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    if id != crate::state::config().user_id {
        return E_NOT_FOUND;
    }
    USER.refs.fetch_add(1, Ordering::AcqRel);
    *out = user_ptr();
    S_OK
}

pub unsafe extern "system" fn XUserGetIsGuest(h: *mut c_void, out: *mut bool) -> HRESULT {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    *out = false;
    S_OK
}

pub unsafe extern "system" fn XUserGetState(h: *mut c_void, out: *mut u32) -> HRESULT {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    *out = 0; // SignedIn
    S_OK
}

pub const GAMERTAG_SIZES: [usize; 4] = [16, 97, 15, 101]; // Classic, Modern, ModernSuffix, UniqueModern

/// A gamertag component of the configured gamertag, as XUserGamertagComponent describes it:
/// the user has no modern suffix, so ModernSuffix is empty and UniqueModern is the gamertag
/// without '#'. Each fits the component's buffer size (`GAMERTAG_SIZES`, terminator included),
/// which titles allocate: a longer configured gamertag is cut at a character boundary.
fn gamertag_component(tag: &str, component: u32) -> String {
    if component == 2 {
        return String::new(); // ModernSuffix: none
    }
    let max = match component {
        0 => 15, // Classic: 15 characters
        3 => 16, // UniqueModern: 16 rendered characters
        _ => GAMERTAG_SIZES[1] - 1,
    };
    let mut end = 0;
    for (i, c) in tag.char_indices() {
        if i + c.len_utf8() > GAMERTAG_SIZES[component as usize] - 1 || tag[..i].chars().count() >= max {
            break;
        }
        end = i + c.len_utf8();
    }
    tag[..end].to_string()
}

pub unsafe extern "system" fn XUserGetGamertag(h: *mut c_void, component: u32, size: usize, buf: *mut u8, used: *mut usize) -> HRESULT {
    if !valid_user(h) || component > 3 {
        return E_INVALIDARG;
    }
    let value = gamertag_component(&crate::state::config().gamertag, component);
    if size < value.len() + 1 {
        if !used.is_null() {
            *used = value.len() + 1;
        }
        return E_NOT_SUFFICIENT_BUFFER;
    }
    write_cstr_out(&value, buf, size, used)
}

pub unsafe extern "system" fn XUserGetAgeGroup(h: *mut c_void, out: *mut u32) -> HRESULT {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    *out = 3; // Adult
    S_OK
}

pub unsafe extern "system" fn XUserRegisterForChangeEvent(_q: *mut c_void, _ctx: *mut c_void, _cb: usize, token: *mut RegistrationToken) -> HRESULT {
    if token.is_null() {
        return E_INVALIDARG;
    }
    (*token).token = 1; // offline: the user never changes
    S_OK
}

pub unsafe extern "system" fn XUserUnregisterForChangeEvent(_token: RegistrationToken, _wait: bool) -> bool {
    true
}

#[repr(C)]
pub struct XUserGetTokenAndSignatureData {
    pub token_size: usize,
    pub signature_size: usize,
    pub token: *const u8,
    pub signature: *const u8,
}

#[repr(C)]
pub struct XUserGetTokenAndSignatureUtf16Data {
    pub token_count: usize,
    pub signature_count: usize,
    pub token: *const u16,
    pub signature: *const u16,
}

static TOKEN_UTF8_ID: u8 = 0;
static TOKEN_UTF16_ID: u8 = 0;
static FIND_CONTROLLER_ID: u8 = 0;

fn build_token_payload(utf16: bool, method: &str, url: &str, body: &[u8]) -> Result<PayloadBuf, HRESULT> {
    let tok_str = xsign::offline_token();
    let sig_str = xsign::sign_request(method, url, &tok_str, body)?;
    let (tok_bytes, sig_bytes, unit) = if utf16 {
        let tb: Vec<u8> = tok_str.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        let sb: Vec<u8> = sig_str.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        (tb, sb, 2usize)
    } else {
        let mut tb = tok_str.into_bytes();
        tb.push(0);
        let mut sb = sig_str.into_bytes();
        sb.push(0);
        (tb, sb, 1usize)
    };
    let header_len = 4 * std::mem::size_of::<usize>(); // 32 bytes on x64
    let sig_offset = header_len + tok_bytes.len();
    let mut bytes = Vec::with_capacity(sig_offset + sig_bytes.len());
    bytes.extend_from_slice(&(tok_bytes.len() / unit).to_le_bytes());
    bytes.extend_from_slice(&(sig_bytes.len() / unit).to_le_bytes());
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&sig_offset.to_le_bytes());
    bytes.extend_from_slice(&tok_bytes);
    bytes.extend_from_slice(&sig_bytes);
    Ok(PayloadBuf { bytes, fixups: vec![16, 24] })
}

unsafe fn copy_capped_body(body_size: usize, body: *const c_void) -> Vec<u8> {
    if body.is_null() || body_size == 0 {
        return Vec::new();
    }
    let len = body_size.min(8192);
    std::slice::from_raw_parts(body as *const u8, len).to_vec()
}

thread_local! {
    /// tests: `xbox_live_tokens` for requests made on this thread (None: the configuration's)
    pub(crate) static TOKENS_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// WININET_E_CANNOT_CONNECT: what a console that cannot reach Xbox Live gets for a token
const E_CANNOT_CONNECT: HRESULT = 0x8007_2EFD_u32 as i32;

/// The title asks for an Xbox Live token for a request (first 24 and every 64th logged).
/// Without `xbox_live_tokens` there is none: XCL has no Xbox Live.
fn note_token_request(method: &str, url: &str) -> Result<(), HRESULT> {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let tokens = TOKENS_HERE.with(|t| t.get()).unwrap_or_else(|| crate::state::config().xbox_live_tokens);
    if n < 24 || n % 64 == 0 {
        crate::xlog_info!("XUser: token #{n} for {method} {url} – {}", if tokens { "XCL's local placeholder" } else { "none: no Xbox Live (0x80072EFD)" });
    }
    if tokens {
        Ok(())
    } else {
        Err(E_CANNOT_CONNECT)
    }
}

pub unsafe extern "system" fn XUserGetTokenAndSignatureAsync(
    user: *mut c_void,
    _options: u32,
    method: *const u8,
    url: *const u8,
    _header_count: usize,
    _headers: *const c_void,
    body_size: usize,
    body: *const c_void,
    b: *mut XAsyncBlock,
) -> HRESULT {
    if !valid_user(user) || url.is_null() {
        return E_INVALIDARG;
    }
    let method = crate::util::from_cstr_ptr(method).filter(|s| !s.is_empty()).unwrap_or_else(|| "GET".into());
    let url = crate::util::from_cstr_ptr(url).unwrap_or_default();
    let allowed = note_token_request(&method, &url);
    let body = copy_capped_body(body_size, body);
    begin_with_fixups(b, &TOKEN_UTF8_ID as *const u8 as usize, move || {
        allowed?;
        build_token_payload(false, &method, &url, &body)
    })
}

pub unsafe extern "system" fn XUserGetTokenAndSignatureResultSize(b: *mut XAsyncBlock, size: *mut usize) -> HRESULT {
    XAsyncGetResultSize(b, size)
}

pub unsafe extern "system" fn XUserGetTokenAndSignatureResult(
    b: *mut XAsyncBlock,
    buffer_size: usize,
    buffer: *mut c_void,
    ptr_to_buffer: *mut *mut XUserGetTokenAndSignatureData,
    buffer_used: *mut usize,
) -> HRESULT {
    let hr = XAsyncGetResult(b, &TOKEN_UTF8_ID as *const u8 as *const c_void, buffer_size, buffer, buffer_used);
    if hr >= 0 && !ptr_to_buffer.is_null() {
        *ptr_to_buffer = buffer as *mut XUserGetTokenAndSignatureData;
    }
    hr
}

pub unsafe extern "system" fn XUserGetTokenAndSignatureUtf16Async(
    user: *mut c_void,
    _options: u32,
    method: *const u16,
    url: *const u16,
    _header_count: usize,
    _headers: *const c_void,
    body_size: usize,
    body: *const c_void,
    b: *mut XAsyncBlock,
) -> HRESULT {
    if !valid_user(user) || url.is_null() {
        return E_INVALIDARG;
    }
    let method = crate::util::from_wide_ptr(method).filter(|s| !s.is_empty()).unwrap_or_else(|| "GET".into());
    let url = crate::util::from_wide_ptr(url).unwrap_or_default();
    let allowed = note_token_request(&method, &url);
    let body = copy_capped_body(body_size, body);
    begin_with_fixups(b, &TOKEN_UTF16_ID as *const u8 as usize, move || {
        allowed?;
        build_token_payload(true, &method, &url, &body)
    })
}

pub unsafe extern "system" fn XUserGetTokenAndSignatureUtf16ResultSize(b: *mut XAsyncBlock, size: *mut usize) -> HRESULT {
    XAsyncGetResultSize(b, size)
}

pub unsafe extern "system" fn XUserGetTokenAndSignatureUtf16Result(
    b: *mut XAsyncBlock,
    buffer_size: usize,
    buffer: *mut c_void,
    ptr_to_buffer: *mut *mut XUserGetTokenAndSignatureUtf16Data,
    buffer_used: *mut usize,
) -> HRESULT {
    let hr = XAsyncGetResult(b, &TOKEN_UTF16_ID as *const u8 as *const c_void, buffer_size, buffer, buffer_used);
    if hr >= 0 && !ptr_to_buffer.is_null() {
        *ptr_to_buffer = buffer as *mut XUserGetTokenAndSignatureUtf16Data;
    }
    hr
}

/// XUserDeviceAssociationChange: the device and the users it was / is now associated with
/// (local ids; 0 is XUserNullUserLocalId)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAssociationChange {
    pub device_id: [u8; 32],
    pub old_user: u64,
    pub new_user: u64,
}

/// device association callbacks the title registered: (token, queue, context, callback)
static ASSOCIATION_CALLBACKS: std::sync::Mutex<Vec<(u64, usize, usize, usize)>> = std::sync::Mutex::new(Vec::new());
/// the local user was handed to the title (XUserAddResult): the input devices are associated
/// with it from then on
static USER_SIGNED_IN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The association a device gets when it becomes the signed-in user's: none → the user.
pub fn gamepad_association() -> DeviceAssociationChange {
    DeviceAssociationChange { device_id: crate::hle::gameinput::gamepad_device_id(), old_user: 0, new_user: LOCAL_ID }
}

struct AssociationCall {
    cb: usize,
    ctx: usize,
    change: DeviceAssociationChange,
}

unsafe extern "system" fn association_call(rec: *mut c_void, _canceled: bool) {
    let r = Box::from_raw(rec as *mut AssociationCall);
    let f: unsafe extern "system" fn(*mut c_void, *const DeviceAssociationChange) = std::mem::transmute(r.cb);
    f(r.ctx as *mut c_void, &r.change);
}

/// The title's association callback gets the gamepad → user association, on its queue's
/// completion port.
unsafe fn notify_association(queue: usize, ctx: usize, cb: usize) {
    let rec = Box::into_raw(Box::new(AssociationCall { cb, ctx, change: gamepad_association() }));
    let hr = xasync::submit_callback(queue as *mut c_void, PORT_COMPLETION, rec as *mut c_void, association_call as usize);
    if hr < 0 {
        drop(Box::from_raw(rec));
        crate::xlog_warn!("XUser: the device association callback could not be queued ({:#x})", hr as u32);
        return;
    }
    crate::xlog_info!("XUser: the gamepad is now associated with the local user (device association callback)");
}

/// XUserRegisterForDeviceAssociationChanged(queue, context, callback, &token). On a console a
/// device the title sees comes associated with the signed-in user it belongs to, and the
/// title hears of it here: Minecraft marks its controller "connected" only from this
/// callback (device arrival alone just attaches the device to the user's player), and its
/// Ore UI screens take gamepad input only from a connected controller.
pub unsafe extern "system" fn XUserRegisterForDeviceAssociationChanged(queue: *mut c_void, ctx: *mut c_void, cb: usize, token: *mut u64) -> HRESULT {
    if token.is_null() || cb == 0 {
        return E_INVALIDARG;
    }
    let t = NEXT_ASSOCIATION_TOKEN.fetch_add(1, Ordering::Relaxed);
    *token = t;
    ASSOCIATION_CALLBACKS.lock().unwrap().push((t, queue as usize, ctx as usize, cb));
    crate::xlog_info!("XUser: device association callback #{t:#x} registered (queue {queue:?}){}", if USER_SIGNED_IN.load(Ordering::Acquire) { "" } else { " – before the user is signed in" });
    if USER_SIGNED_IN.load(Ordering::Acquire) {
        notify_association(queue as usize, ctx as usize, cb);
    }
    S_OK
}

static NEXT_ASSOCIATION_TOKEN: AtomicU64 = AtomicU64::new(0x4153_0001);

pub unsafe extern "system" fn XUserUnregisterForDeviceAssociationChanged(token: u64, _wait: bool) -> bool {
    ASSOCIATION_CALLBACKS.lock().unwrap().retain(|c| c.0 != token);
    true
}

/// The local user reached the title: the gamepad becomes its device (each registered
/// association callback hears of it once).
fn user_signed_in() {
    if USER_SIGNED_IN.swap(true, Ordering::AcqRel) {
        return;
    }
    crate::xlog_info!("XUser: the local user reached the title (XUserAddResult); {} device association callback(s)", ASSOCIATION_CALLBACKS.lock().unwrap().len());
    let calls: Vec<_> = ASSOCIATION_CALLBACKS.lock().unwrap().clone();
    for (_, q, ctx, cb) in calls {
        unsafe { notify_association(q, ctx, cb) };
    }
}

/// Associate every input device with the local signed-in user.
pub unsafe extern "system" fn XUserFindForDevice(_device: *const c_void, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    USER.refs.fetch_add(1, Ordering::AcqRel);
    *out = user_ptr();
    S_OK
}

pub unsafe extern "system" fn XUserFindControllerForUserWithUiAsync(user: *mut c_void, b: *mut XAsyncBlock) -> HRESULT {
    if !valid_user(user) {
        return E_INVALIDARG;
    }
    begin_with(b, &FIND_CONTROLLER_ID as *const u8 as usize, || Ok(crate::hle::gameinput::gamepad_device_id().to_vec()))
}

pub unsafe extern "system" fn XUserFindControllerForUserWithUiResult(b: *mut XAsyncBlock, device_id: *mut [u8; 32]) -> HRESULT {
    if device_id.is_null() {
        return E_INVALIDARG;
    }
    XAsyncGetResult(b, &FIND_CONTROLLER_ID as *const u8 as *const c_void, 32, device_id as *mut c_void, std::ptr::null_mut())
}

// ---------------------------------------------------------------------------------------
// XStore – deliberately unavailable (no fake licences)
// ---------------------------------------------------------------------------------------

pub unsafe extern "system" fn XStoreCreateContext(_user: *mut c_void, out: *mut *mut c_void) -> HRESULT {
    crate::xlog_warn!("XStoreCreateContext → unavailable (offline mode, XCL does not fake licenses)");
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
    E_NOT_SUPPORTED
}

pub unsafe extern "system" fn XStoreCloseContextHandle(_h: *mut c_void) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_availability_covers_the_public_feature_enum() {
        let features = [
            (0, "XAccessibility"),
            (1, "XAppCapture"),
            (2, "XAsync"),
            (3, "XAsyncProvider"),
            (4, "XDisplay"),
            (5, "XGame"),
            (6, "XGameInvite"),
            (7, "XGameSave"),
            (8, "XGameUI"),
            (9, "XLauncher"),
            (10, "XNetworking"),
            (11, "XPackage"),
            (12, "XPersistentLocalStorage"),
            (13, "XSpeechSynthesizer"),
            (14, "XStore"),
            (15, "XSystem"),
            (16, "XTaskQueue"),
            (17, "XThread"),
            (18, "XUser"),
            (19, "XError"),
            (20, "XGameEvent"),
            (21, "XGameStreaming"),
        ];
        for (feature, name) in features {
            assert_eq!(
                unsafe { XGameRuntimeIsFeatureAvailable(feature) },
                feature != 1 && feature != 14,
                "unexpected availability for {name} ({feature})"
            );
        }
        assert!(!unsafe { XGameRuntimeIsFeatureAvailable(22) });
        assert!(!unsafe { XGameRuntimeIsFeatureAvailable(u32::MAX) });
    }

    #[test]
    fn gamertag_components_are_consistent_and_fit_their_buffers() {
        // no modern suffix: UniqueModern is the gamertag itself, not "Player1#0001"
        let parts: Vec<String> = (0..4).map(|c| gamertag_component("Player1", c)).collect();
        assert_eq!(parts, ["Player1", "Player1", "", "Player1"]);
        // a long configured gamertag still fits each documented buffer
        let long = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        assert_eq!(gamertag_component(long, 0), "ABCDEFGHIJKLMNO");
        assert_eq!(gamertag_component(long, 1), long);
        assert_eq!(gamertag_component(long, 3), "ABCDEFGHIJKLMNOP");
        // cut at a character boundary
        assert_eq!(gamertag_component("ÁÉÍÓÚÁÉÍÓÚÁÉÍÓÚÁÉ", 0), "ÁÉÍÓÚÁÉÍÓÚÁÉÍÓÚ".chars().take(7).collect::<String>());
        for c in 0..4 {
            assert!(gamertag_component(&long.repeat(10), c).len() < GAMERTAG_SIZES[c as usize]);
        }
    }

    #[test]
    fn user_add_and_queries() {
    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XUserAddAsync(1, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut u = std::ptr::null_mut();
            assert_eq!(XUserAddResult(&mut b, &mut u), S_OK);
            assert!(!u.is_null());
            let mut tag = [0u8; 16];
            let mut used = 0usize;
            assert_eq!(XUserGetGamertag(u, 0, 16, tag.as_mut_ptr(), &mut used), S_OK);
            assert_eq!(&tag[..used - 1], b"Player1");
            assert_eq!(XUserGetGamertag(u, 0, 3, tag.as_mut_ptr(), &mut used), E_NOT_SUFFICIENT_BUFFER);
            // no modern suffix: UniqueModern matches it
            let mut unique = [0u8; 101];
            assert_eq!(XUserGetGamertag(u, 3, 101, unique.as_mut_ptr(), &mut used), S_OK);
            assert_eq!(&unique[..used - 1], b"Player1");
            let mut id = 0u64;
            assert_eq!(XUserGetId(u, &mut id), S_OK);
            assert_ne!(id, 0);
            XUserCloseHandle(u);
            assert_eq!(XUserGetId(std::ptr::null_mut(), &mut id), E_INVALIDARG);
        }
    }

    #[test]
    fn system_info() {
        unsafe {
            // a buffer of XSystemConsoleIdBytes (39) takes the console id
            let mut id = [0u8; 39];
            let mut used = 0usize;
            assert_eq!(XSystemGetConsoleId(39, id.as_mut_ptr(), &mut used), S_OK);
            assert_eq!(&id[..used - 1], b"00000000.00000000.00000000.00000000.00");
            assert_eq!(XSystemGetDeviceType(), DEVICE_XBOX_SCARLETT_ANACONDA);
            let info = XSystemGetAnalyticsInfo();
            assert!(info.os_version.major >= 10);
            assert_eq!(&info.form[..13], b"Xbox Series X");
            assert_eq!(std::mem::size_of::<XSystemAnalyticsInfo>(), 144);
        }
    }

    #[test]
    fn save_folder_async() {
    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XGameSaveFilesGetFolderWithUiAsync(std::ptr::null_mut(), b"scid-1\0".as_ptr(), &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut buf = [0u8; 512];
            assert_eq!(XGameSaveFilesGetFolderWithUiResult(&mut b, 512, buf.as_mut_ptr()), S_OK);
            let s = std::ffi::CStr::from_ptr(buf.as_ptr() as *const _).to_string_lossy().into_owned();
            assert!(s.ends_with("saves\\scid-1"), "{s}");
            assert!(std::path::Path::new(&s).is_dir());
        }
    }

    fn pls_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("xcl_pls_test_{}_{}", std::process::id(), name))
    }

    #[test]
    fn pls_used_counts_nested_files() {
        let root = pls_test_dir("nested");
        let _ = std::fs::remove_dir_all(&root);
        let deep = root.join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(root.join("top.bin"), [0u8; 100]).unwrap();
        std::fs::write(root.join("a").join("mid.bin"), [0u8; 50]).unwrap();
        std::fs::write(deep.join("deep.bin"), [0u8; 250]).unwrap();
        assert_eq!(pls_used_bytes(&root), 400);
        assert_eq!(pls_used_bytes(&root.join("missing")), 0); // unreadable → 0, not an error
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn pls_space_math_quota_and_saturation() {
        const MIB: u64 = 1024 * 1024;
        // the quota is reduced by measured use when the host disk is not the limit
        let s = pls_space_info(4 * MIB, 300_000, u64::MAX, u64::MAX);
        assert_eq!((s.used_bytes, s.total_bytes), (300_000, 4 * MIB));
        assert_eq!(s.available_free_bytes, 4 * MIB - 300_000);
        assert_eq!(s.total_free_bytes, 4 * MIB - 300_000);
        // the host disk is the tighter limit
        let s = pls_space_info(4 * MIB, 0, 1234, 5678);
        assert_eq!((s.available_free_bytes, s.total_free_bytes), (1234, 5678));
        // over quota: free saturates at 0 while used stays truthful
        let s = pls_space_info(4 * MIB, 4 * MIB + 7, u64::MAX, u64::MAX);
        assert_eq!((s.available_free_bytes, s.total_free_bytes), (0, 0));
        assert_eq!(s.used_bytes, 4 * MIB + 7);
        // a huge configured quota never overflows or underflows
        let s = pls_space_info(u64::MAX, u64::MAX - 1, u64::MAX, u64::MAX);
        assert_eq!((s.available_free_bytes, s.total_free_bytes), (1, 1));
        let s = pls_space_info(10, 20, 0, u64::MAX);
        assert_eq!((s.available_free_bytes, s.total_free_bytes), (0, 0));
    }

    #[cfg(windows)]
    #[test]
    fn pls_used_does_not_follow_dir_links() {
        let base = pls_test_dir("links");
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("pls");
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("escape.bin"), [0u8; 777]).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("real.bin"), [0u8; 111]).unwrap();
        // a junction needs no elevation (unlike a symlink); skip when it can't be made
        let made = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(root.join("linked"))
            .arg(&outside)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !made {
            let _ = std::fs::remove_dir_all(&base);
            eprintln!("mklink /J unavailable – link case skipped");
            return;
        }
        assert_eq!(pls_used_bytes(&root), 111); // the junction and its target are not counted
        // nor is a file symlink's target, when it can be created (needs the privilege)
        if std::os::windows::fs::symlink_file(&outside.join("escape.bin"), &root.join("flink.bin")).is_ok() {
            assert_eq!(pls_used_bytes(&root), 111);
        }
        let _ = std::fs::remove_dir_all(&base); // remove_dir_all removes the link, not the target
    }

    #[test]
    fn pls_space_info_export() {
        unsafe {
            assert_eq!(XPersistentLocalStorageGetSpaceInfo(std::ptr::null_mut()), E_INVALIDARG);
            let mut info: XPersistentLocalStorageSpaceInfo = std::mem::zeroed();
            assert_eq!(XPersistentLocalStorageGetSpaceInfo(&mut info), S_OK);
            let before = info.used_bytes;
            let probe = data_subdir("pls").join(format!("probe_{}.bin", std::process::id()));
            std::fs::write(&probe, [0u8; 321]).unwrap();
            assert_eq!(XPersistentLocalStorageGetSpaceInfo(&mut info), S_OK);
            assert_eq!(info.used_bytes, before + 321);
            assert_eq!(info.used_bytes, pls_used_bytes(&data_subdir("pls")));
            assert_eq!(info.total_bytes, 4096u64 << 20); // default quota: 4 GiB
            assert!(info.available_free_bytes <= info.total_free_bytes);
            assert!(info.total_free_bytes + info.used_bytes <= info.total_bytes);
            let _ = std::fs::remove_file(&probe);
        }
    }
}
