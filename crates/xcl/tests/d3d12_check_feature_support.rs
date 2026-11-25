//! Compare XCL's `CheckFeatureSupport` hook (`d3d12x::real`) against the host PC D3D12 driver on
//! the same adapter, with no console files or game data.
//!
//! Captures the host device's raw `ID3D12Device::CheckFeatureSupport` entry point before
//! `d3d12x::real::init()` adopts the device vtable, then verifies:
//! 1. Exact PC-sized queries (`D3D12_OPTIONS` through `D3D12_OPTIONS13`, architecture, shader
//!    model, root signature, virtual address support, shader cache, existing heaps, and format
//!    queries) match the host driver's HRESULT and output bytes.
//! 2. Oversized console-sized structs (`pc_size + 8`, as titles such as XDefiant pass) fail on the
//!    raw host driver, succeed through XCL with the host driver's prefix bytes, and zero the
//!    trailing console bytes.
//! 3. Console-only formats (`116..=120`) return the host driver's answers for their mapped PC
//!    formats (`10`, `45`, `46`, `47`) while restoring the caller's console format field.

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::OnceLock;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use xcl::gpu::{adapter, agility, recomp::runner};
use xcl::hle::d3d12x::real;

type CheckFeatureSupportFn = unsafe extern "system" fn(*mut c_void, i32, *mut u8, u32) -> i32;

struct Fixture {
    dev: usize,
    raw_cfs: CheckFeatureSupportFn,
    xcl_cfs: CheckFeatureSupportFn,
}

fn fixture() -> Option<&'static Fixture> {
    static FIX: OnceLock<Option<Fixture>> = OnceLock::new();
    FIX.get_or_init(|| unsafe {
        let factory: IDXGIFactory6 = match CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) {
            Ok(f) => f,
            Err(_) => {
                runner::skip_test("no DXGI factory");
                return None;
            }
        };
        let sel = adapter::Selector::from_env_or(&xcl::state::config().d3d12_adapter);
        let mut raw_dev: Option<ID3D12Device> = None;
        if sel != adapter::Selector::Warp {
            let mut list = adapter::candidates(&factory, &sel);
            if list.is_empty() {
                list = adapter::candidates(&factory, &adapter::Selector::Auto);
            }
            for (adp, _) in list {
                if let Ok(d) = agility::create_device(&adp, D3D_FEATURE_LEVEL_12_0) {
                    raw_dev = Some(d);
                    break;
                }
            }
        }
        let raw_dev = match raw_dev {
            Some(d) => d,
            None => match factory
                .EnumWarpAdapter::<IDXGIAdapter>()
                .ok()
                .and_then(|w| agility::create_device(&w, D3D_FEATURE_LEVEL_11_0).ok())
            {
                Some(d) => d,
                None => {
                    runner::skip_test("no D3D12 device");
                    return None;
                }
            },
        };

        // Snapshot vtable #13 (CheckFeatureSupport) and adapter LUID before real::init() adopts it.
        let host_luid = raw_dev.GetAdapterLuid();
        let raw_cfs: CheckFeatureSupportFn =
            std::mem::transmute(*(*(raw_dev.as_raw() as *const *const usize)).add(13));
        drop(raw_dev);

        let adapter_name = match real::init() {
            Ok(name) => name,
            Err(_) => {
                runner::skip_test("d3d12x::real::init failed");
                return None;
            }
        };
        let xcl_dev = real::device_raw();
        assert!(!xcl_dev.is_null(), "real::init succeeded with null device");

        let xcl_luid = ID3D12Device::from_raw_borrowed(&xcl_dev)
            .expect("adopted device")
            .GetAdapterLuid();
        assert_eq!(
            (xcl_luid.LowPart, xcl_luid.HighPart),
            (host_luid.LowPart, host_luid.HighPart),
            "host and XCL devices must target the same adapter ({adapter_name})"
        );

        let xcl_cfs: CheckFeatureSupportFn =
            std::mem::transmute(*(*(xcl_dev as *const *const usize)).add(13));
        assert_ne!(
            xcl_cfs as usize, raw_cfs as usize,
            "adopted device vtable #13 must point to XCL's CheckFeatureSupport hook"
        );
        println!("CheckFeatureSupport host comparison adapter: {adapter_name}");

        Some(Fixture { dev: xcl_dev as usize, raw_cfs, xcl_cfs })
    })
    .as_ref()
}

fn as_bytes<T>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts((v as *const T) as *const u8, size_of::<T>()).to_vec() }
}

/// Apply the configured `raytracing` / `XCL_RAYTRACING` override (if any) to a host OPTIONS5
/// buffer so the test still matches XCL's documented override when `XCL_RAYTRACING` is set.
fn apply_rt_override(feature: D3D12_FEATURE, buf: &mut [u8]) {
    if feature == D3D12_FEATURE_D3D12_OPTIONS5
        && buf.len() >= size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>()
    {
        if let Some(tier) = xcl::state::config().raytracing_tier_override() {
            unsafe {
                let opts = &mut *(buf.as_mut_ptr() as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5);
                opts.RaytracingTier = D3D12_RAYTRACING_TIER(tier as i32);
            }
        }
    }
}

/// Feature queries covered by `real::pc_feature_size`, paired with their PC-sized initial inputs.
fn standard_queries() -> Vec<(&'static str, D3D12_FEATURE, Vec<u8>)> {
    vec![
        ("D3D12_OPTIONS", D3D12_FEATURE_D3D12_OPTIONS, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>()]),
        ("ARCHITECTURE", D3D12_FEATURE_ARCHITECTURE, vec![0; size_of::<D3D12_FEATURE_DATA_ARCHITECTURE>()]),
        ("ARCHITECTURE1", D3D12_FEATURE_ARCHITECTURE1, vec![0; size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>()]),
        ("GPU_VIRTUAL_ADDRESS_SUPPORT", D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, vec![0; size_of::<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT>()]),
        ("SHADER_MODEL", D3D12_FEATURE_SHADER_MODEL, as_bytes(&D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_7 })),
        ("ROOT_SIGNATURE", D3D12_FEATURE_ROOT_SIGNATURE, as_bytes(&D3D12_FEATURE_DATA_ROOT_SIGNATURE { HighestVersion: D3D_ROOT_SIGNATURE_VERSION_1_1 })),
        ("SHADER_CACHE", D3D12_FEATURE_SHADER_CACHE, vec![0; size_of::<D3D12_FEATURE_DATA_SHADER_CACHE>()]),
        ("EXISTING_HEAPS", D3D12_FEATURE_EXISTING_HEAPS, vec![0; size_of::<D3D12_FEATURE_DATA_EXISTING_HEAPS>()]),
        ("D3D12_OPTIONS1", D3D12_FEATURE_D3D12_OPTIONS1, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>()]),
        ("D3D12_OPTIONS2", D3D12_FEATURE_D3D12_OPTIONS2, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS2>()]),
        ("D3D12_OPTIONS3", D3D12_FEATURE_D3D12_OPTIONS3, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS3>()]),
        ("D3D12_OPTIONS4", D3D12_FEATURE_D3D12_OPTIONS4, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>()]),
        ("D3D12_OPTIONS5", D3D12_FEATURE_D3D12_OPTIONS5, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>()]),
        ("D3D12_OPTIONS6", D3D12_FEATURE_D3D12_OPTIONS6, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>()]),
        ("D3D12_OPTIONS7", D3D12_FEATURE_D3D12_OPTIONS7, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>()]),
        ("D3D12_OPTIONS8", D3D12_FEATURE_D3D12_OPTIONS8, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS8>()]),
        ("D3D12_OPTIONS9", D3D12_FEATURE_D3D12_OPTIONS9, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>()]),
        ("D3D12_OPTIONS10", D3D12_FEATURE_D3D12_OPTIONS10, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS10>()]),
        ("D3D12_OPTIONS11", D3D12_FEATURE_D3D12_OPTIONS11, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>()]),
        ("D3D12_OPTIONS12", D3D12_FEATURE_D3D12_OPTIONS12, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>()]),
        ("D3D12_OPTIONS13", D3D12_FEATURE_D3D12_OPTIONS13, vec![0; size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS13>()]),
        ("FORMAT_SUPPORT(RGBA8)", D3D12_FEATURE_FORMAT_SUPPORT, as_bytes(&D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ..Default::default() })),
        ("FORMAT_SUPPORT(BC1)", D3D12_FEATURE_FORMAT_SUPPORT, as_bytes(&D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_BC1_UNORM, ..Default::default() })),
        ("FORMAT_SUPPORT(D32)", D3D12_FEATURE_FORMAT_SUPPORT, as_bytes(&D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_D32_FLOAT, ..Default::default() })),
        ("MULTISAMPLE_QUALITY_LEVELS(RGBA8 x4)", D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, as_bytes(&D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleCount: 4, ..Default::default() })),
        ("FORMAT_INFO(D24S8)", D3D12_FEATURE_FORMAT_INFO, as_bytes(&D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT_D24_UNORM_S8_UINT, PlaneCount: 0 })),
    ]
}

#[test]
fn pc_sized_feature_queries_match_host_driver() {
    let Some(fix) = fixture() else { return };
    let dev = fix.dev as *mut c_void;

    for (name, feature, init) in standard_queries() {
        let mut host = init.clone();
        let mut xcl_out = init;
        let hr_host = unsafe { (fix.raw_cfs)(dev, feature.0, host.as_mut_ptr(), host.len() as u32) };
        let hr_xcl = unsafe { (fix.xcl_cfs)(dev, feature.0, xcl_out.as_mut_ptr(), xcl_out.len() as u32) };
        assert_eq!(hr_xcl, hr_host, "{name}: HRESULT mismatch (xcl {hr_xcl:#x} vs host {hr_host:#x})");
        if hr_host >= 0 {
            apply_rt_override(feature, &mut host);
            assert_eq!(xcl_out, host, "{name}: payload differs from host driver");
        }
    }
}

#[test]
fn oversized_console_structs_match_host_and_zero_tail() {
    let Some(fix) = fixture() else { return };
    let dev = fix.dev as *mut c_void;
    const EXTRA: usize = 8;
    let mut checked = 0usize;

    for (name, feature, init) in standard_queries() {
        let pc_size = init.len();
        let mut host = init.clone();
        let hr_host = unsafe { (fix.raw_cfs)(dev, feature.0, host.as_mut_ptr(), pc_size as u32) };
        if hr_host < 0 {
            continue;
        }
        apply_rt_override(feature, &mut host);

        let mut raw_over = init.clone();
        raw_over.resize(pc_size + EXTRA, 0xA5);
        let hr_raw_over = unsafe { (fix.raw_cfs)(dev, feature.0, raw_over.as_mut_ptr(), raw_over.len() as u32) };
        assert!(hr_raw_over < 0, "{name}: host driver unexpectedly accepted oversized buffer ({pc_size} + {EXTRA})");

        let mut xcl_over = init;
        xcl_over.resize(pc_size + EXTRA, 0xA5);
        let hr_xcl_over = unsafe { (fix.xcl_cfs)(dev, feature.0, xcl_over.as_mut_ptr(), xcl_over.len() as u32) };
        assert!(hr_xcl_over >= 0, "{name}: XCL rejected oversized console struct ({hr_xcl_over:#x})");
        assert_eq!(&xcl_over[..pc_size], &host[..], "{name}: oversized prefix differs from host driver");
        assert!(
            xcl_over[pc_size..].iter().all(|&b| b == 0),
            "{name}: trailing console bytes were not zeroed: {:02x?}",
            &xcl_over[pc_size..]
        );
        checked += 1;
    }

    assert!(checked > 0, "expected at least one host-supported D3D12 feature");
}

#[test]
fn console_only_formats_match_mapped_pc_formats_and_restore_format() {
    let Some(fix) = fixture() else { return };
    let dev = fix.dev as *mut c_void;

    for (console_fmt, pc_fmt) in [(116u32, 10u32), (117, 10), (118, 45), (119, 46), (120, 47)] {
        let mut host_fs = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(pc_fmt as i32), ..Default::default() };
        let mut xcl_fs = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(console_fmt as i32), ..Default::default() };
        let hr_host = unsafe { (fix.raw_cfs)(dev, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut host_fs as *mut _ as *mut u8, size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32) };
        let hr_xcl = unsafe { (fix.xcl_cfs)(dev, D3D12_FEATURE_FORMAT_SUPPORT.0, &mut xcl_fs as *mut _ as *mut u8, size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32) };
        assert_eq!(hr_xcl, hr_host, "FORMAT_SUPPORT {console_fmt} -> {pc_fmt}: HRESULT mismatch");
        assert_eq!(xcl_fs.Format.0, console_fmt as i32, "FORMAT_SUPPORT {console_fmt}: caller format not restored");
        assert_eq!((xcl_fs.Support1, xcl_fs.Support2), (host_fs.Support1, host_fs.Support2), "FORMAT_SUPPORT {console_fmt} -> {pc_fmt}: flags mismatch");

        let mut host_ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(pc_fmt as i32), SampleCount: 4, ..Default::default() };
        let mut xcl_ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(console_fmt as i32), SampleCount: 4, ..Default::default() };
        let hr_host = unsafe { (fix.raw_cfs)(dev, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS.0, &mut host_ms as *mut _ as *mut u8, size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32) };
        let hr_xcl = unsafe { (fix.xcl_cfs)(dev, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS.0, &mut xcl_ms as *mut _ as *mut u8, size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32) };
        assert_eq!(hr_xcl, hr_host, "MULTISAMPLE_QUALITY_LEVELS {console_fmt} -> {pc_fmt}: HRESULT mismatch");
        assert_eq!(xcl_ms.Format.0, console_fmt as i32, "MULTISAMPLE_QUALITY_LEVELS {console_fmt}: caller format not restored");
        assert_eq!((xcl_ms.SampleCount, xcl_ms.Flags, xcl_ms.NumQualityLevels), (host_ms.SampleCount, host_ms.Flags, host_ms.NumQualityLevels), "MULTISAMPLE_QUALITY_LEVELS {console_fmt} -> {pc_fmt}: payload mismatch");

        let mut host_fi = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(pc_fmt as i32), PlaneCount: 0 };
        let mut xcl_fi = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(console_fmt as i32), PlaneCount: 0 };
        let hr_host = unsafe { (fix.raw_cfs)(dev, D3D12_FEATURE_FORMAT_INFO.0, &mut host_fi as *mut _ as *mut u8, size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>() as u32) };
        let hr_xcl = unsafe { (fix.xcl_cfs)(dev, D3D12_FEATURE_FORMAT_INFO.0, &mut xcl_fi as *mut _ as *mut u8, size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>() as u32) };
        assert_eq!(hr_xcl, hr_host, "FORMAT_INFO {console_fmt} -> {pc_fmt}: HRESULT mismatch");
        assert_eq!(xcl_fi.Format.0, console_fmt as i32, "FORMAT_INFO {console_fmt}: caller format not restored");
        assert_eq!(xcl_fi.PlaneCount, host_fi.PlaneCount, "FORMAT_INFO {console_fmt} -> {pc_fmt}: plane count mismatch");
    }
}
