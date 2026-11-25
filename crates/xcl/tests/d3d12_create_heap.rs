//! Verify XCL's `CreateHeap` hook (`d3d12x::real::dev_create_heap`) against the host PC D3D12
//! driver on the same adapter, with no console files or game data.
//!
//! Captures the host device's raw `ID3D12Device::CreateHeap` entry point before
//! `d3d12x::real::init()` adopts the device vtable, then verifies:
//! 1. Standard PC heap descriptions (`DEFAULT`, `UPLOAD`, `READBACK`) succeed through both the
//!    host driver and XCL with matching `ID3D12Heap::GetDesc()` values.
//! 2. Console-only heap descriptions refused by the host driver (console flag bits such as `0x2`
//!    / `0x10` / high bits, unknown heap types outside `1..=5`, unaligned sizes, and alignments
//!    other than 64 KiB / 4 MiB) succeed through XCL with normalized PC descriptors.
//! 3. An unknown console `IID` fails on the raw host driver with `E_NOINTERFACE` and succeeds
//!    through XCL's `creator!` fallback returning a usable `ID3D12Heap`.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::core::{GUID, Interface};
use windows::Win32::Foundation::E_NOINTERFACE;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use xcl::gpu::{adapter, agility, recomp::runner};
use xcl::hle::d3d12x::{layout, real};

type CreateHeapFn =
    unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32;

struct Fixture {
    dev: usize,
    raw_create_heap: CreateHeapFn,
    xcl_create_heap: CreateHeapFn,
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

        let slot = layout::slot(layout::find("ID3D12Device").expect("ID3D12Device layout"), "CreateHeap")
            .expect("CreateHeap slot");
        let host_luid = raw_dev.GetAdapterLuid();
        let raw_create_heap: CreateHeapFn =
            std::mem::transmute(*(*(raw_dev.as_raw() as *const *const usize)).add(slot));
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

        let xcl_create_heap: CreateHeapFn =
            std::mem::transmute(*(*(xcl_dev as *const *const usize)).add(slot));
        assert_ne!(
            xcl_create_heap as usize, raw_create_heap as usize,
            "adopted device CreateHeap slot must point to XCL's hook"
        );
        println!("CreateHeap host comparison adapter: {adapter_name}");

        Some(Fixture { dev: xcl_dev as usize, raw_create_heap, xcl_create_heap })
    })
    .as_ref()
}

unsafe fn call_create_heap(
    f: CreateHeapFn,
    dev: *mut c_void,
    desc: &D3D12_HEAP_DESC,
    iid: &GUID,
) -> (i32, Option<ID3D12Heap>) {
    let mut raw: *mut c_void = std::ptr::null_mut();
    let hr = f(dev, desc as *const _, iid as *const _, &mut raw);
    let heap = if hr >= 0 && !raw.is_null() {
        Some(ID3D12Heap::from_raw(raw))
    } else {
        None
    };
    (hr, heap)
}

#[test]
fn pc_heap_descs_match_host_driver() {
    let Some(fix) = fixture() else { return };
    let dev = fix.dev as *mut c_void;

    for (name, ty, flags) in [
        ("DEFAULT_BUFFERS", D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
        ("UPLOAD_BUFFERS", D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
        ("READBACK_BUFFERS", D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
    ] {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: 0x1_0000,
            Properties: D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() },
            Alignment: 0,
            Flags: flags,
        };
        let (hr_host, host_heap) = unsafe { call_create_heap(fix.raw_create_heap, dev, &desc, &ID3D12Heap::IID) };
        let (hr_xcl, xcl_heap) = unsafe { call_create_heap(fix.xcl_create_heap, dev, &desc, &ID3D12Heap::IID) };
        assert!(hr_host >= 0, "{name}: host CreateHeap failed ({hr_host:#x})");
        assert_eq!(hr_xcl, hr_host, "{name}: HRESULT mismatch (xcl {hr_xcl:#x} vs host {hr_host:#x})");

        let hd = unsafe { host_heap.expect("host heap").GetDesc() };
        let xd = unsafe { xcl_heap.expect("xcl heap").GetDesc() };
        assert_eq!(
            (xd.SizeInBytes, xd.Properties.Type, xd.Alignment, xd.Flags),
            (hd.SizeInBytes, hd.Properties.Type, hd.Alignment, hd.Flags),
            "{name}: GetDesc() mismatch against host driver"
        );
    }
}

#[test]
fn console_heap_descs_refused_by_host_are_normalized_by_xcl() {
    let Some(fix) = fixture() else { return };
    let dev = fix.dev as *mut c_void;
    const PC_HEAP_FLAGS: u32 = 0x3FED;

    // (label, desc, expected_type, expected_min_size, expected_min_align)
    let cases = [
        (
            "console_flags_0xC2_and_high_bits",
            D3D12_HEAP_DESC {
                SizeInBytes: 0x1_0000,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAGS((0x1000_00C2u32) as i32),
            },
            D3D12_HEAP_TYPE_DEFAULT,
            0x1_0000u64,
            0x1_0000u64,
        ),
        (
            "unknown_console_heap_type",
            D3D12_HEAP_DESC {
                SizeInBytes: 0x1_0000,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE(7), ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            },
            D3D12_HEAP_TYPE_DEFAULT,
            0x1_0000u64,
            0x1_0000u64,
        ),
        (
            "unaligned_size_and_small_non_pc_alignment",
            D3D12_HEAP_DESC {
                SizeInBytes: 0x1_2345,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0x2000,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            },
            D3D12_HEAP_TYPE_DEFAULT,
            0x2_0000u64,
            0x1_0000u64,
        ),
        (
            "alignment_above_64k_promoted_to_4m",
            D3D12_HEAP_DESC {
                SizeInBytes: 0x40_0000,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0x2_0000,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            },
            D3D12_HEAP_TYPE_DEFAULT,
            0x40_0000u64,
            D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64,
        ),
        (
            "combined_console_heap",
            D3D12_HEAP_DESC {
                SizeInBytes: 0x1_0001,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE(9), ..Default::default() },
                Alignment: 0x4000,
                Flags: D3D12_HEAP_FLAGS((0x2000_00D2u32) as i32),
            },
            D3D12_HEAP_TYPE_DEFAULT,
            0x2_0000u64,
            0x1_0000u64,
        ),
    ];

    let mut normalized_count = 0usize;
    for (name, desc, exp_type, exp_size, exp_align) in cases {
        let (hr_host, _) = unsafe { call_create_heap(fix.raw_create_heap, dev, &desc, &ID3D12Heap::IID) };
        let (hr_xcl, heap) = unsafe { call_create_heap(fix.xcl_create_heap, dev, &desc, &ID3D12Heap::IID) };
        assert!(hr_xcl >= 0, "{name}: XCL CreateHeap failed ({hr_xcl:#x})");
        let actual = unsafe { heap.expect("heap").GetDesc() };
        if hr_host >= 0 {
            // Some host drivers (e.g. vkd3d-proton under Wine) accept unknown heap flag bits
            // directly, so XCL returns the host's first-attempt heap without normalizing.
            assert_eq!(hr_xcl, hr_host, "{name}: HRESULT mismatch when host accepted desc");
            assert!(actual.SizeInBytes >= exp_size, "{name}: heap size {:#x} < {exp_size:#x}", actual.SizeInBytes);
            continue;
        }

        normalized_count += 1;
        assert_eq!(actual.Properties.Type, exp_type, "{name}: heap type not normalized");
        assert_eq!(actual.SizeInBytes, exp_size, "{name}: heap size not rounded to 64 KiB");
        assert!(actual.Alignment >= exp_align, "{name}: alignment {:#x} < {exp_align:#x}", actual.Alignment);
        assert_eq!(
            (actual.Flags.0 as u32) & !PC_HEAP_FLAGS,
            0,
            "{name}: console-only heap flag bits remained: {:#x}",
            actual.Flags.0
        );
    }
    assert!(
        normalized_count >= 4,
        "expected at least 4 console heap descs to be refused by host and normalized by XCL, got {normalized_count}"
    );
}

#[test]
fn unknown_console_iid_falls_back_to_pc_heap_interface() {
    let Some(fix) = fixture() else { return };
    let dev = fix.dev as *mut c_void;
    let console_iid = GUID::from_u128(0xdeadbeef_1234_5678_9abc_def012345678);
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: 0x1_0000,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };

    let (hr_host, _) = unsafe { call_create_heap(fix.raw_create_heap, dev, &desc, &console_iid) };
    assert_eq!(hr_host, E_NOINTERFACE.0, "host driver should reject unknown IID with E_NOINTERFACE");

    let (hr_xcl, heap) = unsafe { call_create_heap(fix.xcl_create_heap, dev, &desc, &console_iid) };
    assert!(hr_xcl >= 0, "XCL CreateHeap should fall back on unknown IID ({hr_xcl:#x})");
    let actual = unsafe { heap.expect("fallback heap").GetDesc() };
    assert_eq!(actual.SizeInBytes, 0x1_0000);
    assert_eq!(actual.Properties.Type, D3D12_HEAP_TYPE_DEFAULT);
}
