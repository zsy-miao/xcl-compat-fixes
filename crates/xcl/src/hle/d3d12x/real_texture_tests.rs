use super::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};

fn texture_desc() -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 256,
        Height: 256,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        ..Default::default()
    }
}

#[test]
fn texture_backing_size_checks_overflow_mips_arrays_and_samples() {
    let mut d = texture_desc();
    assert!(texture_memory_bytes(&d).unwrap() >= 0x4_0000);
    d.SampleDesc.Count = 4;
    assert!(texture_memory_bytes(&d).unwrap() >= 0x10_0000);
    d.SampleDesc.Count = 1;
    d.DepthOrArraySize = 2;
    d.MipLevels = 0;
    assert!(texture_memory_bytes(&d).unwrap() >= 2 * 0x4_0000);
    d.Width = u64::MAX;
    assert_eq!(texture_memory_bytes(&d), None);
    d = texture_desc();
    d.Width = 1 << 20;
    d.Height = 1 << 20;
    assert_eq!(texture_memory_bytes(&d), None);
    d = texture_desc();
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    assert_eq!(texture_memory_bytes(&d), None);
}

#[test]
fn committed_texture_address_survives_the_retail_clear_and_is_freed() {
    unsafe {
        let mut device: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device).is_err() {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP adapter");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut device).expect("D3D12 device");
        }
        let device = device.expect("D3D12 device");
        let mut resource: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &texture_desc(),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        ).expect("256x256 R32_FLOAT texture");
        let resource = resource.expect("resource");
        assert_eq!(resource.GetGPUVirtualAddress(), 0, "PC textures have no buffer GPU address");
        let raw = resource.as_raw();
        let key = raw as usize;
        adopt(raw, Kind::Resource);
        // The adopted slot distinguishes title code from PC/runtime callers. This test is
        // host code, so its normal COM call must retain the PC answer; invoke the console
        // target directly with the genuine resource to exercise the title's path.
        assert_eq!(resource.GetGPUVirtualAddress(), 0, "host calls must keep the native PC address");
        let address = res_gpu_va_x(raw) as usize;
        assert_ne!(address, 0);
        assert_eq!(res_gpu_va_x(raw) as usize, address, "the console fallback address must be stable");
        assert_eq!(resource.GetGPUVirtualAddress(), 0, "the fallback must not change PC/runtime calls");
        assert_eq!(title_address(key), Some(address));
        assert_eq!(TEXTURE_MEMORY.lock().unwrap().as_ref().unwrap().get(&key), Some(&address));
        // The current backend reserves lazily. Host tests do not install the title's
        // access-violation handler, so use its normal pre-touch path before writing.
        crate::hle::kernelx::lazy::ensure(address, 0x4_0000);
        // Exact operation that crashed the retail executable: memset(address, 0, 256*256*4).
        std::ptr::write_bytes(address as *mut u8, 0xA5, 0x4_0000);
        std::ptr::write_bytes(address as *mut u8, 0, 0x4_0000);
        assert_eq!(std::ptr::read_volatile(address as *const u8), 0);
        assert_eq!(std::ptr::read_volatile((address + 0x3_FFFF) as *const u8), 0);
        let mut mapped = std::ptr::null_mut();
        assert_eq!(res_map(raw, 0, std::ptr::null(), &mut mapped), S_OK);
        assert_eq!(mapped as usize, address, "Map and GetGPUVirtualAddress must agree");
        res_unmap(raw, 0, std::ptr::null());
        drop(resource);
        assert!(TEXTURE_MEMORY.lock().unwrap().as_ref().unwrap().get(&key).is_none());
        assert_eq!(title_address(key), None, "released resource must not retain the old address");
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_FREE};
        let mut memory: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        assert_ne!(VirtualQuery(address as *const c_void, &mut memory, std::mem::size_of_val(&memory)), 0);
        assert_eq!(memory.State, MEM_FREE, "only XCL-owned texture backing should be released");
    }
}
