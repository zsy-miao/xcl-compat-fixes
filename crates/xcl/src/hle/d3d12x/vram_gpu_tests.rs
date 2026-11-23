use super::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

#[test]
fn aliased_reserved_render_targets_share_title_pages_and_release_users() {
    assert_eq!(crate::hle::d3d12x::backend(), crate::hle::d3d12x::Backend::Real, "real D3D12 backend required");
    unsafe {
        let raw_device = super::super::real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap().clone();
        let title_va = 0x6FED_0000_0000usize;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 128,
            Height: 128,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let iid = ID3D12Resource::IID;
        let mut first_raw = std::ptr::null_mut();
        let hr = place(raw_device, title_va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(PAGE as u64), &iid as *const _ as *const windows_sys::core::GUID, &mut first_raw);
        assert!(hr.is_some_and(|hr| hr >= 0) && !first_raw.is_null(), "first reserved render target");
        let first = ID3D12Resource::from_raw(first_raw);
        assert_eq!(first.GetDesc().Layout, D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, "test must exercise tiled pages, not the arena fallback");
        let first_bytes = page_bytes();
        let mut second_raw = std::ptr::null_mut();
        let hr = place(raw_device, title_va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(PAGE as u64), &iid as *const _ as *const windows_sys::core::GUID, &mut second_raw);
        assert!(hr.is_some_and(|hr| hr >= 0) && !second_raw.is_null(), "second aliased render target");
        let second = ID3D12Resource::from_raw(second_raw);
        assert_eq!(second.GetDesc().Layout, D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);
        assert_ne!(first_raw, second_raw, "distinct resource objects must share their storage");
        assert_eq!(page_bytes(), first_bytes, "the second alias must not allocate duplicate target pages or pools");
        {
            let g = PAGES.lock().unwrap();
            let pages = g.as_ref().unwrap().as_ref().unwrap();
            assert_eq!(pages.targets[&(first_raw as usize)].0, [title_va]);
            assert_eq!(pages.targets[&(second_raw as usize)].0, [title_va]);
            assert_eq!(pages.pages[&title_va].users, 2);
        }
        // Exercise the synchronization path: work on a title queue cannot pass tile mapping.
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        before_execute(&queue);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, &dev, None, 5_000).expect("mapping fence must complete before title work");
        drop(first);
        {
            let g = PAGES.lock().unwrap();
            let pages = g.as_ref().unwrap().as_ref().unwrap();
            assert!(!pages.targets.contains_key(&(first_raw as usize)));
            assert_eq!(pages.pages[&title_va].users, 1);
        }
        drop(second);
        {
            let g = PAGES.lock().unwrap();
            let pages = g.as_ref().unwrap().as_ref().unwrap();
            assert!(!pages.targets.contains_key(&(second_raw as usize)));
            let page = &pages.pages[&title_va];
            assert_eq!(page.users, 0);
            assert!(page.idle_since.is_some(), "unused pages are retained briefly for reuse, not left referenced");
        }
        assert!(!is_placed_texture(first_raw as usize));
        assert!(!is_placed_texture(second_raw as usize));
        dev.GetDeviceRemovedReason().expect("tile alias mapping must not remove the device");
    }
}
