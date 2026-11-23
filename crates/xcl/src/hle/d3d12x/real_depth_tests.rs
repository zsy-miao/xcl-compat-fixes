use super::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

#[repr(C)]
struct ResourceWithDesc {
    vtable: *const usize,
    desc: D3D12_RESOURCE_DESC,
}

unsafe extern "system" fn get_desc(this: *mut c_void, out: *mut D3D12_RESOURCE_DESC) {
    out.write((*(this as *const ResourceWithDesc)).desc);
}

fn with_resource_desc<T>(format: i32, f: impl FnOnce(*mut c_void) -> T) -> T {
    // pc_depth_srv only borrows the interface and calls ID3D12Resource::GetDesc (#10).
    let mut vtable = [0usize; 15];
    vtable[10] = get_desc as usize;
    let mut resource = ResourceWithDesc {
        vtable: vtable.as_ptr(),
        desc: D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Format: DXGI_FORMAT(format), DepthOrArraySize: 1, MipLevels: 1, SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
    };
    f(&mut resource as *mut ResourceWithDesc as *mut c_void)
}

#[test]
fn console_depth_views_become_pc_ones() {
    for (resource, view, expected) in [
        (19, 41, 21), // a plain R32_FLOAT name for D32S8's depth plane
        (19, 61, 22), // an R8 name for D32S8's stencil plane
        (44, 56, 46), // a plain R16_UNORM name for the mapped D16S8 depth
        (44, 62, 47), // an R8_UINT name for the mapped stencil plane
        (39, 0, 41),
        (53, 0, 56),
        (28, 28, 28),
    ] {
        assert_eq!(pc_depth_view_format(resource, view), expected);
    }

    with_resource_desc(19, |resource| unsafe {
        let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(41),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: 5768,
            ..Default::default()
        };
        desc.Anonymous.Texture2D.MostDetailedMip = 2;
        desc.Anonymous.Texture2D.MipLevels = 3;
        let converted = pc_depth_srv(resource, &desc).expect("plain depth view is converted");
        assert_eq!(converted.Format.0, 21);
        assert_eq!(converted.Shader4ComponentMapping, desc.Shader4ComponentMapping);
        assert_eq!(converted.Anonymous.Texture2D.MostDetailedMip, 2);
        assert_eq!(converted.Anonymous.Texture2D.MipLevels, 3);
        assert_eq!(converted.Anonymous.Texture2D.PlaneSlice, 0);

        desc.Format = DXGI_FORMAT(22);
        let stencil = pc_depth_srv(resource, &desc).expect("existing stencil format still selects plane 1");
        assert_eq!(stencil.Format.0, 22);
        assert_eq!(stencil.Anonymous.Texture2D.PlaneSlice, 1);
    });
}

#[test]
fn console_depth_stencil_array_keeps_plane_after_format_conversion() {
    with_resource_desc(44, |resource| unsafe {
        let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(120),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: 5768,
            ..Default::default()
        };
        desc.Anonymous.Texture2DArray.MostDetailedMip = 2;
        desc.Anonymous.Texture2DArray.MipLevels = 3;
        desc.Anonymous.Texture2DArray.FirstArraySlice = 4;
        desc.Anonymous.Texture2DArray.ArraySize = 5;
        desc.Anonymous.Texture2DArray.ResourceMinLODClamp = 0.75;
        // Match dev_create_srv_x: console format mapping precedes depth-plane selection.
        let mapped = pc_view_format(&desc).expect("console stencil format is mapped");
        assert_eq!(mapped.Format.0, 47);
        let converted = pc_depth_srv(resource, &mapped).expect("mapped stencil still selects plane 1");
        let array = converted.Anonymous.Texture2DArray;
        assert_eq!(converted.Format.0, 47);
        assert_eq!(array.PlaneSlice, 1);
        assert_eq!(array.MostDetailedMip, 2);
        assert_eq!(array.MipLevels, 3);
        assert_eq!(array.FirstArraySlice, 4);
        assert_eq!(array.ArraySize, 5);
        assert_eq!(array.ResourceMinLODClamp, 0.75);
        assert_eq!(desc.Format.0, 120);
        assert_eq!(desc.Anonymous.Texture2DArray.PlaneSlice, 0);
    });
}

#[test]
fn typeless_depth_defaults_have_typed_views_and_keep_explicit_fields() {
    let mut resource = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Format: DXGI_FORMAT(44), DepthOrArraySize: 3, MipLevels: 2,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let dsv = pc_depth_dsv(&resource, None).expect("typeless resource needs a typed default DSV");
    assert_eq!(dsv.Format.0, 45);
    assert_eq!(dsv.ViewDimension, D3D12_DSV_DIMENSION_TEXTURE2DARRAY);
    unsafe { assert_eq!(dsv.Anonymous.Texture2DArray.ArraySize, 3); }
    let mut explicit = dsv;
    explicit.Format.0 = 0;
    explicit.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    explicit.Anonymous.Texture2DArray.MipSlice = 1;
    let mapped = pc_depth_dsv(&resource, Some(&explicit)).unwrap();
    assert_eq!(mapped.Format.0, 45);
    assert_eq!(mapped.Flags, explicit.Flags);
    unsafe { assert_eq!(mapped.Anonymous.Texture2DArray.MipSlice, 1); }
    resource.SampleDesc.Count = 4;
    assert_eq!(pc_depth_dsv(&resource, None).unwrap().ViewDimension, D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY);
    for (resource, depth) in [(19, 21), (39, 41), (44, 46), (53, 56)] {
        with_resource_desc(resource, |raw| unsafe {
            let srv = pc_depth_srv(raw, std::ptr::null()).expect("typeless resource needs a typed default SRV");
            assert_eq!(srv.Format.0, depth);
            assert_eq!(srv.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
            assert_eq!(srv.Anonymous.Texture2D.MipLevels, u32::MAX);
        });
    }
}

#[test]
fn shader_view_conversion_preserves_plain_packed_textures() {
    for (resource, view, expected) in [(86, 86, 86), (28, 86, 28), (44, 119, 46), (44, 120, 47)] {
        assert_eq!(super::super::device_x::pc_shader_view_format(resource, view), expected);
    }
}

#[test]
fn console_depth_view_format_preserves_descriptor_fields_and_plain_views() {
    unsafe {
        let mut desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT(118),
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Flags: D3D12_DSV_FLAGS(3),
            ..Default::default()
        };
        desc.Anonymous.Texture2DArray.MipSlice = 2;
        desc.Anonymous.Texture2DArray.FirstArraySlice = 3;
        desc.Anonymous.Texture2DArray.ArraySize = 4;
        let converted = pc_view_format(&desc).expect("console DSV format is mapped");
        assert_eq!(converted.Format.0, 45);
        assert_eq!(converted.ViewDimension, desc.ViewDimension);
        assert_eq!(converted.Flags, desc.Flags);
        assert_eq!(converted.Anonymous.Texture2DArray.MipSlice, 2);
        assert_eq!(converted.Anonymous.Texture2DArray.FirstArraySlice, 3);
        assert_eq!(converted.Anonymous.Texture2DArray.ArraySize, 4);
        assert_eq!(desc.Format.0, 118);
        desc.Format = DXGI_FORMAT(45);
        assert!(pc_view_format(&desc).is_none());
        assert!(pc_view_format::<D3D12_DEPTH_STENCIL_VIEW_DESC>(std::ptr::null()).is_none());
    }
    with_resource_desc(28, |resource| unsafe {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT(28), ..Default::default() };
        assert!(pc_depth_srv(resource, &desc).is_none());
        assert!(pc_depth_srv(resource, std::ptr::null()).is_none());
    });
}
