//! GFX10 resource descriptors – T# (image, 8 dwords), S# (sampler, 4), V# (buffer, 4) –
//! decoded and expressed in D3D12 terms (DXGI format, SRV with component mapping, sampler).
//! Plain functions: whatever ends up feeding the GPU (descriptor tables the title writes,
//! a cache keyed by the raw words …) can use them. Field positions as in AMD's GFX10
//! register docs, cross-checked with AnyPS5 (GPL-2.0); format numbers = GFX10 IMG_FORMAT.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// GFX10 image TYPE (T# word 3 [31:28]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
    D1,
    D2,
    D3,
    Cube,
    D1Array,
    D2Array,
    D2Msaa,
    D2MsaaArray,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Image {
    /// byte address (BASE_ADDRESS is in 256-byte units)
    pub address: u64,
    /// GFX10 IMG_FORMAT
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// depth (3D) or last array slice + 1 (arrays, cubes: faces)
    pub depth: u32,
    pub ty: ImageType,
    pub base_level: u32,
    pub last_level: u32,
    /// highest mip of the resource
    pub max_mip: u32,
    pub base_array: u32,
    /// DST_SEL x, y, z, w: 0 = 0, 1 = 1, 4..7 = x..w
    pub dst_sel: [u32; 4],
    /// SW_MODE (tiling): 0 = linear
    pub swizzle_mode: u32,
    /// MIN_LOD (u4.8)
    pub min_lod: f32,
}

/// Decode a T#. None for a null / unknown-type descriptor.
pub fn image(w: &[u32; 8]) -> Option<Image> {
    let ty = match w[3] >> 28 {
        8 => ImageType::D1,
        9 => ImageType::D2,
        10 => ImageType::D3,
        11 => ImageType::Cube,
        12 => ImageType::D1Array,
        13 => ImageType::D2Array,
        14 => ImageType::D2Msaa,
        15 => ImageType::D2MsaaArray,
        _ => return None,
    };
    Some(Image {
        address: ((w[0] as u64) | ((w[1] as u64 & 0xFF) << 32)) << 8,
        format: (w[1] >> 20) & 0x1FF,
        width: ((w[1] >> 30) & 3 | (w[2] & 0xFFF) << 2) + 1,
        height: ((w[2] >> 14) & 0x3FFF) + 1,
        depth: (w[4] & 0x1FFF) + 1,
        ty,
        base_level: (w[3] >> 12) & 0xF,
        last_level: (w[3] >> 16) & 0xF,
        max_mip: (w[5] >> 4) & 0xF,
        base_array: (w[4] >> 16) & 0x1FFF,
        dst_sel: [w[3] & 7, (w[3] >> 3) & 7, (w[3] >> 6) & 7, (w[3] >> 9) & 7],
        swizzle_mode: (w[3] >> 20) & 0x1F,
        min_lod: ((w[1] >> 8) & 0xFFF) as f32 / 256.0,
    })
}

/// DXGI format of a GFX10 IMG_FORMAT and where GFX components x, y, z, w live among the
/// DXGI channels R, G, B, A (the B-first packed formats store x in the blue bits).
pub fn dxgi_format(f: u32) -> Option<(DXGI_FORMAT, [u32; 4])> {
    const RGBA: [u32; 4] = [0, 1, 2, 3];
    const BGRA: [u32; 4] = [2, 1, 0, 3];
    let d = match f {
        1 => DXGI_FORMAT_R8_UNORM,
        2 => DXGI_FORMAT_R8_SNORM,
        5 => DXGI_FORMAT_R8_UINT,
        6 => DXGI_FORMAT_R8_SINT,
        7 => DXGI_FORMAT_R16_UNORM,
        8 => DXGI_FORMAT_R16_SNORM,
        11 => DXGI_FORMAT_R16_UINT,
        12 => DXGI_FORMAT_R16_SINT,
        13 => DXGI_FORMAT_R16_FLOAT,
        14 => DXGI_FORMAT_R8G8_UNORM,
        15 => DXGI_FORMAT_R8G8_SNORM,
        18 => DXGI_FORMAT_R8G8_UINT,
        19 => DXGI_FORMAT_R8G8_SINT,
        20 => DXGI_FORMAT_R32_UINT,
        21 => DXGI_FORMAT_R32_SINT,
        22 => DXGI_FORMAT_R32_FLOAT,
        23 => DXGI_FORMAT_R16G16_UNORM,
        24 => DXGI_FORMAT_R16G16_SNORM,
        27 => DXGI_FORMAT_R16G16_UINT,
        28 => DXGI_FORMAT_R16G16_SINT,
        29 => DXGI_FORMAT_R16G16_FLOAT,
        36 => DXGI_FORMAT_R11G11B10_FLOAT,
        50 => DXGI_FORMAT_R10G10B10A2_UNORM,
        54 => DXGI_FORMAT_R10G10B10A2_UINT,
        56 => DXGI_FORMAT_R8G8B8A8_UNORM,
        57 => DXGI_FORMAT_R8G8B8A8_SNORM,
        60 => DXGI_FORMAT_R8G8B8A8_UINT,
        61 => DXGI_FORMAT_R8G8B8A8_SINT,
        62 => DXGI_FORMAT_R32G32_UINT,
        63 => DXGI_FORMAT_R32G32_SINT,
        64 => DXGI_FORMAT_R32G32_FLOAT,
        65 => DXGI_FORMAT_R16G16B16A16_UNORM,
        66 => DXGI_FORMAT_R16G16B16A16_SNORM,
        69 => DXGI_FORMAT_R16G16B16A16_UINT,
        70 => DXGI_FORMAT_R16G16B16A16_SINT,
        71 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        72 => DXGI_FORMAT_R32G32B32_UINT,
        73 => DXGI_FORMAT_R32G32B32_SINT,
        74 => DXGI_FORMAT_R32G32B32_FLOAT,
        75 => DXGI_FORMAT_R32G32B32A32_UINT,
        76 => DXGI_FORMAT_R32G32B32A32_SINT,
        77 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        130 => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        132 => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        133 => return Some((DXGI_FORMAT_B5G6R5_UNORM, BGRA)),
        // 1_5_5_5 (names list the fields from the top bit down): x in bits 0-4, w in bit 15.
        // 5_5_5_1 (135) keeps its 1-bit field in bit 0, which no DXGI format has
        134 => return Some((DXGI_FORMAT_B5G5R5A1_UNORM, BGRA)),
        136 => return Some((DXGI_FORMAT_B4G4R4A4_UNORM, BGRA)),
        169 => DXGI_FORMAT_BC1_UNORM,
        170 => DXGI_FORMAT_BC1_UNORM_SRGB,
        171 => DXGI_FORMAT_BC2_UNORM,
        172 => DXGI_FORMAT_BC2_UNORM_SRGB,
        173 => DXGI_FORMAT_BC3_UNORM,
        174 => DXGI_FORMAT_BC3_UNORM_SRGB,
        175 => DXGI_FORMAT_BC4_UNORM,
        176 => DXGI_FORMAT_BC4_SNORM,
        177 => DXGI_FORMAT_BC5_UNORM,
        178 => DXGI_FORMAT_BC5_SNORM,
        179 => DXGI_FORMAT_BC6H_UF16,
        180 => DXGI_FORMAT_BC6H_SF16,
        181 => DXGI_FORMAT_BC7_UNORM,
        182 => DXGI_FORMAT_BC7_UNORM_SRGB,
        _ => return None,
    };
    Some((d, RGBA))
}

/// D3D12 component mapping for a T#'s DST_SEL on a DXGI format with the given channel order.
pub fn component_mapping(dst_sel: [u32; 4], order: [u32; 4]) -> u32 {
    let src = |sel: u32| match sel {
        0 => D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0.0 as u32,
        1 => D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1.0 as u32,
        4..=7 => order[(sel - 4) as usize],
        _ => D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0.0 as u32,
    };
    // D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING
    src(dst_sel[0]) | src(dst_sel[1]) << 3 | src(dst_sel[2]) << 6 | src(dst_sel[3]) << 9 | 1 << 12
}

/// Resource and SRV descriptions of a T# (None: format without a DXGI equivalent).
pub fn valid_cube_view(t: &Image) -> bool {
    if t.ty != ImageType::Cube {
        return true;
    }
    t.depth.checked_sub(t.base_array).is_some_and(|faces| faces != 0 && faces % 6 == 0)
}

pub fn valid_image_shape(t: &Image) -> bool {
    if t.width == 0 || (!matches!(t.ty, ImageType::D1 | ImageType::D1Array) && t.height == 0) {
        return false;
    }
    if matches!(t.ty, ImageType::D1Array | ImageType::D2Array | ImageType::D2MsaaArray) && (t.depth == 0 || t.base_array >= t.depth) {
        return false;
    }
    if t.ty == ImageType::D3 && t.depth == 0 {
        return false;
    }
    if matches!(t.ty, ImageType::D1Array | ImageType::D2Array | ImageType::D2MsaaArray | ImageType::Cube | ImageType::D3) && t.depth > u16::MAX as u32 {
        return false;
    }
    let within_d3d12_limits = match t.ty {
        ImageType::D1 => t.width <= D3D12_REQ_TEXTURE1D_U_DIMENSION,
        ImageType::D1Array => t.width <= D3D12_REQ_TEXTURE1D_U_DIMENSION && t.depth <= D3D12_REQ_TEXTURE1D_ARRAY_AXIS_DIMENSION,
        ImageType::D2 | ImageType::D2Msaa => t.width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION && t.height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
        ImageType::D2Array | ImageType::D2MsaaArray | ImageType::Cube => {
            t.width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
                && t.height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
                && t.depth <= D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION
        }
        ImageType::D3 => {
            t.width <= D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION
                && t.height <= D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION
                && t.depth <= D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION
        }
    };
    if !within_d3d12_limits {
        return false;
    }
    if matches!(t.ty, ImageType::D2Msaa | ImageType::D2MsaaArray) {
        return true;
    }
    let largest = match t.ty {
        ImageType::D1 | ImageType::D1Array => t.width,
        ImageType::D3 => t.width.max(t.height).max(t.depth),
        _ => t.width.max(t.height),
    };
    let highest = 31 - largest.leading_zeros();
    t.base_level <= t.last_level && t.last_level <= t.max_mip && t.max_mip <= highest
}

pub fn image_d3d12(t: &Image) -> Option<(D3D12_RESOURCE_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
    if !valid_cube_view(t) || !valid_image_shape(t) {
        return None;
    }
    let (format, order) = dxgi_format(t.format)?;
    let mips = t.max_mip + 1;
    let (dim, depth) = match t.ty {
        ImageType::D1 => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, 1),
        ImageType::D1Array => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, t.depth),
        ImageType::D3 => (D3D12_RESOURCE_DIMENSION_TEXTURE3D, t.depth),
        ImageType::D2 | ImageType::D2Msaa => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1),
        ImageType::D2Array | ImageType::D2MsaaArray | ImageType::Cube => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, t.depth),
    };
    let samples = if matches!(t.ty, ImageType::D2Msaa | ImageType::D2MsaaArray) { 1u32 << t.last_level.min(4) } else { 1 };
    let res = D3D12_RESOURCE_DESC {
        Dimension: dim,
        Width: t.width as u64,
        Height: if dim == D3D12_RESOURCE_DIMENSION_TEXTURE1D { 1 } else { t.height },
        DepthOrArraySize: depth as u16,
        MipLevels: if samples > 1 { 1 } else { mips as u16 },
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
        ..Default::default()
    };
    let levels = t.last_level.saturating_sub(t.base_level) + 1;
    let slices = depth.saturating_sub(t.base_array).max(1);
    let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, Shader4ComponentMapping: component_mapping(t.dst_sel, order), ..Default::default() };
    let lod = t.min_lod;
    match t.ty {
        ImageType::D1 => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            v.Anonymous.Texture1D = D3D12_TEX1D_SRV { MostDetailedMip: t.base_level, MipLevels: levels, ResourceMinLODClamp: lod };
        }
        ImageType::D1Array => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            v.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { MostDetailedMip: t.base_level, MipLevels: levels, FirstArraySlice: t.base_array, ArraySize: slices, ResourceMinLODClamp: lod };
        }
        ImageType::D2 => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            v.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: t.base_level, MipLevels: levels, PlaneSlice: 0, ResourceMinLODClamp: lod };
        }
        ImageType::D2Array => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            v.Anonymous.Texture2DArray =
                D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: t.base_level, MipLevels: levels, FirstArraySlice: t.base_array, ArraySize: slices, PlaneSlice: 0, ResourceMinLODClamp: lod };
        }
        ImageType::D3 => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            v.Anonymous.Texture3D = D3D12_TEX3D_SRV { MostDetailedMip: t.base_level, MipLevels: levels, ResourceMinLODClamp: lod };
        }
        // always a cube array (of one cube, too): translated shaders declare TextureCubeArray,
        // as a cube T# says nothing at compile time about how many cubes it has
        ImageType::Cube => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            v.Anonymous.TextureCubeArray =
                D3D12_TEXCUBE_ARRAY_SRV { MostDetailedMip: t.base_level, MipLevels: levels, First2DArrayFace: t.base_array, NumCubes: slices / 6, ResourceMinLODClamp: lod };
        }
        ImageType::D2Msaa => v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS,
        ImageType::D2MsaaArray => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            v.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: t.base_array, ArraySize: slices };
        }
    }
    Some((res, v))
}

/// D3D12 sampler from an S#. Border colours from the console's table (type 3) become
/// transparent black. `comparison` selects the comparison variant (image_*_c).
pub fn sampler_d3d12(w: &[u32; 4], comparison: bool) -> D3D12_SAMPLER_DESC {
    let address = |c: u32| match c {
        0 => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        1 => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        2 | 4 => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        6 => D3D12_TEXTURE_ADDRESS_MODE_BORDER,
        _ => D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
    };
    let mag = (w[2] >> 20) & 3;
    let min = (w[2] >> 22) & 3;
    let mip = (w[2] >> 26) & 3;
    let aniso = 1u32 << ((w[0] >> 9) & 7).min(4);
    let reduction = match (comparison, (w[0] >> 29) & 3) {
        (true, _) => 1,
        (false, 1) => 2, // min
        (false, 2) => 3, // max
        _ => 0,
    };
    let filter = if (mag >= 2 || min >= 2) && aniso > 1 {
        0x55 | reduction << 7 // D3D12_ENCODE_ANISOTROPIC_FILTER
    } else {
        // D3D12_ENCODE_BASIC_FILTER(min, mag, mip, reduction): linear = 1
        let lin = |x: u32| (x & 1 == 1 || x >= 2) as u32;
        (lin(min) << 4) | (lin(mag) << 2) | ((mip == 2) as u32) | reduction << 7
    };
    let s5_8 = |v: u32| ((v as i32) << 18 >> 18) as f32 / 256.0;
    let border = match w[3] >> 30 {
        1 => [0.0, 0.0, 0.0, 1.0],
        2 => [1.0, 1.0, 1.0, 1.0],
        _ => [0.0; 4],
    };
    D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER(filter as i32),
        AddressU: address(w[0] & 7),
        AddressV: address((w[0] >> 3) & 7),
        AddressW: address((w[0] >> 6) & 7),
        MipLODBias: s5_8(w[2] & 0x3FFF),
        MaxAnisotropy: aniso,
        // SQ_TEX_DEPTH_COMPARE never..always = D3D12_COMPARISON_FUNC 1..8
        ComparisonFunc: D3D12_COMPARISON_FUNC(((w[0] >> 12) & 7) as i32 + 1),
        BorderColor: border,
        MinLOD: (w[1] & 0xFFF) as f32 / 256.0,
        MaxLOD: if mip == 0 { (w[1] & 0xFFF) as f32 / 256.0 } else { ((w[1] >> 12) & 0xFFF) as f32 / 256.0 },
    }
}

/// The T# as recompiled shaders read it (XCL shader ABI): BASE_ADDRESS – word 0 and the
/// high byte of word 1 – replaced by the descriptor heap index of the SRV (UAV at index + 1);
/// every other field (format, size, swizzle …) stays as the console wrote it.
pub fn xcl_image_words(t: &[u32; 8], srv_heap_index: u32) -> [u32; 8] {
    let mut w = *t;
    w[0] = srv_heap_index;
    w[1] &= !0xFF;
    w
}

/// The S# as recompiled shaders read it: sampler heap index in word 3 [11:0]
/// (BORDER_COLOR_PTR; the border colour type stays).
pub fn xcl_sampler_words(s: &[u32; 4], sampler_heap_index: u32) -> [u32; 4] {
    let mut w = *s;
    w[3] = (w[3] & !0xFFF) | (sampler_heap_index & 0xFFF);
    w
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    pub address: u64,
    pub stride: u32,
    pub num_records: u32,
    pub dst_sel: [u32; 4],
    /// GFX10 buffer FORMAT (word 3 [18:12])
    pub format: u32,
}

pub fn buffer(w: &[u32; 4]) -> Buffer {
    Buffer {
        address: w[0] as u64 | ((w[1] as u64 & 0xFFFF) << 32),
        stride: (w[1] >> 16) & 0x3FFF,
        num_records: w[2],
        dst_sel: [w[3] & 7, (w[3] >> 3) & 7, (w[3] >> 6) & 7, (w[3] >> 9) & 7],
        format: (w[3] >> 12) & 0x7F,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbv_is_float4_rows() {
        let v = buffer(&cbv_words(0x4_4E10_DC00, 0x600));
        assert_eq!((v.address, v.stride, v.num_records, v.format), (0x4_4E10_DC00, 16, 0x60, 77));
        // an idxen load of row n reads base + 16 n; the scalar-load limit stays the view's size
        assert_eq!(v.num_records * v.stride, 0x600);
        // a size that is not a whole number of rows still covers its last bytes
        assert_eq!(buffer(&cbv_words(0x1000, 0x608)).num_records, 0x61);
    }

    /// T# of a 2D BGRA-swizzled RGBA8 texture, 256×128, 8 mips, view of mips 1..3.
    fn tsharp() -> [u32; 8] {
        let (w, h) = (256u32 - 1, 128u32 - 1);
        [
            0x1234_5600 >> 8,
            0x12 | (3 << 8) | (56 << 20) | (w & 3) << 30,
            (w >> 2) | h << 14,
            6 | 5 << 3 | 4 << 6 | 7 << 9 | 1 << 12 | 3 << 16 | 9 << 28, // z y x w, levels 1..3, 2D
            0,
            7 << 4,
            0,
            0,
        ]
    }

    #[test]
    fn decodes_tsharp() {
        let t = image(&tsharp()).unwrap();
        assert_eq!(t.address, 0x1200_1234_5600); // BASE_ADDRESS (incl. hi byte) counts 256-byte units
        assert_eq!((t.format, t.width, t.height, t.ty), (56, 256, 128, ImageType::D2));
        assert_eq!((t.base_level, t.last_level, t.max_mip, t.dst_sel), (1, 3, 7, [6, 5, 4, 7]));
        assert!((t.min_lod - 3.0 / 256.0).abs() < 1e-6);
        let (res, srv) = image_d3d12(&t).unwrap();
        assert_eq!((res.Width, res.Height, res.MipLevels, res.Format), (256, 128, 8, DXGI_FORMAT_R8G8B8A8_UNORM));
        assert_eq!(srv.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
        assert_eq!(unsafe { (srv.Anonymous.Texture2D.MostDetailedMip, srv.Anonymous.Texture2D.MipLevels) }, (1, 3));
        // r ← z (channel 2), g ← y, b ← x, a ← w
        assert_eq!(srv.Shader4ComponentMapping, 2 | 1 << 3 | 0 << 6 | 3 << 9 | 1 << 12);
        assert!(image(&[0; 8]).is_none());
    }

    #[test]
    fn packed_formats_reorder_channels() {
        // 5_6_5: x lives in the blue bits of DXGI B5G6R5; identity DST_SEL → r ← B channel
        assert_eq!(dxgi_format(133), Some((DXGI_FORMAT_B5G6R5_UNORM, [2, 1, 0, 3])));
        assert_eq!(component_mapping([4, 5, 6, 1], [2, 1, 0, 3]), 2 | 1 << 3 | 0 << 6 | 5 << 9 | 1 << 12);
        assert_eq!(dxgi_format(181).map(|x| x.0), Some(DXGI_FORMAT_BC7_UNORM));
        assert!(dxgi_format(44).is_none()); // 2_10_10_10: no DXGI equivalent
        // 1_5_5_5 is B5G5R5A1 (and the format XCL writes into its views' T#s); 5_5_5_1 has none
        assert_eq!(dxgi_format(134), Some((DXGI_FORMAT_B5G5R5A1_UNORM, [2, 1, 0, 3])));
        assert!(dxgi_format(135).is_none());
        assert_eq!(gfx10_format(DXGI_FORMAT_B5G5R5A1_UNORM), Some(134));
    }

    #[test]
    fn non_array_resources_ignore_tsharp_depth() {
        let mut t = image(&tsharp()).unwrap();
        t.depth = 19;
        for (ty, depth, want) in [
            (ImageType::D1, 19, 1),
            (ImageType::D2, 19, 1),
            (ImageType::D2Msaa, 19, 1),
            (ImageType::D1Array, 19, 19),
            (ImageType::D2Array, 19, 19),
            (ImageType::D2MsaaArray, 19, 19),
            (ImageType::Cube, 18, 18),
            (ImageType::D3, 19, 19),
        ] {
            t.ty = ty;
            t.depth = depth;
            assert_eq!(image_d3d12(&t).unwrap().0.DepthOrArraySize, want, "{ty:?}");
        }
    }

    #[test]
    fn cube_views_require_a_nonempty_whole_number_of_faces() {
        let mut t = image(&tsharp()).unwrap();
        t.ty = ImageType::Cube;
        t.depth = 12;
        for (base, valid, cubes) in [(0, true, 2), (6, true, 1), (12, false, 0), (13, false, 0), (1, false, 0)] {
            t.base_array = base;
            assert_eq!(valid_cube_view(&t), valid, "base {base}");
            let d = image_d3d12(&t);
            assert_eq!(d.is_some(), valid, "base {base}");
            if let Some((_, srv)) = d {
                assert_eq!(unsafe { srv.Anonymous.TextureCubeArray.NumCubes }, cubes);
            }
        }
        t.depth = 5;
        t.base_array = 0;
        assert!(!valid_cube_view(&t));
        assert!(image_d3d12(&t).is_none());
    }

    #[test]
    fn image_mips_and_array_views_stay_within_resource_dimensions() {
        let mut t = image(&tsharp()).unwrap();
        t.width = 8;
        t.height = 4;
        t.base_level = 0;
        t.last_level = 3;
        t.max_mip = 3;
        assert!(valid_image_shape(&t));
        assert_eq!(image_d3d12(&t).unwrap().0.MipLevels, 4);
        let changes: [fn(&mut Image); 6] = [
            |x: &mut Image| x.width = 0,
            |x: &mut Image| x.height = 0,
            |x: &mut Image| x.max_mip = 32,
            |x: &mut Image| x.max_mip = 4,
            |x: &mut Image| x.last_level = 4,
            |x: &mut Image| x.base_level = 4,
        ];
        for change in changes {
            let mut bad = t;
            change(&mut bad);
            assert!(!valid_image_shape(&bad));
            assert!(image_d3d12(&bad).is_none());
        }
        t.ty = ImageType::D2Array;
        t.depth = 3;
        t.base_array = 2;
        assert!(valid_image_shape(&t));
        t.base_array = 3;
        assert!(!valid_image_shape(&t));
        t.base_array = 0;
        t.depth = 0;
        assert!(!valid_image_shape(&t));
    }

    #[test]
    fn resource_depth_is_rejected_instead_of_silently_clamped() {
        let mut t = image(&tsharp()).unwrap();
        t.depth = u16::MAX as u32 + 1;
        for ty in [ImageType::D1Array, ImageType::D2Array, ImageType::D2MsaaArray, ImageType::Cube, ImageType::D3] {
            t.ty = ty;
            assert!(!valid_image_shape(&t), "{ty:?}");
            assert!(image_d3d12(&t).is_none(), "{ty:?}");
        }
        for ty in [ImageType::D1, ImageType::D2, ImageType::D2Msaa] {
            t.ty = ty;
            assert!(valid_image_shape(&t), "{ty:?}");
            assert_eq!(image_d3d12(&t).unwrap().0.DepthOrArraySize, 1, "{ty:?}");
        }
    }

    #[test]
    fn resource_dimensions_respect_d3d12_texture_limits() {
        let mut t = image(&tsharp()).unwrap();
        t.base_level = 0;
        t.last_level = 0;
        t.max_mip = 0;

        t.ty = ImageType::D1;
        t.width = D3D12_REQ_TEXTURE1D_U_DIMENSION;
        assert!(image_d3d12(&t).is_some());
        t.width += 1;
        assert!(image_d3d12(&t).is_none());

        t.ty = ImageType::D2;
        for axis in 0..2 {
            t.width = 1;
            t.height = 1;
            if axis == 0 {
                t.width = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
            } else {
                t.height = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
            }
            assert!(image_d3d12(&t).is_some(), "2D axis {axis} at the limit");
            if axis == 0 {
                t.width += 1;
            } else {
                t.height += 1;
            }
            assert!(image_d3d12(&t).is_none(), "2D axis {axis} over the limit");
        }

        for ty in [ImageType::D1Array, ImageType::D2Array, ImageType::D2MsaaArray] {
            t.width = 1;
            t.height = 1;
            t.ty = ty;
            t.depth = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
            assert!(image_d3d12(&t).is_some(), "{ty:?} at the array limit");
            t.depth += 1;
            assert!(image_d3d12(&t).is_none(), "{ty:?} over the array limit");
        }

        t.ty = ImageType::Cube;
        t.base_array = 0;
        t.depth = 2046; // largest whole number of faces within the 2048-slice limit
        assert!(image_d3d12(&t).is_some());
        t.depth = 2052;
        assert!(image_d3d12(&t).is_none());

        t.ty = ImageType::D3;
        t.depth = 1;
        for axis in 0..3 {
            t.width = 1;
            t.height = 1;
            t.depth = 1;
            match axis {
                0 => t.width = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                1 => t.height = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                _ => t.depth = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
            }
            assert!(image_d3d12(&t).is_some(), "3D axis {axis} at the limit");
            match axis {
                0 => t.width += 1,
                1 => t.height += 1,
                _ => t.depth += 1,
            }
            assert!(image_d3d12(&t).is_none(), "3D axis {axis} over the limit");
        }
    }

    #[test]
    fn decodes_ssharp() {
        // clamp/wrap/border, aniso 8, compare LESS, bias -1.5, mag/min linear, mip linear, lod 0..13
        let w = [2 | 0 << 3 | 6 << 6 | 3 << 9 | 1 << 12, 0 | (13 * 256) << 12, (((-384i32) as u32) & 0x3FFF) | 1 << 20 | 1 << 22 | 2 << 26, 2 << 30];
        let s = sampler_d3d12(&w, false);
        assert_eq!((s.AddressU, s.AddressV, s.AddressW), (D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_BORDER));
        assert_eq!(s.Filter, D3D12_FILTER_MIN_MAG_MIP_LINEAR);
        assert_eq!((s.MipLODBias, s.MinLOD, s.MaxLOD, s.BorderColor), (-1.5, 0.0, 13.0, [1.0; 4]));
        assert_eq!(s.ComparisonFunc, D3D12_COMPARISON_FUNC_LESS);
        assert_eq!(sampler_d3d12(&w, true).Filter, D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR);
        let mut a = w;
        a[2] = 2 << 20 | 2 << 22 | 2 << 26;
        assert_eq!(sampler_d3d12(&a, false).Filter, D3D12_FILTER_ANISOTROPIC);
    }

    #[test]
    fn decodes_vsharp() {
        let b = buffer(&[0x1000, 0x12 | 16 << 16, 64, 4 | 5 << 3 | 6 << 6 | 7 << 9 | 77 << 12]);
        assert_eq!((b.address, b.stride, b.num_records, b.format, b.dst_sel), (0x12_0000_1000, 16, 64, 77, [4, 5, 6, 7]));
    }
}

/// GFX10 IMG_FORMAT of a DXGI format (the inverse of `dxgi_format`; typeless → None).
pub fn gfx10_format(f: DXGI_FORMAT) -> Option<u32> {
    (0..=182).find(|&g| dxgi_format(g).map(|(d, _)| d) == Some(f))
}

/// A T# in the XCL shader ABI (see `xcl_image_words`) for a D3D12 view: SRV at
/// `srv_heap_index`, UAV at + 1. Size, format, type and mips as the console fields, so
/// shaders that read them (image class, resinfo) see the real values.
pub fn xcl_image_for_view(srv_heap_index: u32, format: u32, ty: ImageType, width: u32, height: u32, depth: u32, mips: u32) -> [u32; 8] {
    let (w, h, d) = (width.max(1) - 1, height.max(1) - 1, depth.max(1) - 1);
    let ty = match ty {
        ImageType::D1 => 8,
        ImageType::D2 => 9,
        ImageType::D3 => 10,
        ImageType::Cube => 11,
        ImageType::D1Array => 12,
        ImageType::D2Array => 13,
        ImageType::D2Msaa => 14,
        ImageType::D2MsaaArray => 15,
    };
    let last = mips.max(1) - 1;
    let mut t = [0u32; 8];
    t[0] = srv_heap_index;
    t[1] = (format & 0x1FF) << 20 | (w & 3) << 30;
    t[2] = (w >> 2) & 0xFFF | (h & 0x3FFF) << 14;
    t[3] = 4 | 5 << 3 | 6 << 6 | 7 << 9 | (last & 0xF) << 16 | ty << 28;
    t[4] = d & 0x1FFF;
    t[5] = (last & 0xF) << 4;
    t
}

#[cfg(test)]
mod view_tests {
    use super::*;

    #[test]
    fn view_descriptors_decode_back() {
        assert_eq!(gfx10_format(DXGI_FORMAT_R16G16B16A16_FLOAT), Some(71));
        assert_eq!(gfx10_format(DXGI_FORMAT_R32_FLOAT), Some(22));
        assert_eq!(gfx10_format(DXGI_FORMAT_R16G16_TYPELESS), None);
        let t = xcl_image_for_view(14, 71, ImageType::D2, 1920, 1080, 1, 3);
        let i = image(&t).unwrap();
        assert_eq!((i.format, i.width, i.height, i.depth, i.ty, i.last_level, i.max_mip), (71, 1920, 1080, 1, ImageType::D2, 2, 2));
        assert_eq!(i.dst_sel, [4, 5, 6, 7]);
        assert_eq!(t[0], 14);
        let v = buffer(&buffer_words(0x5_1234_5670, 16, 100, 77));
        assert_eq!((v.address, v.stride, v.num_records, v.format, v.dst_sel), (0x5_1234_5670, 16, 100, 77, [4, 5, 6, 7]));
        assert_eq!((format_bytes(22), format_bytes(29), format_bytes(71), format_bytes(77)), (4, 4, 8, 16));
        let b = bvh_words(0x12_3456_789A_00, true);
        assert_eq!(b, [0x3456_789A, 0x8000_0012, u32::MAX, 0]);
        assert!(b[3] >> 28 < 8, "BVH descriptor must not look like a T# image to CopyDescriptors");
        assert_eq!(bvh_words(0x12_3456_789A_00, false)[1], 0x12);
    }
}

/// Bytes per element of a GFX10 buffer / image format (0–77: 8, 16, 32, 64, 96, 128 bits).
pub fn format_bytes(f: u32) -> u32 {
    match f {
        1..=6 => 1,
        7..=19 => 2,
        20..=61 => 4,
        62..=71 => 8,
        72..=74 => 12,
        75..=77 => 16,
        _ => 4,
    }
}

/// A V# for a constant buffer view: 16-byte elements (one float4 row each), num_records in
/// rows, as the console writes it (a title that writes this descriptor itself writes stride 16
/// and size / 16). Console shaders index constant-buffer arrays with `buffer_load_* … idxen`
/// and the row as the index, which a stride of 0 ignores (every row would read row 0). Loads by
/// byte offset read the same addresses as with a raw V#, and scalar loads keep their byte limit
/// (num_records × stride).
pub fn cbv_words(address: u64, size: u32) -> [u32; 4] {
    buffer_words(address, 16, size.div_ceil(16), 77)
}

/// A V# for a buffer view: `stride` 0 = raw (num_records in bytes), else structured / typed
/// (num_records in elements); `format` = GFX10 format for typed loads (raw: 32_FLOAT).
pub fn buffer_words(address: u64, stride: u32, num_records: u32, format: u32) -> [u32; 4] {
    let oob = if stride == 0 { 3 } else { 0 };
    [address as u32, (address >> 32) as u32 & 0xFFFF | (stride & 0x3FFF) << 16, num_records, 0xFAC | (format & 0x7F) << 12 | 1 << 24 | oob << 28]
}

/// An RDNA2 BVH resource descriptor (SRSRC, 4 dwords) for `image_bvh(64)_intersect_ray`:
/// word 0 = base address bits 39:8 (256-byte units), word 1 = bits 47:40 | box_sort << 31,
/// word 2 = size / max bounds, word 3 = 0 (type bits 31:28 < 8 so descriptor copies leave it
/// verbatim instead of treating it as a T# image).
pub fn bvh_words(address: u64, box_sort: bool) -> [u32; 4] {
    [(address >> 8) as u32, ((address >> 40) as u32 & 0xFF) | (u32::from(box_sort) << 31), u32::MAX, 0]
}
