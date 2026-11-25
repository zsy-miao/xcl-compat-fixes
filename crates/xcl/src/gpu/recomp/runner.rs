//! Runs recompiled kernels on the real GPU (D3D12 compute): the test bench for
//! instruction semantics and for whole console shaders.
//!
//! Kernel ABI (see `recomp`): u0 `io` = initial SGPRs (128 dwords, SCC in slot 125) then
//! 64 lanes × 256 VGPRs; u1 `res` = the same layout after the program plus exports;
//! u2 `mem` = flat memory for scalar/buffer loads.

use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use super::{KERNEL_LANES, KERNEL_RES_DWORDS, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS};
use crate::gpu::fence;

/// Initial state of a kernel run.
#[derive(Clone)]
pub struct KernelState {
    pub sgpr: Vec<u32>,
    /// `vgpr[lane][reg]`
    pub vgpr: Vec<Vec<u32>>,
    pub scc: u32,
}

impl Default for KernelState {
    fn default() -> Self {
        let mut sgpr = vec![0u32; KERNEL_SGPRS];
        sgpr[126] = u32::MAX; // EXEC: all 64 lanes
        sgpr[127] = u32::MAX;
        KernelState { sgpr, vgpr: vec![vec![0u32; KERNEL_VGPRS]; KERNEL_LANES], scc: 0 }
    }
}

/// Final state + exports (`exports[target][lane][component]`, raw dwords).
pub struct KernelResult {
    pub sgpr: Vec<u32>,
    pub vgpr: Vec<Vec<u32>>,
    pub scc: u32,
    pub raw: Vec<u32>,
    /// flat memory after the kernel (stores)
    pub mem: Vec<u32>,
    /// contents of the UAV test textures after the kernel (tightly packed)
    pub textures: Vec<Vec<u8>>,
}

impl KernelResult {
    pub fn export(&self, target: usize, lane: usize, comp: usize) -> u32 {
        self.raw[super::KERNEL_EXPORTS + (target * KERNEL_LANES + lane) * 4 + comp]
    }
}

struct Gpu {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    rs: ID3D12RootSignature,
    fence: ID3D12Fence,
    value: u64,
    adapter: String,
    wave64: bool,
    /// 32-lane waves (`[WaveSize(32)]`: the split wave64 form; WARP has neither 32 nor 64)
    wave32: bool,
    /// mesh shader tier 1 or higher (`mesh_draw`)
    mesh: bool,
}

// D3D12 objects are free-threaded; access is serialised by the mutex.
unsafe impl Send for Gpu {}

static GPU: Mutex<Option<Result<Gpu, String>>> = Mutex::new(None);

fn e(x: windows::core::Error) -> String {
    format!("{x}")
}

unsafe fn wait_for_submission(
    dev: &ID3D12Device, queue: &ID3D12CommandQueue, fence: &ID3D12Fence, value: u64, stage: &str,
) -> Result<(), String> {
    queue.Signal(fence, value).map_err(|e| format!("{stage}: {}", fence::api_error(dev, "Signal", e)))?;
    fence::wait(fence, value, dev, None, 10_000).map_err(|e| format!("{stage}: {e}"))
}

fn init() -> Result<Gpu, String> {
    unsafe {
        if crate::state::config().d3d12_debug || std::env::var_os("XCL_D3D12_DEBUG").is_some() {
            // The debug layer must be enabled before this runner creates a device.
            let _ = crate::gpu::agility::enable_debug_layer();
        }
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(e)?;
        // `XCL_ADAPTER` picks the adapter (default: the first in high-performance order)
        let sel = crate::gpu::adapter::Selector::from_env_or("auto");
        let (adapter, name): (IDXGIAdapter1, String) = if sel == crate::gpu::adapter::Selector::Warp {
            (factory.EnumWarpAdapter().map_err(e)?, "WARP (software renderer)".into())
        } else {
            let (a, desc) = crate::gpu::adapter::candidates(&factory, &sel)
                .into_iter()
                .next()
                .ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
            (a, crate::gpu::adapter::name(&desc))
        };
        let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).map_err(e)?;
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
        dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32)
            .map_err(|x| format!("SM 6.6: {x}"))?;
        if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            return Err(crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state()));
        }
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mut o1 as *mut _ as _, std::mem::size_of_val(&o1) as u32).map_err(e)?;
        if !o1.WaveOps.as_bool() || !o1.Int64ShaderOps.as_bool() {
            return Err(format!("{name}: no wave operations/int64"));
        }
        let wave64 = o1.WaveLaneCountMin <= 64 && o1.WaveLaneCountMax >= 64;
        let wave32 = o1.WaveLaneCountMin <= 32 && o1.WaveLaneCountMax >= 32;
        let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let mesh = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32).is_ok()
            && o7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
        let queue: ID3D12CommandQueue =
            dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        let rs = root_signature(&dev, false)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        Ok(Gpu { dev, queue, rs, fence, value: 0, adapter: name, wave64, wave32, mesh })
    }
}

/// Root signature: u0 io, u1 res, u2 mem as root UAVs; CBV/SRV/UAV and sampler heaps
/// directly indexed (SM 6.6 `ResourceDescriptorHeap`); `graphics` adds the input assembler.
fn root_signature(dev: &ID3D12Device, graphics: bool) -> Result<ID3D12RootSignature, String> {
    super::pso::abi_root_signature(dev, graphics)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexDim {
    D1Array,
    D2,
    D2Array,
    D3,
    Cube,
    /// Typed buffer (test only): `width` elements of `format`, `height` = `depth` = 1.
    Buffer,
    /// Multisampled 2D texture with this many samples (SRV only). `data` holds every
    /// sample: texel (x, y) sample s at `((y · width + x) · samples + s) · texel_bytes`;
    /// `format` needs 32-bit channels (the samples are written by a per-sample draw).
    D2Ms(u32),
    /// Multisampled 2D array (`depth` slices, laid out as `D2Ms` slice after slice).
    D2MsArray(u32),
}

impl TexDim {
    fn samples(self) -> Option<u32> {
        match self {
            TexDim::D2Ms(n) | TexDim::D2MsArray(n) => Some(n),
            _ => None,
        }
    }
}

/// A texture for kernel tests: SRV at heap index 2·i, UAV (if `uav`) at 2·i + 1.
pub struct TestTexture {
    pub dim: TexDim,
    pub format: DXGI_FORMAT,
    pub width: u32,
    pub height: u32,
    /// array slices (cube: 6 per cube) or depth
    pub depth: u32,
    pub texel_bytes: u32,
    /// tightly packed texels, slice after slice
    pub data: Vec<u8>,
    pub uav: bool,
}

/// A sampler for kernel tests at sampler heap index j.
pub struct TestSampler {
    pub filter: D3D12_FILTER,
    pub address: D3D12_TEXTURE_ADDRESS_MODE,
    pub comparison: Option<D3D12_COMPARISON_FUNC>,
}

struct Bound {
    _heaps: [ID3D12DescriptorHeap; 2],
    /// what the multisampled-texture fill draws use (alive until the list ran)
    _msaa: Vec<MsaaFill>,
    textures: Vec<(ID3D12Resource, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, u64)>,
    _upload: Option<ID3D12Resource>,
}

/// Create the descriptor heaps (always, so the directly indexed root signature is valid),
/// the test textures and samplers; record the uploads into `list`.
unsafe fn bind_resources(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, textures: &[TestTexture], samplers: &[TestSampler]) -> Result<Bound, String> {
    let n = (2 * textures.len()).max(2) as u32;
    let heap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: n,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
        .map_err(e)?;
    let sheap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: samplers.len().max(1) as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
        .map_err(e)?;
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
    let sinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
    let at = |h: &ID3D12DescriptorHeap, k: usize, step: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + k * step };
    for (j, sm) in samplers.iter().enumerate() {
        let mut sd = D3D12_SAMPLER_DESC {
            Filter: sm.filter,
            AddressU: sm.address,
            AddressV: sm.address,
            AddressW: sm.address,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        if let Some(c) = sm.comparison {
            sd.ComparisonFunc = c;
        }
        dev.CreateSampler(&sd, at(&sheap, j, sinc));
    }
    // all texture data through one upload buffer
    let mut descs = Vec::new();
    let mut total = 0u64;
    let mut plans = Vec::new();
    let mut msaa_plans = Vec::new();
    for t in textures {
        if let Some(n) = t.dim.samples() {
            // the samples go through a raw buffer read by the fill draw (+16: Load4 at the end)
            let size = t.data.len() as u64 + 16;
            msaa_plans.push(total);
            total = (total + size + 511) & !511;
            descs.push(D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: t.width as u64,
                Height: t.height,
                DepthOrArraySize: t.depth as u16,
                MipLevels: 1,
                Format: t.format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: n, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            });
            plans.push(Vec::new());
            continue;
        }
        let (dim, depth_or_array) = match t.dim {
            TexDim::D3 => (D3D12_RESOURCE_DIMENSION_TEXTURE3D, t.depth),
            TexDim::D1Array => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, t.depth),
            _ => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, t.depth),
        };
        let rd = if t.dim == TexDim::Buffer {
            D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: (t.width * t.texel_bytes) as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: if t.uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
                ..Default::default()
            }
        } else { D3D12_RESOURCE_DESC {
            Dimension: dim,
            Width: t.width as u64,
            Height: t.height,
            DepthOrArraySize: depth_or_array as u16,
            MipLevels: 1,
            Format: t.format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: if t.uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
            ..Default::default()
        } };
        let subs = if t.dim == TexDim::D3 { 1 } else { t.depth };
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
        let mut size = 0u64;
        dev.GetCopyableFootprints(&rd, 0, subs, total, Some(fps.as_mut_ptr()), None, None, Some(&mut size));
        total = (total + size + 511) & !511;
        plans.push(fps);
        descs.push(rd);
    }
    let upload = if total > 0 { Some(buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?) } else { None };
    let mut out = Vec::new();
    if let Some(up) = &upload {
        let mut p = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).map_err(e)?;
        let mut mk = 0;
        for t in textures.iter().filter(|t| t.dim.samples().is_some()) {
            std::ptr::copy_nonoverlapping(t.data.as_ptr(), (p as *mut u8).add(msaa_plans[mk] as usize), t.data.len());
            mk += 1;
        }
        for (t, fps) in textures.iter().zip(&plans).filter(|(t, _)| t.dim.samples().is_none()) {
            let row = (t.width * t.texel_bytes) as usize;
            let slices = t.depth as usize;
            for z in 0..slices {
                let fp = if t.dim == TexDim::D3 { &fps[0] } else { &fps[z] };
                let slice_pitch = fp.Footprint.RowPitch as usize * t.height as usize;
                let zoff = if t.dim == TexDim::D3 { z * slice_pitch } else { 0 };
                for y in 0..t.height as usize {
                    let src = &t.data[(z * t.height as usize + y) * row..][..row];
                    let dst = (p as *mut u8).add(fp.Offset as usize + zoff + y * fp.Footprint.RowPitch as usize);
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, row);
                }
            }
        }
        up.Unmap(0, None);
    }
    let mut msaa = Vec::new();
    let mut mk = 0;
    for (i, ((t, rd), fps)) in textures.iter().zip(&descs).zip(&plans).enumerate() {
        if t.dim.samples().is_some() {
            assert!(!t.uav, "multisampled test textures are SRVs");
            let f = msaa_fill(dev, list, t, rd, upload.as_ref().unwrap(), msaa_plans[mk])?;
            mk += 1;
            let mut sv = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: t.format, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
            if let TexDim::D2MsArray(_) = t.dim {
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                sv.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 0, ArraySize: t.depth };
            } else {
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
            }
            dev.CreateShaderResourceView(&f.tex, Some(&sv), at(&heap, 2 * i, inc));
            out.push((f.tex.clone(), Vec::new(), 0));
            msaa.push(f);
            continue;
        }
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
            .map_err(e)?;
        let r = r.ok_or("texture")?;
        if t.dim == TexDim::Buffer {
            list.CopyBufferRegion(&r, 0, upload.as_ref().unwrap(), fps[0].Offset, rd.Width);
        }
        for (k, fp) in fps.iter().enumerate().filter(|_| t.dim != TexDim::Buffer) {
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: transmute_copy(upload.as_ref().unwrap()),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        let after = if t.uav { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE };
        list.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_STATE_COPY_DEST, after)]);
        let mut sv = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: t.format, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
        let mut uv = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: t.format, ..Default::default() };
        match t.dim {
            TexDim::D1Array => {
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                sv.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { MipLevels: 1, ArraySize: t.depth, ..Default::default() };
                uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                uv.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV { ArraySize: t.depth, ..Default::default() };
            }
            TexDim::D2 => {
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                sv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
                uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            }
            TexDim::D2Array => {
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                sv.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: t.depth, ..Default::default() };
                uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                uv.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { ArraySize: t.depth, ..Default::default() };
            }
            TexDim::D3 => {
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                sv.Anonymous.Texture3D = D3D12_TEX3D_SRV { MipLevels: 1, ..Default::default() };
                uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                uv.Anonymous.Texture3D = D3D12_TEX3D_UAV { WSize: t.depth, ..Default::default() };
            }
            TexDim::Cube => {
                // a cube array, as translated shaders declare it (depth = 6 per cube)
                sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                sv.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { MipLevels: 1, NumCubes: t.depth / 6, ..Default::default() };
                uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                uv.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { ArraySize: t.depth, ..Default::default() };
            }
            TexDim::D2Ms(_) | TexDim::D2MsArray(_) => unreachable!(),
            TexDim::Buffer => {
                sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                sv.Anonymous.Buffer = D3D12_BUFFER_SRV { NumElements: t.width, ..Default::default() };
                uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uv.Anonymous.Buffer = D3D12_BUFFER_UAV { NumElements: t.width, ..Default::default() };
            }
        }
        // a UAV texture has its SRV too, as D3D12.X gives a console image UAV a companion SRV
        // (RDNA images have no SRV / UAV distinction: the shader may read the slot)
        dev.CreateShaderResourceView(&r, Some(&sv), at(&heap, 2 * i, inc));
        if t.uav {
            dev.CreateUnorderedAccessView(&r, None::<&ID3D12Resource>, Some(&uv), at(&heap, 2 * i + 1, inc));
        }
        let mut readable = 0u64;
        let subs = fps.len() as u32;
        let mut rfps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
        dev.GetCopyableFootprints(rd, 0, subs, 0, Some(rfps.as_mut_ptr()), None, None, Some(&mut readable));
        out.push((r, rfps, readable));
    }
    list.SetDescriptorHeaps(&[Some(heap.clone()), Some(sheap.clone())]);
    Ok(Bound { _heaps: [heap, sheap], _msaa: msaa, textures: out, _upload: upload })
}

struct MsaaFill {
    tex: ID3D12Resource,
    _data: ID3D12Resource,
    _pso: ID3D12PipelineState,
    _rs: ID3D12RootSignature,
    _rtv: ID3D12DescriptorHeap,
}

/// Write every sample of a multisampled test texture: a full-screen draw per slice whose
/// pixel shader runs per sample (`SV_SampleIndex`) and returns the sample's bits from u0.
/// Leaves `list` with another pipeline and root signature bound.
unsafe fn msaa_fill(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, t: &TestTexture, rd: &D3D12_RESOURCE_DESC, upload: &ID3D12Resource, at: u64) -> Result<MsaaFill, String> {
    let n = t.dim.samples().unwrap();
    assert!(t.texel_bytes % 4 == 0 && t.texel_bytes <= 16, "multisampled test textures need 32-bit channels");
    let ty = match t.format {
        DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R32G32B32A32_FLOAT => "float4",
        DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32B32A32_UINT => "uint4",
        DXGI_FORMAT_R32_SINT | DXGI_FORMAT_R32G32_SINT | DXGI_FORMAT_R32G32B32A32_SINT => "int4",
        _ => return Err(format!("multisampled test texture format {:?}", t.format)),
    };
    // u0: the samples of this slice
    let src = format!(
        "RWByteAddressBuffer D : register(u0);\n\
         float4 vs(uint id : SV_VertexID) : SV_Position {{ float2 p = float2((id << 1) & 2, id & 2); return float4(p * float2(2, -2) + float2(-1, 1), 0, 1); }}\n\
         {ty} ps(float4 pos : SV_Position, uint s : SV_SampleIndex) : SV_Target {{\n\
           uint2 xy = uint2(pos.xy);\n\
           uint i = ((xy.y * {w}u + xy.x) * {n}u + s) * {b}u;\n\
           uint4 v = D.Load4(i);\n\
           uint k = {b}u / 4u; v = uint4(v.x, k > 1u ? v.y : 0u, k > 2u ? v.z : 0u, k > 3u ? v.w : 0u);\n\
           return ({ty})({conv}(v));\n\
         }}\n",
        w = t.width,
        b = t.texel_bytes,
        conv = if ty == "float4" { "asfloat" } else { "" },
    );
    let vs = super::dxc::compile(&src, "vs", "vs_6_6", &[])?;
    let ps = super::dxc::compile(&src, "ps", "ps_6_6", &[])?;
    let rs = root_signature(dev, false)?;
    let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: transmute_copy(&rs),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
        SampleMask: u32::MAX,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: n, Quality: 0 },
        ..Default::default()
    };
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = true.into();
    pd.RTVFormats[0] = t.format;
    let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&pd).map_err(|x| format!("CreateGraphicsPipelineState (MSAA fill): {x}"))?;
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, rd, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut r)
        .map_err(|x| format!("MSAA texture: {x}"))?;
    let tex = r.ok_or("texture")?;
    let size = t.data.len() as u64 + 16;
    let data = buffer(dev, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
    list.CopyBufferRegion(&data, 0, upload, at, size);
    list.ResourceBarrier(&[transition(&data, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    let rtv_heap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: t.depth.max(1), ..Default::default() })
        .map_err(e)?;
    let rinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
    list.SetPipelineState(&pso);
    list.SetGraphicsRootSignature(&rs);
    list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: t.width as f32, Height: t.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
    list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: t.width as i32, bottom: t.height as i32 }]);
    let slice_bytes = (t.width * t.height * n * t.texel_bytes) as u64;
    for z in 0..t.depth.max(1) {
        let h = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr + z as usize * rinc };
        let mut rv = D3D12_RENDER_TARGET_VIEW_DESC { Format: t.format, ..Default::default() };
        if let TexDim::D2MsArray(_) = t.dim {
            rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            rv.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_RTV { FirstArraySlice: z, ArraySize: 1 };
        } else {
            rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
        }
        dev.CreateRenderTargetView(&tex, Some(&rv), h);
        list.OMSetRenderTargets(1, Some(&h), false, None);
        list.SetGraphicsRootUnorderedAccessView(0, data.GetGPUVirtualAddress() + z as u64 * slice_bytes);
        list.DrawInstanced(3, 1, 0, 0);
    }
    list.ResourceBarrier(&[transition(&tex, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)]);
    Ok(MsaaFill { tex, _data: data, _pso: pso, _rs: rs, _rtv: rtv_heap })
}

fn buffer(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    unsafe {
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
            .map_err(e)?;
    }
    r.ok_or_else(|| "buffer".into())
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Name of the GPU the kernels run on (None: no usable GPU).
pub fn adapter() -> Result<String, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init);
    gpu.as_ref().map(|x| x.adapter.clone()).map_err(|x| x.clone())
}

pub fn wave64_supported() -> Result<bool, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init);
    gpu.as_ref().map(|x| x.wave64).map_err(|x| x.clone())
}

/// Whether the adapter runs 32-lane waves (`[WaveSize(32)]`).
pub fn wave32_supported() -> Result<bool, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init);
    gpu.as_ref().map(|x| x.wave32).map_err(|x| x.clone())
}

/// Whether wave64 programs need the split form here (`Options::split_wave64`): the
/// adapter runs only 32-lane waves. False without a usable GPU.
pub fn split_wave64() -> bool {
    wave64_supported() == Ok(false) || (super::pso::split_wave64_forced() && wave64_programs_supported() == Ok(true))
}

/// Whether wave64 programs run on this adapter at all: natively (64-lane waves) or split (32).
pub fn wave64_programs_supported() -> Result<bool, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init);
    gpu.as_ref().map(|x| x.wave64 || x.wave32).map_err(|x| x.clone())
}

/// Whether a wave64 VS / PS may run in 32-lane waves here (`pso::graphics_wave32`): the
/// adapter runs 32-lane waves at all, or the split form is forced. False without a usable GPU.
pub fn graphics_wave32() -> bool {
    if split_wave64() {
        return true;
    }
    let mut g = GPU.lock().unwrap();
    g.get_or_insert_with(init).as_ref().is_ok_and(|x| x.wave32)
}

/// The split flag for a program of `mode` on this adapter: `graphics_wave32` for a VS / PS,
/// `split_wave64` for the rest.
pub fn split_for(mode: &super::Mode) -> bool {
    if matches!(mode, super::Mode::Vertex(_) | super::Mode::Pixel(_)) {
        graphics_wave32()
    } else {
        split_wave64()
    }
}

/// `Options` for programs run on this adapter: the split wave64 form where it is needed.
pub fn options(mode: super::Mode) -> super::Options {
    super::Options { split_wave64: split_for(&mode), mode, ..Default::default() }
}

/// A split wave64 kernel stores the upper half's SGPRs too: both halves must end with the
/// same scalar state (they would otherwise have branched differently).
fn split_halves_agree(raw: &[u32]) -> Result<(), String> {
    let at = super::KERNEL_SPLIT_SGPRS;
    if raw[at + KERNEL_SGPRS] != super::KERNEL_SPLIT_MAGIC {
        return Ok(());
    }
    let differ: Vec<String> = (0..KERNEL_SGPRS).filter(|&i| raw[i] != raw[at + i]).map(|i| format!("s{i}: {:#x} / {:#x}", raw[i], raw[at + i])).collect();
    if differ.is_empty() {
        Ok(())
    } else {
        Err(format!("split wave64: the halves ended with different SGPRs (lower / upper): {}", differ.join(", ")))
    }
}

/// Whether the adapter runs mesh shaders (`mesh_draw`); an integrated RDNA2 GPU, for one,
/// reports wave64 but no mesh shader tier.
pub fn mesh_shaders_supported() -> Result<bool, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init);
    gpu.as_ref().map(|x| x.mesh).map_err(|x| x.clone())
}

/// A GPU test (or a part of one) that cannot run here, `reason` one of "no GPU", "no DXC",
/// "no wave64", "no mesh shaders": `SKIPPED <test>: <reason>` written to stderr itself, so the
/// line shows without `--nocapture` (libtest captures only `print!` / `eprint!`), and
/// `scripts/test.ps1` counts them. `XCL_REQUIRE_GPU=wave64` or `=mesh` (or both, comma-separated)
/// turns a skip that leaves that coverage out into a failure.
pub fn skip_test(reason: &str) {
    use std::io::Write;
    let thread = std::thread::current();
    let test = thread.name().unwrap_or("?");
    let required = std::env::var("XCL_REQUIRE_GPU").unwrap_or_default();
    let fails = required.split(',').map(str::trim).any(|r| match r {
        "wave64" => reason != "no mesh shaders",
        "mesh" => reason != "no wave64",
        _ => false,
    });
    if fails {
        panic!("{test}: {reason}, but XCL_REQUIRE_GPU={required}");
    }
    let _ = writeln!(std::io::stderr(), "SKIPPED {test}: {reason}");
}

/// Run a compiled kernel (DXIL) with `state` and flat memory `mem`.
pub fn run(dxil: &[u8], state: &KernelState, mem: &[u32]) -> Result<KernelResult, String> {
    run_with(dxil, state, mem, &[], &[])
}

/// `run` with test textures (SRV at heap index 2·i, UAV at 2·i + 1) and samplers.
pub fn run_with(dxil: &[u8], state: &KernelState, mem: &[u32], textures: &[TestTexture], samplers: &[TestSampler]) -> Result<KernelResult, String> {
    run_with_statics(dxil, state, mem, textures, samplers, &[])
}

/// `run_with` for a kernel with static samplers (`Translation::static_samplers`).
pub fn run_with_statics(
    dxil: &[u8],
    state: &KernelState,
    mem: &[u32],
    textures: &[TestTexture],
    samplers: &[TestSampler],
    statics: &[super::pso::StaticSampler],
) -> Result<KernelResult, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    let rs: ID3D12RootSignature = if statics.is_empty() { gpu.rs.clone() } else { super::pso::abi_root_signature_with(&gpu.dev, false, statics)? };
    let mut io: Vec<u32> = state.sgpr.clone();
    io.resize(KERNEL_SGPRS, 0);
    io[KERNEL_SCC_SLOT] = state.scc;
    for lane in 0..KERNEL_LANES {
        let mut regs = state.vgpr.get(lane).cloned().unwrap_or_default();
        regs.resize(KERNEL_VGPRS, 0);
        io.extend_from_slice(&regs);
    }
    let mem_len = mem.len().max(64);
    unsafe {
        let dev = &gpu.dev;
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .map_err(|x| format!("CreateComputePipelineState: {x}"))?;
        let io_bytes = (io.len() * 4) as u64;
        let res_bytes = (KERNEL_RES_DWORDS * 4) as u64;
        let mem_bytes = (mem_len * 4) as u64;
        let upload = buffer(dev, io_bytes + mem_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
        let io_buf = buffer(dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let mem_buf = buffer(dev, mem_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let res_buf = buffer(dev, res_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let readback = buffer(dev, res_bytes + mem_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).map_err(e)?;
        std::ptr::copy_nonoverlapping(io.as_ptr(), p as *mut u32, io.len());
        let mut m = mem.to_vec();
        m.resize(mem_len, 0);
        std::ptr::copy_nonoverlapping(m.as_ptr(), (p as *mut u32).add(io.len()), m.len());
        upload.Unmap(0, None);

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(e)?;
        list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_bytes);
        list.CopyBufferRegion(&mem_buf, 0, &upload, io_bytes, mem_bytes);
        list.ResourceBarrier(&[
            transition(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&mem_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        let bound = bind_resources(dev, &list, textures, samplers)?;
        // multisampled test textures were filled by draws
        list.SetPipelineState(&pso);
        list.SetComputeRootSignature(&rs);
        list.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, mem_buf.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[
            transition(&res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(&mem_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        list.CopyBufferRegion(&readback, 0, &res_buf, 0, res_bytes);
        list.CopyBufferRegion(&readback, res_bytes, &mem_buf, 0, mem_bytes);
        // UAV textures back for checks
        let mut tex_back = Vec::new();
        for (i, t) in textures.iter().enumerate() {
            if !t.uav {
                continue;
            }
            let (r, fps, size) = &bound.textures[i];
            let rb = buffer(dev, *size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
            list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            if t.dim == TexDim::Buffer {
                list.CopyBufferRegion(&rb, fps[0].Offset, r, 0, (t.width * t.texel_bytes) as u64);
            }
            for (k, fp) in fps.iter().enumerate().filter(|_| t.dim != TexDim::Buffer) {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            tex_back.push((i, rb));
        }
        list.Close().map_err(e)?;
        gpu.queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        gpu.value += 1;
        wait_for_submission(dev, &gpu.queue, &gpu.fence, gpu.value, "kernel dispatch")?;
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).map_err(e)?;
        let raw = std::slice::from_raw_parts(p as *const u32, KERNEL_RES_DWORDS).to_vec();
        let mem_out = std::slice::from_raw_parts((p as *const u32).add(KERNEL_RES_DWORDS), mem_len).to_vec();
        readback.Unmap(0, None);
        let mut tex_out = Vec::new();
        for (i, rb) in &tex_back {
            let t = &textures[*i];
            let (_, fps, _) = &bound.textures[*i];
            let mut q = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut q)).map_err(e)?;
            let row = (t.width * t.texel_bytes) as usize;
            let mut v = Vec::new();
            for z in 0..t.depth as usize {
                let fp = if t.dim == TexDim::D3 { &fps[0] } else { &fps[z] };
                let zoff = if t.dim == TexDim::D3 { z * fp.Footprint.RowPitch as usize * t.height as usize } else { 0 };
                for y in 0..t.height as usize {
                    let s = std::slice::from_raw_parts((q as *const u8).add(fp.Offset as usize + zoff + y * fp.Footprint.RowPitch as usize), row);
                    v.extend_from_slice(s);
                }
            }
            rb.Unmap(0, None);
            tex_out.push(v);
        }
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).map_err(e)?;
        readback.Unmap(0, None);
        let vgpr = (0..KERNEL_LANES)
            .map(|l| raw[KERNEL_SGPRS + l * KERNEL_VGPRS..KERNEL_SGPRS + (l + 1) * KERNEL_VGPRS].to_vec())
            .collect();
        drop(bound);
        split_halves_agree(&raw)?;
        Ok(KernelResult { sgpr: raw[..KERNEL_SGPRS].to_vec(), vgpr, scc: raw[KERNEL_SCC_SLOT], raw, mem: mem_out, textures: tex_out })
    }
}

/// Run a whole console program: its code region is also placed in `mem` at address 0,
/// so constants addressed through s_getpc_b64 are found; `extra` follows the code
/// (dword-aligned, at the returned offset).
pub fn run_program(code: &[u8], state: &KernelState, extra: &[u32]) -> Result<(KernelResult, usize), String> {
    let mut mem: Vec<u32> = code.chunks(4).map(|c| {
        let mut b = [0u8; 4];
        b[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(b)
    }).collect();
    let at = mem.len() * 4;
    mem.extend_from_slice(extra);
    Ok((run_code(code, state, &mem)?, at))
}

/// `XCL_DIRECT_DXIL=1`: kernels are translated by the direct DXIL backend (`gpu::dxil::kernel`)
/// instead of HLSL + DXC; an unsupported instruction is an error (no fallback).
pub fn direct_dxil() -> bool {
    crate::gpu::dxil::kernel::enabled()
}

/// A kernel through the direct DXIL backend: a wave64 program in the split form where this
/// adapter needs it, as `options` does for the HLSL path.
pub fn direct_kernel(code: &[u8], wave64: bool) -> Result<Vec<u8>, String> {
    direct_kernel_full(code, wave64).map(|k| k.dxil)
}

/// `direct_kernel` with the static samplers the kernel's root signature needs.
pub fn direct_kernel_full(code: &[u8], wave64: bool) -> Result<crate::gpu::dxil::kernel::Kernel, String> {
    crate::gpu::dxil::kernel::translate_full(code, crate::gpu::dxil::kernel::KernelOptions { wave64, split: wave64 && split_wave64(), mem_windows: 1, sampler_base: 0 })
}

/// Translate + compile + run RDNA2 code as a kernel, with test textures and samplers.
pub fn run_code_with(code: &[u8], state: &KernelState, mem: &[u32], textures: &[TestTexture], samplers: &[TestSampler]) -> Result<KernelResult, String> {
    if direct_dxil() {
        let k = direct_kernel_full(code, true)?;
        return run_with_statics(&k.dxil, state, mem, textures, samplers, &k.static_samplers);
    }
    let t = super::translate(code, &options(super::Mode::Kernel))?;
    if !t.complete() {
        return Err(format!("unsupported instructions: {:?}", t.unsupported));
    }
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|x| format!("{x}\n--- HLSL ---\n{}", t.hlsl))?;
    run_with_statics(&dxil, state, mem, textures, samplers, &t.static_samplers)
}

/// Translate + compile + run RDNA2 code as a kernel.
pub fn run_code(code: &[u8], state: &KernelState, mem: &[u32]) -> Result<KernelResult, String> {
    if direct_dxil() {
        let k = direct_kernel_full(code, true)?;
        return run_with_statics(&k.dxil, state, mem, &[], &[], &k.static_samplers);
    }
    let opt = options(super::Mode::Kernel);
    let t = super::translate(code, &opt)?;
    if !t.complete() {
        return Err(format!("unsupported instructions: {:?}", t.unsupported));
    }
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|x| format!("{x}\n--- HLSL ---\n{}", t.hlsl))?;
    run(&dxil, state, mem)
}

/// One vertex input element: (semantic name, index, format, byte offset).
pub struct VertexElement {
    pub semantic: &'static str,
    pub index: u32,
    pub format: DXGI_FORMAT,
    pub offset: u32,
}

/// Draw the vertices (triangle list) with recompiled VS/PS (DXIL) into a `size`×`size`
/// R32G32B32A32_FLOAT target cleared to `clear`; returns the pixels row by row.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    vs: &[u8],
    ps: &[u8],
    layout: &[VertexElement],
    vertices: &[f32],
    stride: u32,
    sgprs: &[u32],
    mem: &[u32],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    draw_gs(vs, None, ps, layout, vertices, stride, sgprs, mem, size, clear)
}

/// `draw` with a geometry shader (DXIL) between the VS and the PS.
#[allow(clippy::too_many_arguments)]
pub fn draw_gs(
    vs: &[u8],
    gs: Option<&[u8]>,
    ps: &[u8],
    layout: &[VertexElement],
    vertices: &[f32],
    stride: u32,
    sgprs: &[u32],
    mem: &[u32],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    draw_gs_views(vs, gs, ps, layout, vertices, stride, &[], sgprs, mem, &[], &[], size, clear)
}

/// `draw_gs` of an indexed triangle list (`indices`).
#[allow(clippy::too_many_arguments)]
pub fn draw_gs_indexed(
    vs: &[u8],
    gs: Option<&[u8]>,
    ps: &[u8],
    layout: &[VertexElement],
    vertices: &[f32],
    stride: u32,
    indices: &[u32],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    draw_gs_views(vs, gs, ps, layout, vertices, stride, indices, &[], &[], &[], &[], size, clear)
}

/// `draw` with textures and samplers in the directly indexed heaps (see `ViewTexture`).
#[allow(clippy::too_many_arguments)]
pub fn draw_views(
    vs: &[u8],
    ps: &[u8],
    layout: &[VertexElement],
    vertices: &[f32],
    stride: u32,
    sgprs: &[u32],
    mem: &[u32],
    textures: &[ViewTexture],
    samplers: &[D3D12_SAMPLER_DESC],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    draw_gs_views(vs, None, ps, layout, vertices, stride, &[], sgprs, mem, textures, samplers, size, clear)
}

#[allow(clippy::too_many_arguments)]
fn draw_gs_views(
    vs: &[u8],
    gs: Option<&[u8]>,
    ps: &[u8],
    layout: &[VertexElement],
    vertices: &[f32],
    stride: u32,
    indices: &[u32],
    sgprs: &[u32],
    mem: &[u32],
    textures: &[ViewTexture],
    samplers: &[D3D12_SAMPLER_DESC],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    unsafe {
        let dev = &gpu.dev;
        let rs = root_signature(dev, true)?;
        let names: Vec<std::ffi::CString> = layout.iter().map(|l| std::ffi::CString::new(l.semantic).unwrap()).collect();
        let elems: Vec<D3D12_INPUT_ELEMENT_DESC> = layout
            .iter()
            .zip(&names)
            .map(|(l, n)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(n.as_ptr() as *const u8),
                SemanticIndex: l.index,
                Format: l.format,
                InputSlot: 0,
                AlignedByteOffset: l.offset,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            })
            .collect();
        let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
            GS: gs.map_or_else(D3D12_SHADER_BYTECODE::default, |g| D3D12_SHADER_BYTECODE { pShaderBytecode: g.as_ptr() as _, BytecodeLength: g.len() }),
            SampleMask: u32::MAX,
            InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: elems.len() as u32 },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = true.into();
        pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&pd).map_err(|x| format!("CreateGraphicsPipelineState: {x}"))?;
        let draw = Draw { sgprs: sgprs.to_vec(), scissor: [0, 0, size as i32, size as i32] };
        draw_indexed_on(dev, &pso, &rs, vertices, stride, indices, &[draw], mem, textures, samplers, size, clear)
    }
}

/// Render with an existing graphics pipeline on `dev` (its own queue): vertex buffer
/// `vertices` (`stride` bytes per vertex, triangle list), the XCL shader ABI bound – root
/// signature `rs`, u0 = `sgprs` (initial SGPRs), u1 = scratch, u2 = `mem` (flat memory) –
/// into a `size`² RGBA32F target cleared to `clear`. Returns the pixels, row by row.
#[allow(clippy::too_many_arguments)]
pub fn draw_on(
    dev: &ID3D12Device,
    pso: &ID3D12PipelineState,
    rs: &ID3D12RootSignature,
    vertices: &[f32],
    stride: u32,
    sgprs: &[u32],
    mem: &[u32],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    let draw = Draw { sgprs: sgprs.to_vec(), scissor: [0, 0, size as i32, size as i32] };
    draw_on_multi(dev, pso, rs, vertices, stride, &[draw], mem, &[], &[], size, clear)
}

/// A texture in D3D12 terms (e.g. decoded from a console T# by `gpu::desc`): SRV at
/// descriptor heap index 2·i; `data` = the `MipLevels` levels one after the other, each in
/// tightly packed rows (`row_bytes` for level 0, its width's share for the smaller ones).
pub struct ViewTexture {
    pub res: D3D12_RESOURCE_DESC,
    pub srv: D3D12_SHADER_RESOURCE_VIEW_DESC,
    pub data: Vec<u8>,
    pub row_bytes: usize,
}

/// One draw of `draw_on_multi`: its initial SGPRs (u0 – user data) and scissor rectangle
/// (left, top, right, bottom).
pub struct Draw {
    pub sgprs: Vec<u32>,
    pub scissor: [i32; 4],
}

/// Heaps with `textures` (SRV at 2·i) and `samplers` (at j), bound on `list`.
unsafe fn bind_views(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, textures: &[ViewTexture], samplers: &[D3D12_SAMPLER_DESC]) -> Result<Vec<windows::core::IUnknown>, String> {
    let heap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: (2 * textures.len()).max(2) as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
        .map_err(e)?;
    let sheap: ID3D12DescriptorHeap = dev
        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: samplers.len().max(1) as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
        .map_err(e)?;
    let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
    let sinc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
    let at = |h: &ID3D12DescriptorHeap, k: usize, step: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + k * step };
    for (j, sd) in samplers.iter().enumerate() {
        dev.CreateSampler(sd, at(&sheap, j, sinc));
    }
    let mut keep: Vec<windows::core::IUnknown> = vec![heap.cast().map_err(e)?, sheap.cast().map_err(e)?];
    for (i, t) in textures.iter().enumerate() {
        let levels = t.res.MipLevels.max(1) as usize;
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); levels];
        let (mut rows, mut total) = (vec![0u32; levels], 0u64);
        dev.GetCopyableFootprints(&t.res, 0, levels as u32, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        let up = buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
        let mut p = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).map_err(e)?;
        let texel = t.row_bytes / (t.res.Width as usize).max(1);
        let mut offset = 0usize;
        for (k, fp) in fps.iter().enumerate() {
            let row_bytes = ((t.res.Width as usize) >> k).max(1) * texel;
            for y in 0..rows[k] as usize {
                if let Some(src) = t.data.get(offset + y * row_bytes..offset + (y + 1) * row_bytes) {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), (p as *mut u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), row_bytes);
                }
            }
            offset += rows[k] as usize * row_bytes;
        }
        up.Unmap(0, None);
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &t.res, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
            .map_err(|x| format!("texture {i}: {x}"))?;
        let r = r.ok_or("texture")?;
        for (k, fp) in fps.iter().enumerate() {
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        list.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)]);
        dev.CreateShaderResourceView(&r, Some(&t.srv), at(&heap, 2 * i, inc));
        keep.push(r.cast().map_err(e)?);
        keep.push(up.cast().map_err(e)?);
    }
    list.SetDescriptorHeaps(&[Some(heap), Some(sheap)]);
    Ok(keep)
}

/// Several draws with one pipeline in one command list: each draw gets its own user data
/// (u0 – e.g. a different descriptor table pointer) and scissor; textures and samplers in
/// the directly indexed heaps (see `ViewTexture`). Otherwise as `draw_on`.
#[allow(clippy::too_many_arguments)]
pub fn draw_on_multi(
    dev: &ID3D12Device,
    pso: &ID3D12PipelineState,
    rs: &ID3D12RootSignature,
    vertices: &[f32],
    stride: u32,
    draws: &[Draw],
    mem: &[u32],
    textures: &[ViewTexture],
    samplers: &[D3D12_SAMPLER_DESC],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    draw_indexed_on(dev, pso, rs, vertices, stride, &[], draws, mem, textures, samplers, size, clear)
}

/// `draw_on_multi`, the draws indexed when there are `indices` (one index buffer for all).
#[allow(clippy::too_many_arguments)]
fn draw_indexed_on(
    dev: &ID3D12Device,
    pso: &ID3D12PipelineState,
    rs: &ID3D12RootSignature,
    vertices: &[f32],
    stride: u32,
    indices: &[u32],
    draws: &[Draw],
    mem: &[u32],
    textures: &[ViewTexture],
    samplers: &[D3D12_SAMPLER_DESC],
    size: u32,
    clear: [f32; 4],
) -> Result<Vec<[f32; 4]>, String> {
    unsafe {
        let queue: ID3D12CommandQueue =
            dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;

        let tex = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: size as u64,
            Height: size,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear } };
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &tex,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&cv),
            &mut rt,
        )
        .map_err(e)?;
        let rt = rt.ok_or("render target")?;
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
            .map_err(e)?;
        let rtv = heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&rt, None, rtv);

        // vertex buffer + io (SGPRs) + mem in one upload heap
        let vb_bytes = (vertices.len() * 4) as u64;
        // user data of every draw, 128 dwords (512 bytes) apart
        let mut io = Vec::with_capacity(draws.len() * KERNEL_SGPRS);
        for d in draws {
            let mut r = d.sgprs.clone();
            r.resize(KERNEL_SGPRS, 0);
            io.extend_from_slice(&r);
        }
        let io_bytes = (io.len() * 4) as u64;
        let mem_len = mem.len().max(64);
        let mem_bytes = (mem_len * 4) as u64;
        let upload = buffer(dev, vb_bytes + io_bytes + mem_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).map_err(e)?;
        std::ptr::copy_nonoverlapping(vertices.as_ptr() as *const u32, p as *mut u32, vertices.len());
        std::ptr::copy_nonoverlapping(io.as_ptr(), (p as *mut u32).add(vertices.len()), io.len());
        let mut m = mem.to_vec();
        m.resize(mem_len, 0);
        std::ptr::copy_nonoverlapping(m.as_ptr(), (p as *mut u32).add(vertices.len() + io.len()), m.len());
        upload.Unmap(0, None);
        let io_buf = buffer(dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let mem_buf = buffer(dev, mem_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let res_buf = buffer(dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;

        let mut layout_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&tex, 0, 1, 0, Some(&mut layout_fp), None, None, Some(&mut total));
        let readback = buffer(dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso).map_err(e)?;
        list.CopyBufferRegion(&io_buf, 0, &upload, vb_bytes, io_bytes);
        list.CopyBufferRegion(&mem_buf, 0, &upload, vb_bytes + io_bytes, mem_bytes);
        list.ResourceBarrier(&[
            transition(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&mem_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.ClearRenderTargetView(rtv, &clear, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: size as f32, Height: size as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: size as i32, bottom: size as i32 }]);
        let _bound = if textures.is_empty() && samplers.is_empty() { bind_resources(dev, &list, &[], &[]).map(|_| Vec::new())? } else { bind_views(dev, &list, textures, samplers)? };
        list.SetGraphicsRootSignature(rs);
        list.SetGraphicsRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        list.SetGraphicsRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        list.SetGraphicsRootUnorderedAccessView(2, mem_buf.GetGPUVirtualAddress());
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        let vbv = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: upload.GetGPUVirtualAddress(), SizeInBytes: vb_bytes as u32, StrideInBytes: stride };
        list.IASetVertexBuffers(0, Some(&[vbv]));
        let _index_buf = if indices.is_empty() {
            None
        } else {
            let bytes = (indices.len() * 4) as u64;
            let b = buffer(dev, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
            let mut p = std::ptr::null_mut();
            b.Map(0, None, Some(&mut p)).map_err(e)?;
            std::ptr::copy_nonoverlapping(indices.as_ptr(), p as *mut u32, indices.len());
            b.Unmap(0, None);
            list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: b.GetGPUVirtualAddress(), SizeInBytes: bytes as u32, Format: DXGI_FORMAT_R32_UINT }));
            Some(b)
        };
        for (k, d) in draws.iter().enumerate() {
            list.SetGraphicsRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress() + (k * KERNEL_SGPRS * 4) as u64);
            let [left, top, right, bottom] = d.scissor;
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left, top, right, bottom }]);
            if indices.is_empty() {
                list.DrawInstanced((vertices.len() as u32 * 4) / stride, 1, 0, 0);
            } else {
                list.DrawIndexedInstanced(indices.len() as u32, 1, 0, 0, 0);
            }
        }
        list.ResourceBarrier(&[transition(&rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(&readback),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: layout_fp },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(&rt),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        wait_for_submission(dev, &queue, &fence, 1, "draw")?;
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).map_err(e)?;
        let pitch = layout_fp.Footprint.RowPitch as usize;
        let mut px = Vec::with_capacity((size * size) as usize);
        for y in 0..size as usize {
            for x in 0..size as usize {
                let q = (p as *const u8).add(y * pitch + x * 16) as *const f32;
                px.push([*q, *q.add(1), *q.add(2), *q.add(3)]);
            }
        }
        readback.Unmap(0, None);
        Ok(px)
    }
}

fn uav_barrier(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: unsafe { transmute_copy(r) } }) },
    }
}

/// One pipeline-state-stream subobject (aligned to a pointer, as in d3dx12.h).
#[repr(C, align(8))]
struct Sub<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    v: T,
}

#[repr(C)]
struct MeshStream {
    rs: Sub<*mut std::ffi::c_void>,
    ms: Sub<D3D12_SHADER_BYTECODE>,
    ps: Sub<D3D12_SHADER_BYTECODE>,
    blend: Sub<D3D12_BLEND_DESC>,
    mask: Sub<u32>,
    rast: Sub<D3D12_RASTERIZER_DESC>,
    depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
    rtv: Sub<D3D12_RT_FORMAT_ARRAY>,
    samples: Sub<DXGI_SAMPLE_DESC>,
}

/// Mesh mode, both passes: `cs` = the translated NGG program (writes one record of
/// `record_dwords` per group into u1), `ms` = its pass-through mesh shader, `ps` = pixel
/// shader; `groups` thread groups, initial SGPRs `sgprs` (u0), flat memory `mem` (u2).
/// Returns the RGBA32F pixels of a `size`² target and the records.
#[allow(clippy::too_many_arguments)]
pub fn mesh_draw(
    cs: &[u8],
    ms: &[u8],
    ps: &[u8],
    groups: u32,
    record_dwords: usize,
    sgprs: &[u32],
    mem: &[u32],
    size: u32,
    clear: [f32; 4],
) -> Result<(Vec<[f32; 4]>, Vec<u32>), String> {
    mesh_draw_split(cs, ms, ps, groups, 1, record_dwords, sgprs, mem, size, clear)
}

/// `mesh_draw` with `split` mesh groups per record (`emit::passthrough_split`).
#[allow(clippy::too_many_arguments)]
pub fn mesh_draw_split(
    cs: &[u8],
    ms: &[u8],
    ps: &[u8],
    groups: u32,
    split: u32,
    record_dwords: usize,
    sgprs: &[u32],
    mem: &[u32],
    size: u32,
    clear: [f32; 4],
) -> Result<(Vec<[f32; 4]>, Vec<u32>), String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    unsafe {
        let dev = &gpu.dev;
        let cpso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&gpu.rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as _, BytecodeLength: cs.len() },
                ..Default::default()
            })
            .map_err(|x| format!("CreateComputePipelineState: {x}"))?;
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let rast = D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
        let mut rtv_formats = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rtv_formats.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let mut stream = MeshStream {
            rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: gpu.rs.as_raw() },
            ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: D3D12_SHADER_BYTECODE { pShaderBytecode: ms.as_ptr() as _, BytecodeLength: ms.len() } },
            ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() } },
            blend: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, v: blend },
            mask: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, v: u32::MAX },
            rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, v: rast },
            depth: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, v: D3D12_DEPTH_STENCIL_DESC::default() },
            rtv: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rtv_formats },
            samples: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, v: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 } },
        };
        let dev2: ID3D12Device2 = dev.cast().map_err(e)?;
        let mpso: ID3D12PipelineState = dev2
            .CreatePipelineState(&D3D12_PIPELINE_STATE_STREAM_DESC {
                SizeInBytes: std::mem::size_of::<MeshStream>(),
                pPipelineStateSubobjectStream: &mut stream as *mut MeshStream as *mut std::ffi::c_void,
            })
            .map_err(|x| format!("CreatePipelineState (mesh): {x}"))?;

        let tex = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: size as u64,
            Height: size,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear } };
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &tex,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&cv),
            &mut rt,
        )
        .map_err(e)?;
        let rt = rt.ok_or("render target")?;
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
            .map_err(e)?;
        let rtv = heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&rt, None, rtv);

        let mut io = sgprs.to_vec();
        io.resize(KERNEL_SGPRS, 0);
        let io_bytes = (io.len() * 4) as u64;
        let mem_len = mem.len().max(64);
        let mem_bytes = (mem_len * 4) as u64;
        let res_len = record_dwords * groups as usize;
        let res_bytes = (res_len * 4) as u64;
        let upload = buffer(dev, io_bytes + mem_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).map_err(e)?;
        std::ptr::copy_nonoverlapping(io.as_ptr(), p as *mut u32, io.len());
        let mut m = mem.to_vec();
        m.resize(mem_len, 0);
        std::ptr::copy_nonoverlapping(m.as_ptr(), (p as *mut u32).add(io.len()), m.len());
        upload.Unmap(0, None);
        let io_buf = buffer(dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let mem_buf = buffer(dev, mem_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let res_buf = buffer(dev, res_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&tex, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let readback = buffer(dev, total + res_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &cpso).map_err(e)?;
        list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_bytes);
        list.CopyBufferRegion(&mem_buf, 0, &upload, io_bytes, mem_bytes);
        list.ResourceBarrier(&[
            transition(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&mem_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        // pass 1: the console program as a compute shader
        let _bound = bind_resources(dev, &list, &[], &[])?;
        list.SetComputeRootSignature(&gpu.rs);
        list.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, mem_buf.GetGPUVirtualAddress());
        list.Dispatch(groups, 1, 1);
        list.ResourceBarrier(&[uav_barrier(&res_buf)]);
        // pass 2: rasterise the records
        list.SetPipelineState(&mpso);
        list.SetGraphicsRootSignature(&gpu.rs);
        list.SetGraphicsRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        list.SetGraphicsRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        list.SetGraphicsRootUnorderedAccessView(2, mem_buf.GetGPUVirtualAddress());
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.ClearRenderTargetView(rtv, &clear, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: size as f32, Height: size as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: size as i32, bottom: size as i32 }]);
        let list6: ID3D12GraphicsCommandList6 = list.cast().map_err(e)?;
        list6.DispatchMesh(groups * split, 1, 1);
        list.ResourceBarrier(&[
            transition(&rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(&res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rt), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.CopyBufferRegion(&readback, total, &res_buf, 0, res_bytes);
        list.Close().map_err(e)?;
        gpu.queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        gpu.value += 1;
        wait_for_submission(dev, &gpu.queue, &gpu.fence, gpu.value, "mesh draw")?;
        let mut q = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut q)).map_err(e)?;
        let pitch = fp.Footprint.RowPitch as usize;
        let mut px = Vec::with_capacity((size * size) as usize);
        for y in 0..size as usize {
            for x in 0..size as usize {
                let c = (q as *const u8).add(y * pitch + x * 16) as *const f32;
                px.push([*c, *c.add(1), *c.add(2), *c.add(3)]);
            }
        }
        let recs = std::slice::from_raw_parts((q as *const u8).add(total as usize) as *const u32, res_len).to_vec();
        readback.Unmap(0, None);
        Ok((px, recs))
    }
}

/// Process memory regions for `run_in_region` (see `gpu::uma`): the memory windows.
struct Arena {
    pub windows: Vec<(*mut u8, usize)>,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}


/// `run` with a unified memory region as `mem` (u2): the GPU works on the process memory
/// itself, no copies.
pub fn run_in_arena(dxil: &[u8], state: &KernelState, arena: &crate::gpu::uma::Uma) -> Result<KernelResult, String> {
    let arena = Arena { windows: (0..arena.windows()).map(|k| (arena.window(k).0 as *mut u8, arena.window(k).1)).collect() };
    let r = run_in_region(dxil, state, &arena);
    std::mem::forget(arena); // owned by the Uma
    r
}

fn run_in_region(dxil: &[u8], state: &KernelState, arena: &Arena) -> Result<KernelResult, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    let mut io: Vec<u32> = state.sgpr.clone();
    io.resize(KERNEL_SGPRS, 0);
    io[KERNEL_SCC_SLOT] = state.scc;
    if let Some(&(base, _)) = arena.windows.first() {
        io[super::IO_UMA_BASE_HI] = ((base as usize) >> 32) as u32;
    }
    for lane in 0..KERNEL_LANES {
        let mut regs = state.vgpr.get(lane).cloned().unwrap_or_default();
        regs.resize(KERNEL_VGPRS, 0);
        io.extend_from_slice(&regs);
    }
    unsafe {
        let dev = &gpu.dev;
        let dev3: ID3D12Device3 = dev.cast().map_err(e)?;
        let mut bufs: Vec<ID3D12Resource> = Vec::new();
        for &(base, size) in &arena.windows {
        let heap: ID3D12Heap = dev3.OpenExistingHeapFromAddress(base as *const std::ffi::c_void).map_err(|x| format!("OpenExistingHeapFromAddress: {x}"))?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
            ..Default::default()
        };
        let mut mem_buf: Option<ID3D12Resource> = None;
        dev.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut mem_buf).map_err(|x| format!("CreatePlacedResource: {x}"))?;
        bufs.push(mem_buf.ok_or("arena")?);
        }
        let mem_buf = bufs[0].clone();
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&gpu.rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .map_err(|x| format!("CreateComputePipelineState: {x}"))?;
        let io_bytes = (io.len() * 4) as u64;
        let res_bytes = (KERNEL_RES_DWORDS * 4) as u64;
        let upload = buffer(dev, io_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
        let io_buf = buffer(dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let res_buf = buffer(dev, res_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let readback = buffer(dev, res_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).map_err(e)?;
        std::ptr::copy_nonoverlapping(io.as_ptr(), p as *mut u32, io.len());
        upload.Unmap(0, None);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(e)?;
        list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_bytes);
        list.ResourceBarrier(&[
            transition(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&mem_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        for b in &bufs[1..] {
            list.ResourceBarrier(&[transition(b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        }
        // the windows as raw UAVs at MEM_HEAP_BASE.. of a shader-visible heap
        // (`Options::heap_windows`; programs with the root-UAV switch ignore it)
        let win_count = arena.windows.len().clamp(crate::gpu::uma::MAX_WINDOWS, crate::gpu::uma::MAX_HEAP_WINDOWS);
        let dh: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: super::MEM_HEAP_BASE + win_count as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(e)?;
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        for k in 0..win_count {
            let (b, size) = bufs.get(k).map_or((&mem_buf, arena.windows[0].1), |b| (b, arena.windows[k].1));
            let mut h = dh.GetCPUDescriptorHandleForHeapStart();
            h.ptr += (super::MEM_HEAP_BASE as usize + k) * inc;
            dev.CreateUnorderedAccessView(b, None::<&ID3D12Resource>, Some(&raw_window_uav(size as u64)), h);
        }
        list.SetDescriptorHeaps(&[Some(dh.clone())]);
        list.SetComputeRootSignature(&gpu.rs);
        list.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        for k in 0..4 {
            list.SetComputeRootUnorderedAccessView(2 + k as u32, bufs.get(k).unwrap_or(&mem_buf).GetGPUVirtualAddress());
        }
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[
            transition(&res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(&mem_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        ]);
        for b in &bufs[1..] {
            list.ResourceBarrier(&[transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)]);
        }
        list.CopyBufferRegion(&readback, 0, &res_buf, 0, res_bytes);
        list.Close().map_err(e)?;
        gpu.queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        gpu.value += 1;
        wait_for_submission(dev, &gpu.queue, &gpu.fence, gpu.value, "arena dispatch")?;
        let mut q = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut q)).map_err(e)?;
        let raw = std::slice::from_raw_parts(q as *const u32, KERNEL_RES_DWORDS).to_vec();
        readback.Unmap(0, None);
        let vgpr = (0..KERNEL_LANES).map(|l| raw[KERNEL_SGPRS + l * KERNEL_VGPRS..KERNEL_SGPRS + (l + 1) * KERNEL_VGPRS].to_vec()).collect();
        split_halves_agree(&raw)?;
        Ok(KernelResult { sgpr: raw[..KERNEL_SGPRS].to_vec(), vgpr, scc: raw[KERNEL_SCC_SLOT], raw, mem: Vec::new(), textures: Vec::new() })
    }
}

/// A raw UAV of a memory window of `bytes` (`Options::heap_windows`): at most 4 GiB − 32 bytes.
/// An older D3D12 runtime (the one in Windows 11 23H2, D3D12Core 10.0.22621) refuses a raw view
/// that ends above that from the start of its resource and removes the device; newer ones take
/// 2^30 − 1 dwords (measured by CharlyRien on PR #680). The sparse windows hand out their last
/// 64 KiB tile – a view of 4 GiB − 64 KiB left it unreachable (zeros, dropped stores); this one
/// misses only the window's last 32 bytes
/// (`heap_window_tests::gpu_raw_window_view_reaches_its_top`).
pub fn raw_window_uav(bytes: u64) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let bytes = bytes.min((1u64 << 32) - 32);
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: (bytes / 4) as u32, StructureByteStride: 0, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_RAW },
        },
    }
}

/// Create a committed resource from a description (validation of decoded descriptors).
pub fn try_create_texture(desc: &D3D12_RESOURCE_DESC) -> Result<(), String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    let mut r: Option<ID3D12Resource> = None;
    unsafe {
        gpu.dev
            .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r)
            .map_err(|x| format!("CreateCommittedResource: {x}"))
    }
}

/// A compute shader with the console dispatch ABI (`Mode::Compute`): `groups` thread groups,
/// u0 = user SGPRs, u2 = flat memory. Returns the memory afterwards.
pub fn dispatch(dxil: &[u8], groups: [u32; 3], sgprs: &[u32], mem: &[u32]) -> Result<Vec<u32>, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    let pso: ID3D12PipelineState = unsafe {
        gpu.dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&gpu.rs),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
            ..Default::default()
        })
    }
    .map_err(|x| format!("CreateComputePipelineState: {x}"))?;
    dispatch_with(&gpu.dev, &gpu.queue, &gpu.fence, &mut gpu.value, &pso, &gpu.rs, groups, sgprs, mem, 1, false, None).map(|(m, _)| m)
}

#[cfg(test)]
pub(crate) fn dispatch_split(
    chain: &crate::gpu::dxil::kernel::DispatchChain, groups: [u32; 3], mem: &[u32],
) -> Result<Vec<u32>, String> {
    dispatch_split_by(chain, groups, mem, u64::MAX)
}

/// `dispatch_split`, at most `most` groups at once (`pso::dispatch_slabs`).
#[cfg(test)]
pub(crate) fn dispatch_split_by(
    chain: &crate::gpu::dxil::kernel::DispatchChain, groups: [u32; 3], mem: &[u32], most: u64,
) -> Result<Vec<u32>, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|e| e.clone())?;
    let rs = super::pso::dispatch_chain_root_signature(&gpu.dev, &[])?;
    let pieces: Vec<ID3D12PipelineState> = chain.pieces.iter().map(|dxil| unsafe {
        gpu.dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
            ..Default::default()
        }).map_err(e)
    }).collect::<Result<_,_>>()?;
    dispatch_with(&gpu.dev, &gpu.queue, &gpu.fence, &mut gpu.value, &pieces[0], &rs, groups, &[], mem, 1, false,
                  Some((&pieces, chain.dwords_per_group, most))).map(|(m,_)| m)
}

/// `dispatch` run `reps` times in one command list (a UAV barrier between them, memory not
/// reset), each between two GPU timestamps: GPU milliseconds per dispatch – no pipeline,
/// buffer, upload, submission or CPU wait time in them. For benchmarks.
pub fn dispatch_timed(dxil: &[u8], groups: [u32; 3], sgprs: &[u32], mem: &[u32], reps: usize) -> Result<Vec<f64>, String> {
    let mut g = GPU.lock().unwrap();
    let gpu = g.get_or_insert_with(init).as_mut().map_err(|x| x.clone())?;
    let pso: ID3D12PipelineState = unsafe {
        gpu.dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&gpu.rs),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
            ..Default::default()
        })
    }
    .map_err(|x| format!("CreateComputePipelineState: {x}"))?;
    dispatch_with(&gpu.dev, &gpu.queue, &gpu.fence, &mut gpu.value, &pso, &gpu.rs, groups, sgprs, mem, reps.max(1), true, None).map(|(_, t)| t)
}

/// Dispatch with an existing compute pipeline on `dev` (its own queue), e.g. one the D3D12.X
/// backend built from a console shader: root signature `rs` with the XCL shader ABI, u0 =
/// `sgprs`, u2 = `mem`. Returns the memory afterwards.
pub fn dispatch_on(dev: &ID3D12Device, pso: &ID3D12PipelineState, rs: &ID3D12RootSignature, groups: [u32; 3], sgprs: &[u32], mem: &[u32]) -> Result<Vec<u32>, String> {
    unsafe {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        dispatch_with(dev, &queue, &fence, &mut 0, pso, rs, groups, sgprs, mem, 1, false, None).map(|(m, _)| m)
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_with(
    dev: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    fence: &ID3D12Fence,
    value: &mut u64,
    pso: &ID3D12PipelineState,
    rs: &ID3D12RootSignature,
    groups: [u32; 3],
    sgprs: &[u32],
    mem: &[u32],
    reps: usize,
    timed: bool,
    chain: Option<(&[ID3D12PipelineState], u32, u64)>,
) -> Result<(Vec<u32>, Vec<f64>), String> {
    let mut io = sgprs.to_vec();
    io.resize(KERNEL_SGPRS, 0);
    let mem_len = mem.len().max(64);
    unsafe {
        let (io_bytes, mem_bytes) = ((io.len() * 4) as u64, (mem_len * 4) as u64);
        let upload = buffer(dev, io_bytes + mem_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false)?;
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).map_err(e)?;
        std::ptr::copy_nonoverlapping(io.as_ptr(), p as *mut u32, io.len());
        let mut m = mem.to_vec();
        m.resize(mem_len, 0);
        std::ptr::copy_nonoverlapping(m.as_ptr(), (p as *mut u32).add(io.len()), m.len());
        upload.Unmap(0, None);
        let io_buf = buffer(dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let mem_buf = buffer(dev, mem_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let res_buf = buffer(dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true)?;
        let readback = buffer(dev, mem_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso).map_err(e)?;
        list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_bytes);
        list.CopyBufferRegion(&mem_buf, 0, &upload, io_bytes, mem_bytes);
        list.ResourceBarrier(&[
            transition(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&mem_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        let bound = bind_resources(dev, &list, &[], &[])?;
        // A PSO built by the backend's hook uses the runtime's window count: with heap windows
        // (`Options::heap_windows`) its accesses go to ResourceDescriptorHeap[MEM_HEAP_BASE + w],
        // so every window slot is a raw view of `mem` (no textures here to share the heap with).
        let win_count = crate::gpu::uma::MAX_HEAP_WINDOWS;
        let dh: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: super::MEM_HEAP_BASE + win_count as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(e)?;
        let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        for k in 0..win_count {
            let mut h = dh.GetCPUDescriptorHandleForHeapStart();
            h.ptr += (super::MEM_HEAP_BASE as usize + k) * inc;
            dev.CreateUnorderedAccessView(&mem_buf, None::<&ID3D12Resource>, Some(&raw_window_uav(mem_bytes)), h);
        }
        list.SetDescriptorHeaps(&[Some(dh.clone()), Some(bound._heaps[1].clone())]);
        list.SetComputeRootSignature(rs);
        list.SetComputeRootUnorderedAccessView(0, io_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(1, res_buf.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, mem_buf.GetGPUVirtualAddress());
        let scratch = chain.map(|(_, words, most)| {
            let first = super::pso::dispatch_slabs(groups, most)[0].1;
            let bytes = first.iter().fold(4 * u64::from(words), |n, &g| n * u64::from(g));
            buffer(dev, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true)
        }).transpose()?;
        // `reps` dispatches, each between two timestamps when `timed`
        let stamps = 2 * reps as u32;
        let queries = if timed {
            let mut h: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: stamps, NodeMask: 0 }, &mut h).map_err(e)?;
            let heap = h.ok_or("CreateQueryHeap")?;
            let out = buffer(dev, stamps as u64 * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false)?;
            Some((heap, out))
        } else {
            None
        };
        for k in 0..reps as u32 {
            if k > 0 {
                list.ResourceBarrier(&[uav_barrier(&mem_buf), uav_barrier(&io_buf)]);
            }
            if let Some((heap, _)) = &queries {
                list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k);
            }
            if let (Some((pieces, _, most)), Some(scratch)) = (chain, &scratch) {
                super::pso::record_dispatch_chain(&list, groups, most, scratch, k > 0, pieces);
            } else {
                list.Dispatch(groups[0], groups[1], groups[2]);
            }
            if let Some((heap, _)) = &queries {
                list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * k + 1);
            }
        }
        if let Some((heap, out)) = &queries {
            list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, stamps, out, 0);
        }
        list.ResourceBarrier(&[transition(&mem_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&readback, 0, &mem_buf, 0, mem_bytes);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        *value += 1;
        wait_for_submission(dev, queue, fence, *value, "compute dispatch")?;
        let mut q = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut q)).map_err(e)?;
        let out = std::slice::from_raw_parts(q as *const u32, mem_len).to_vec();
        readback.Unmap(0, None);
        let mut ms = Vec::new();
        if let Some((_, stamps_buf)) = &queries {
            let hz = queue.GetTimestampFrequency().map_err(e)? as f64;
            let mut t = std::ptr::null_mut();
            stamps_buf.Map(0, None, Some(&mut t)).map_err(e)?;
            let ticks = std::slice::from_raw_parts(t as *const u64, stamps as usize);
            ms = ticks.chunks(2).map(|c| c[1].saturating_sub(c[0]) as f64 * 1000.0 / hz).collect();
            stamps_buf.Unmap(0, None);
        }
        Ok((out, ms))
    }
}
