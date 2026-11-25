//! D3D12.X checks of the test title. The device comes from `d3d12_xs.dll` (XCL);
//! the `windows` crate is only used for D3D12 types and vtable calls – exactly what a
//! console title compiled against a PC-compatible D3D12.X layout would do.

use std::ffi::c_void;
use std::mem::{transmute_copy, ManuallyDrop};

use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

type HRESULT = i32;

type D3D12CreateDeviceFn = unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT;
type D3D12SerializeRootSignatureFn = unsafe extern "system" fn(*const c_void, u32, *mut *mut c_void, *mut *mut c_void) -> HRESULT;

// the console's texture layout library
#[link(name = "xg_xs", kind = "raw-dylib")]
extern "system" {
    fn XGCreateTextureComputer(desc: *const D3D12_RESOURCE_DESC, computer: *mut *mut c_void) -> HRESULT;
}

// Fortnite reads its data through the console's DirectStorage
#[link(name = "xcrdapi", kind = "raw-dylib")]
extern "system" {
    fn DStorageGetFactory(riid: *const windows::core::GUID, factory: *mut *mut c_void) -> HRESULT;
}

// Media Foundation: standard exports fall back to Windows' mfplat.dll, while
// MFResetDXGIDeviceManagerX routes to XCL's HLE export.
#[link(name = "mfplat", kind = "raw-dylib")]
extern "system" {
    fn MFStartup(version: u32, flags: u32) -> HRESULT;
    fn MFShutdown() -> HRESULT;
    fn MFCreateDXGIDeviceManager(reset_token: *mut u32, manager: *mut *mut c_void) -> HRESULT;
    fn MFResetDXGIDeviceManagerX(manager: *mut c_void, device: *mut c_void, reset_token: u32) -> HRESULT;
}

#[link(name = "d3dcompiler_47", kind = "raw-dylib")]
extern "system" {
    fn D3DCompile(
        src: *const u8,
        len: usize,
        name: *const u8,
        defines: *const c_void,
        include: *const c_void,
        entry: *const u8,
        target: *const u8,
        flags1: u32,
        flags2: u32,
        code: *mut *mut c_void,
        errors: *mut *mut c_void,
    ) -> HRESULT;
}

#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn CreateEventW(sa: *const c_void, manual: i32, initial: i32, name: *const u16) -> *mut c_void;
    fn WaitForSingleObject(h: *mut c_void, ms: u32) -> u32;
    fn CloseHandle(h: *mut c_void) -> i32;
}

macro_rules! ensure {
    ($c:expr, $($m:tt)*) => { if !$c { return Err(format!($($m)*)); } };
}

struct EventHandle(*mut c_void);

impl EventHandle {
    unsafe fn new() -> Result<Self, String> {
        let h = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
        ensure!(!h.is_null(), "CreateEventW");
        Ok(Self(h))
    }

    fn handle(&self) -> windows::Win32::Foundation::HANDLE {
        windows::Win32::Foundation::HANDLE(self.0)
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

/// An export of `d3d12_xs.dll` (XCL), looked up by name. Not a static import: the `windows`
/// crate imports the same two functions from `d3d12.dll`, and of two imports of one symbol
/// the linker keeps one – the system's. The title then got the system's device for the
/// default adapter, which is XCL's device only while that adapter is the one XCL chose
/// (D3D12 hands out one device per adapter); on a machine with two GPUs it is another one.
unsafe fn xs<F: Copy>(name: &str) -> Result<F, String> {
    let h = crate::LoadLibraryW(crate::wide("d3d12_xs.dll").as_ptr());
    ensure!(h != 0, "LoadLibrary(d3d12_xs.dll) failed");
    let f = crate::GetProcAddress(h, format!("{name}\0").as_ptr());
    ensure!(f != 0, "d3d12_xs.dll!{name} not found");
    Ok(transmute_copy(&f))
}

/// A command list method at its console slot: from SetDescriptorHeaps (PC 28) on, the
/// console list has the PC methods 4 slots further (the query methods come first).
unsafe fn xslot<F: Copy>(list: &ID3D12GraphicsCommandList, slot: usize) -> F {
    transmute_copy(&*(*(list.as_raw() as *const *const usize)).add(slot))
}

type OmSetRenderTargets = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_CPU_DESCRIPTOR_HANDLE, i32, *const D3D12_CPU_DESCRIPTOR_HANDLE);
type ClearDsv = unsafe extern "system" fn(*mut c_void, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CLEAR_FLAGS, f32, u8, u32, *const RECT);
type ClearRtv = unsafe extern "system" fn(*mut c_void, D3D12_CPU_DESCRIPTOR_HANDLE, *const [f32; 4], u32, *const RECT);
type SetRootSignature = unsafe extern "system" fn(*mut c_void, *mut c_void);
const X_SET_GRAPHICS_ROOT_SIGNATURE: usize = 34;
const X_OM_SET_RENDER_TARGETS: usize = 50;
const X_CLEAR_DEPTH_STENCIL_VIEW: usize = 51;
const X_CLEAR_RENDER_TARGET_VIEW: usize = 52;

fn e(x: windows::core::Error) -> String {
    format!("{x}")
}

fn device() -> Result<ID3D12Device, String> {
    unsafe {
        let mut p = std::ptr::null_mut();
        let create: D3D12CreateDeviceFn = xs("D3D12CreateDevice")?;
        let hr = create(std::ptr::null_mut(), 0xC000, &ID3D12Device::IID as *const _ as *const c_void, &mut p);
        ensure!(hr >= 0 && !p.is_null(), "D3D12CreateDevice hr={hr:#x}");
        Ok(ID3D12Device::from_raw(p))
    }
}

fn compile(src: &str, target: &str) -> Result<ID3DBlob, String> {
    unsafe {
        let mut blob = std::ptr::null_mut();
        let t = format!("{target}\0");
        let hr = D3DCompile(src.as_ptr(), src.len(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
            b"main\0".as_ptr(), t.as_ptr(), 0, 0, &mut blob, std::ptr::null_mut());
        ensure!(hr == 0 && !blob.is_null(), "D3DCompile({target}) hr={hr:#x}");
        Ok(ID3DBlob::from_raw(blob))
    }
}

fn bc(b: &ID3DBlob) -> D3D12_SHADER_BYTECODE {
    unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: b.GetBufferPointer(), BytecodeLength: b.GetBufferSize() } }
}

fn root_signature(dev: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    unsafe {
        let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
        let (mut blob, mut err) = (std::ptr::null_mut(), std::ptr::null_mut());
        let serialize: D3D12SerializeRootSignatureFn = xs("D3D12SerializeRootSignature")?;
        let hr = serialize(&desc as *const _ as *const c_void, 1, &mut blob, &mut err);
        ensure!(hr == 0 && !blob.is_null(), "D3D12SerializeRootSignature hr={hr:#x}");
        let blob = ID3DBlob::from_raw(blob);
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
        dev.CreateRootSignature(0, bytes).map_err(e)
    }
}

fn pso_desc(rs: &ID3D12RootSignature, vs: D3D12_SHADER_BYTECODE, ps: D3D12_SHADER_BYTECODE) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
    let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
    d.pRootSignature = unsafe { transmute_copy(rs) };
    d.VS = vs;
    d.PS = ps;
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    d.SampleMask = u32::MAX;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = true.into();
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = 1;
    d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
    d
}

fn wait(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, value: u64) -> Result<(), String> {
    unsafe {
        queue.Signal(fence, value).map_err(e)?;
        let ev = EventHandle::new()?;
        fence.SetEventOnCompletion(value, ev.handle()).map_err(e)?;
        ensure!(WaitForSingleObject(ev.0, 5000) == 0, "GPU fence {value} not reached");
        ensure!(fence.GetCompletedValue() >= value, "fence value {}", fence.GetCompletedValue());
        Ok(())
    }
}

pub fn queue_fence() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 3)?;
        wait(&queue, &fence, 4)
    }
}

/// Fake console container: unknown part "XBOX" with RDNA2 code
/// (s_mov_b32 s0, s1 ; v_add_f32 v5, v1, v2 ; s_endpgm).
fn console_container() -> Vec<u8> {
    let code: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x05, 0x0a, 0x06, 0x00, 0x00, 0x81, 0xbf];
    let mut xb = vec![0u8; 36];
    xb[0..4].copy_from_slice(b"DXBC");
    xb[28..32].copy_from_slice(&1u32.to_le_bytes());
    xb[32..36].copy_from_slice(&36u32.to_le_bytes());
    xb.extend_from_slice(b"XBOX");
    xb.extend_from_slice(&(code.len() as u32).to_le_bytes());
    xb.extend_from_slice(&code);
    let total = xb.len() as u32;
    xb[24..28].copy_from_slice(&total.to_le_bytes());
    xb
}

fn le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// Console shader in the retail layout: DXBC whose DXIL part carries no bitcode ("NOBC")
/// but XDAT → XBCH (zlib) → FSSM with native code. The code uses v_mac_f32, which only
/// the console GPU has (removed in retail RDNA2):
/// v_mac_f32 v6, s40, v21 ; s_endpgm
fn xbox_native_container() -> Vec<u8> {
    // FSSM: code pointer at +0xC0/+0xC8 (relative to +0x10), PS hardware-stage descriptor at +0x100
    let code: [u8; 8] = [0x28, 0x2a, 0x0c, 0x3e, 0x00, 0x00, 0x81, 0xbf];
    let mut fssm = vec![0u8; 0x308];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&0x2f8u64.to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    fssm[0x100..0x108].copy_from_slice(&0x0000_00C0_0002_0105u64.to_le_bytes());
    fssm.extend_from_slice(&code);
    let size = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&size.to_le_bytes());
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    for x in [3, z.len() as u32, 0, fssm.len() as u32, 0] {
        le(&mut xbch, x);
    }
    xbch.extend_from_slice(&z);
    // DXIL part: program header (24 B), XDAT, entries, NOBC placeholder
    let hash = [0x5au8; 16];
    let table = 0x18 + 12;
    let data_at = table + 2 * 12;
    let mut d = Vec::new();
    for x in [0x66u32, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4] {
        le(&mut d, x);
    }
    d.extend_from_slice(b"XDAT");
    for x in [2, table as u32, 0x12, data_at as u32, 16, 0x0d, (data_at + 16) as u32, xbch.len() as u32] {
        le(&mut d, x);
    }
    d.extend_from_slice(&hash);
    d.extend_from_slice(&xbch);
    while d.len() % 4 != 0 {
        d.push(0);
    }
    let nobc = d.len();
    d.extend_from_slice(b"NOBC");
    let dwords = (d.len() / 4) as u32;
    d[4..8].copy_from_slice(&dwords.to_le_bytes());
    d[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());
    // DXBC with a single DXIL part
    let mut v = vec![0u8; 36];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&1u32.to_le_bytes());
    v[32..36].copy_from_slice(&36u32.to_le_bytes());
    v.extend_from_slice(b"DXIL");
    le(&mut v, d.len() as u32);
    v.extend_from_slice(&d);
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

pub fn xbox_native_pso() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let rs = root_signature(&dev)?;
        let vs = compile("float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, 0, 0, 1); }", "vs_5_0")?;
        let xb = xbox_native_container();
        let ps = D3D12_SHADER_BYTECODE { pShaderBytecode: xb.as_ptr() as *const c_void, BytecodeLength: xb.len() };
        let pso: Result<ID3D12PipelineState, _> = dev.CreateGraphicsPipelineState(&pso_desc(&rs, bc(&vs), ps));
        ensure!(pso.is_ok(), "CreateGraphicsPipelineState with a retail-format console shader failed: {:?}", pso.err());
        Ok(())
    }
}

pub fn pso_capture_and_fallback() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let rs = root_signature(&dev)?;
        let vs = compile("float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, 0, 0, 1); }", "vs_5_0")?;
        let xb = console_container();
        let console_ps = D3D12_SHADER_BYTECODE { pShaderBytecode: xb.as_ptr() as *const c_void, BytecodeLength: xb.len() };
        let desc = pso_desc(&rs, bc(&vs), console_ps);
        // the PC driver rejects the console PS; XCL must substitute a stand-in shader
        let pso: Result<ID3D12PipelineState, _> = dev.CreateGraphicsPipelineState(&desc);
        ensure!(pso.is_ok(), "CreateGraphicsPipelineState with a console shader failed: {:?}", pso.err());
        Ok(())
    }
}

/// Console presentation as a title does it (as Microsoft's GDK samples): the display output
/// through the device's IDXGIDevice1 → adapter → output, a 60 Hz frame clock on it (device
/// #78 / #79), the
/// origin wait that hands out the frame token (#80), a cleared 640×360 back buffer handed to
/// `PresentX` (queue #30) with that token, then the frame's statistics (#81).
pub fn present_x_frame() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let slot = |obj: *mut c_void, i: usize| *(*(obj as *const *const usize)).add(i);
        let d = dev.as_raw();
        let set_interval: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32) -> HRESULT = std::mem::transmute(slot(d, 78));
        let schedule: unsafe extern "system" fn(*mut c_void, u32, u32, *const u8, u32) -> HRESULT = std::mem::transmute(slot(d, 79));
        let wait_event: unsafe extern "system" fn(*mut c_void, u32, u32, *const u8, u32, *mut u64) -> HRESULT = std::mem::transmute(slot(d, 80));
        let statistics: unsafe extern "system" fn(*mut c_void, u64, u32, *mut u32, *mut u8) -> HRESULT = std::mem::transmute(slot(d, 81));
        let dxgi: windows::Win32::Graphics::Dxgi::IDXGIDevice1 = dev.cast().map_err(|x| format!("IDXGIDevice1 from the device: {x}"))?;
        let adapter = dxgi.GetAdapter().map_err(e)?;
        let desc = adapter.GetDesc().map_err(e)?;
        let luid = dev.GetAdapterLuid();
        ensure!((desc.AdapterLuid.LowPart, desc.AdapterLuid.HighPart) == (luid.LowPart, luid.HighPart), "IDXGIDevice1::GetAdapter: another adapter");
        let back: ID3D12Device = dxgi.cast().map_err(e)?;
        ensure!(back.as_raw() == dev.as_raw(), "IDXGIDevice1 → ID3D12Device is not the device");
        // an adapter without a display (e.g. a laptop's discrete GPU) has no output
        let output = adapter.EnumOutputs(0).ok();
        let output_raw = output.as_ref().map_or(std::ptr::null_mut(), |o| o.as_raw());
        ensure!(set_interval(d, output_raw, 16667, 1, 0) >= 0, "SetFrameIntervalX");
        let res = (|| -> Result<(), String> {
            ensure!(schedule(d, 0, 0, std::ptr::null(), 0) >= 0, "ScheduleFrameEventX (origin)");
            let mut token = 0u64;
            let hr = wait_event(d, 0, 500, std::ptr::null(), 0, &mut token);
            ensure!(hr == 0 && token != 0, "WaitFrameEventX hr={hr:#x} token={token}");

            let (w, h) = (640u32, 360u32);
            let tex = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: w as u64,
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut bb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut bb)
                .map_err(e)?;
            let bb = bb.ok_or("no back buffer")?;
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
                .map_err(e)?;
            let start: unsafe extern "system" fn(*mut c_void) -> usize = std::mem::transmute(slot(heap.as_raw(), 9));
            let rtv = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start(heap.as_raw()) };
            dev.CreateRenderTargetView(&bb, None, rtv);
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
            xslot::<ClearRtv>(&list, X_CLEAR_RENDER_TARGET_VIEW)(list.as_raw(), rtv, &[0.1, 0.5, 0.9, 1.0], 0, std::ptr::null());
            // consoles want the back buffer in COMMON / PRESENT for PresentX
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: transmute_copy(&bb),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                    }),
                },
            }]);
            list.Close().map_err(e)?;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);

            // plane 0: { u64 frame token; u32 resource count @8; ID3D12Resource** @0x10 }
            let resources = [bb.as_raw()];
            let mut plane = [0u8; 0x40];
            plane[..8].copy_from_slice(&token.to_le_bytes());
            plane[8..12].copy_from_slice(&1u32.to_le_bytes());
            plane[0x10..0x18].copy_from_slice(&(resources.as_ptr() as u64).to_le_bytes());
            let present: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const u8) -> HRESULT = std::mem::transmute(slot(queue.as_raw(), 30));
            let hr = present(queue.as_raw(), 1, plane.as_ptr(), std::ptr::null());
            ensure!(hr == 0, "PresentX hr={hr:#x}");
            let (mut n, mut stats) = (1u32, [0u8; 0x40]);
            let hr = statistics(d, token, 0, &mut n, stats.as_mut_ptr());
            ensure!(hr == 0 && n == 1, "GetFrameStatisticsX hr={hr:#x} count={n}");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
            wait(&queue, &fence, 1)?;
            Ok(())
        })();
        let stop = set_interval(d, std::ptr::null_mut(), 0, 0, 0);
        res?;
        ensure!(stop >= 0, "stopping the frame clock");
        Ok(())
    }
}

/// The console device's own methods as a title calls them: a buffer placed at the title's
/// graphics memory (#44 CreatePlacedResourceX) is copied by a DMA queue (#48
/// CreateCommandQueueX), allocator (#9) and list (#73 CreateCommandListX, the DMA list's IID)
/// into a readback buffer; page pools (#74 / #75), SetDebugFlagsX (#52),
/// SerializePipelineStateX (#56) and an extension XCL does not model (#47).
pub fn console_device_methods() -> Result<(), String> {
    const DMA: u32 = 0x10;
    const IID_DMA_LIST: windows::core::GUID = windows::core::GUID::from_u128(0x30bc77b9_8379_4b25_95de_b8588c42275d);
    type Create = unsafe extern "system" fn(*mut c_void, *const u32, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
    type PlacedX = unsafe extern "system" fn(*mut c_void, u64, *const D3D12_RESOURCE_DESC, u32, *const c_void, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
    type ListX = unsafe extern "system" fn(*mut c_void, *const u32, *mut c_void, *mut c_void, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
    unsafe {
        let dev = device()?;
        let d = dev.as_raw();
        let slot = |obj: *mut c_void, i: usize| *(*(obj as *const *const usize)).add(i);
        let n = 0x1_0000usize;
        let graphics = 1u64 << 29;
        let mem = crate::XMemAlloc(n, graphics) as *mut u32;
        ensure!(!mem.is_null() && (mem as usize).is_multiple_of(n), "graphics memory {mem:?}");
        for i in 0..n / 4 {
            *mem.add(i) = 0xC0DE_0000 | i as u32;
        }
        let buffer = |width: usize| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: width as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let placed: PlacedX = std::mem::transmute(slot(d, 44));
        let mut raw = std::ptr::null_mut();
        let hr = placed(d, mem as u64, &buffer(n), D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), &ID3D12Resource::IID, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreatePlacedResourceX hr={hr:#x}");
        let src = ID3D12Resource::from_raw(raw);
        ensure!(src.GetGPUVirtualAddress() == mem as u64, "placed buffer at {:#x}, not the title's {mem:?}", src.GetGPUVirtualAddress());
        let mut dst: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buffer(n),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut dst,
        )
        .map_err(e)?;
        let dst = dst.ok_or("no readback buffer")?;

        let queue_x: Create = std::mem::transmute(slot(d, 48));
        let mut raw = std::ptr::null_mut();
        let hr = queue_x(d, [DMA, 0, 0, 0].as_ptr(), &ID3D12CommandQueue::IID, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreateCommandQueueX (DMA) hr={hr:#x}");
        let queue = ID3D12CommandQueue::from_raw(raw);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(DMA as i32)).map_err(e)?;
        let list_x: ListX = std::mem::transmute(slot(d, 73));
        let mut raw = std::ptr::null_mut();
        let hr = list_x(d, [DMA, 0, 1, 0].as_ptr(), alloc.as_raw(), std::ptr::null_mut(), &IID_DMA_LIST, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreateCommandListX (DMA) hr={hr:#x}");
        let list = ID3D12GraphicsCommandList::from_raw(raw);
        // the DMA list's own vtable: Close #9, Reset #10, CopyBufferRegion #11 (then
        // CopyTextureRegion #12, CopyResource #13, CopyTiles #14)
        type CopyBuffer = unsafe extern "system" fn(*mut c_void, *mut c_void, u64, *mut c_void, u64, u64);
        std::mem::transmute::<usize, CopyBuffer>(slot(raw, 11))(raw, dst.as_raw(), 0, src.as_raw(), 0, n as u64);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 1)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        dst.Map(0, None, Some(&mut p)).map_err(e)?;
        let copied = std::slice::from_raw_parts(p as *const u32, n / 4).iter().enumerate().all(|(i, &v)| v == 0xC0DE_0000 | i as u32);
        dst.Unmap(0, None);
        ensure!(copied, "the DMA copy did not read the title's memory");
        drop(src);
        ensure!(*mem.add(7) == 0xC0DE_0007, "the title's memory went with the placed buffer");
        crate::XMemFree(mem as *mut c_void, graphics);

        let register: unsafe extern "system" fn(*mut c_void, u64, u32, *mut u32) -> HRESULT = std::mem::transmute(slot(d, 74));
        let unregister: unsafe extern "system" fn(*mut c_void, u64) -> HRESULT = std::mem::transmute(slot(d, 75));
        let mut pool = u32::MAX;
        ensure!(register(d, 0x2_0000_0000, 16, &mut pool) == 0 && pool != u32::MAX, "RegisterPagePoolX");
        ensure!(unregister(d, pool as u64) == 0 && unregister(d, pool as u64) < 0, "UnregisterPagePoolX");
        let debug_flags: unsafe extern "system" fn(*mut c_void, u32) -> HRESULT = std::mem::transmute(slot(d, 52));
        ensure!(debug_flags(d, 1) == 0, "SetDebugFlagsX");
        let serialize: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT = std::mem::transmute(slot(d, 56));
        ensure!(serialize(d, std::ptr::null_mut()) == 0x8007_0005u32 as i32, "SerializePipelineStateX");
        // #47 is PC CreatePipelineState: with no arguments it would fail or crash
        let unmodelled: unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> HRESULT = std::mem::transmute(slot(d, 47));
        ensure!(unmodelled(d, 0, 0, 0) == 0, "console extension #47");
        Ok(())
    }
}

/// The console command list's memory methods as a title calls them, on its own graphics
/// memory: Write32BitValueBottomOfPipeX #69, Write64BitValueTopOfPipeX #89, a GPU timestamp
/// (#70), FillMemoryWith32BitValueX #61, CopyMemoryToMemoryX #60, Wait32BitValueX #71,
/// Wait64BitValueX #72, and KickoffX #93.
pub fn console_list_methods() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let graphics = 1u64 << 29;
        let mem = crate::XMemAlloc(0x1_0000, graphics) as *mut u8;
        ensure!(!mem.is_null(), "graphics memory");
        let at = |o: usize| mem as u64 + o as u64;
        for i in 0..64 {
            *mem.add(0x1000 + i) = i as u8 ^ 0x5A;
        }
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let l = list.as_raw();
        let slot = |i: usize| *(*(l as *const *const usize)).add(i);
        let write32_bottom: unsafe extern "system" fn(*mut c_void, u64, u32, u32, u32) = std::mem::transmute(slot(69));
        let write64_bottom: unsafe extern "system" fn(*mut c_void, u64, u64, u32, u32) = std::mem::transmute(slot(70));
        let write64_top: unsafe extern "system" fn(*mut c_void, u64, u64, u32) = std::mem::transmute(slot(89));
        let fill32: unsafe extern "system" fn(*mut c_void, u64, u64, u32) = std::mem::transmute(slot(61));
        let copy: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u32) = std::mem::transmute(slot(60));
        let wait32: unsafe extern "system" fn(*mut c_void, u64, u32, u32, u32, u32) = std::mem::transmute(slot(71));
        let wait64: unsafe extern "system" fn(*mut c_void, u64, u32, u64, u64, u32) = std::mem::transmute(slot(72));
        let kickoff: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(slot(93));

        // 1. Memory writes, timestamp, fill, copy, and same-list write-then-wait (must not deadlock)
        write32_bottom(l, at(0), 0x1234_5678, 0, 0);
        write64_top(l, at(8), 0x1122_3344_5566_7788, 0);
        write64_bottom(l, at(0x10), 0, 0, 1); // GPU timestamp
        fill32(l, at(0x100), 0x100, 0xABCD_EF01);
        copy(l, at(0x2000), at(0x1000), 64, 0);
        wait32(l, at(0), 3, 0x1234_5678, u32::MAX, 0);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 1)?;
        let r32 = |o: usize| (mem.add(o) as *const u32).read();
        let r64 = |o: usize| (mem.add(o) as *const u64).read();
        ensure!(r32(0) == 0x1234_5678, "Write32BitValueBottomOfPipeX: {:#x}", r32(0));
        ensure!(r64(8) == 0x1122_3344_5566_7788, "Write64BitValueTopOfPipeX: {:#x}", r64(8));
        ensure!(r64(0x10) != 0, "no GPU timestamp");
        ensure!((0..64).all(|i| r32(0x100 + 4 * i) == 0xABCD_EF01) && r32(0x200) == 0 && r32(0xFC) == 0, "FillMemoryWith32BitValueX");
        ensure!((0..64).all(|i| *mem.add(0x2000 + i) == i as u8 ^ 0x5A) && *mem.add(0x2040) == 0, "CopyMemoryToMemoryX");

        // 2. KickoffX (#93): flushes work to the GPU BEFORE Close() is called, then continues recording
        alloc.Reset().map_err(e)?;
        list.Reset(&alloc, None).map_err(e)?;
        write32_bottom(l, at(0x400), 0x1111_2222, 0, 0);
        kickoff(l);
        // Wait for the kicked-off write on the GPU while the list is still open!
        let t0 = std::time::Instant::now();
        while (mem.add(0x400) as *const std::sync::atomic::AtomicU32).as_ref().unwrap().load(std::sync::atomic::Ordering::Acquire) != 0x1111_2222
            && t0.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        ensure!(r32(0x400) == 0x1111_2222, "KickoffX did not flush work before Close ({:#x})", r32(0x400));
        // Continue recording on the same list and submit normally via Close + ExecuteCommandLists
        write32_bottom(l, at(0x404), 0x3333_4444, 0, 0);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 2)?;
        ensure!(r32(0x404) == 0x3333_4444, "post-KickoffX write failed ({:#x})", r32(0x404));

        // 3. Cross-thread Wait32BitValueX (#71, masked EQUAL) + Wait64BitValueX (#72, GREATER_EQUAL)
        alloc.Reset().map_err(e)?;
        list.Reset(&alloc, None).map_err(e)?;
        wait32(l, at(0x300), 3, 0x0000_00AA, 0x0000_00FF, 0);
        wait64(l, at(0x308), 7, 0x0000_0002_0000_0000, u64::MAX, 0);
        write32_bottom(l, at(0x310), 0x55AA_55AA, 0, 0);
        list.Close().map_err(e)?;
        let addr32 = at(0x300) as usize;
        let addr64 = at(0x308) as usize;
        let writer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(15));
            (*(addr32 as *const std::sync::atomic::AtomicU32)).store(0xDEAD_00AA, std::sync::atomic::Ordering::Release);
            (*(addr64 as *const std::sync::atomic::AtomicU64)).store(0x0000_0003_0000_0000, std::sync::atomic::Ordering::Release);
        });
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 3)?;
        writer.join().map_err(|_| "wait writer thread panicked".to_string())?;
        ensure!(r32(0x300) == 0xDEAD_00AA && r64(0x308) == 0x0000_0003_0000_0000, "wait cells not written");
        ensure!(r32(0x310) == 0x55AA_55AA, "post-Wait32/64 write failed ({:#x})", r32(0x310));

        crate::XMemFree(mem as *mut c_void, graphics);
        Ok(())
    }
}

/// Raytracing D3D12.X surface: CheckFeatureSupport(D3D12_OPTIONS5), console heap SRV for a
/// raytracing acceleration structure (writes an RDNA2 BVH descriptor instead of zeros), and
/// the command list extensions `XboxBvhBuilder.dll` uses on default title lists (#148
/// AllocateUploadMemory, #146 UAV barrier).
pub fn raytracing_surface() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        // 1. CheckFeatureSupport(D3D12_OPTIONS5) — both PC size (12 B) and oversized console probe (16 B)
        let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &mut opts5 as *mut _ as *mut c_void, std::mem::size_of_val(&opts5) as u32)
            .map_err(e)?;
        let mut wide = [0xAAu8; 16];
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, wide.as_mut_ptr() as *mut c_void, wide.len() as u32)
            .map_err(e)?;
        ensure!(wide[12..16] == [0, 0, 0, 0], "oversized OPTIONS5 tail not zeroed: {:?}", &wide[12..16]);

        // 2. Acceleration-structure SRV in a console CBV/SRV/UAV descriptor heap (pResource = NULL)
        let graphics = 1u64 << 29;
        let tlas = crate::XMemAlloc(0x1_0000, graphics);
        ensure!(!tlas.is_null(), "graphics memory for TLAS");
        let tlas_va = tlas as u64;
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(e)?;
        // console heap start handle returns in rax
        let start: unsafe extern "system" fn(*mut c_void) -> usize = transmute_copy(&(*(*(heap.as_raw() as *const *const usize)).add(9)));
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start(heap.as_raw()) };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: tlas_va },
            },
        };
        dev.CreateShaderResourceView(None, Some(&srv), cpu);
        let words = std::slice::from_raw_parts(cpu.ptr as *const u32, 4);
        let expected = [(tlas_va >> 8) as u32, (((tlas_va >> 40) as u32) & 0xFF) | (1 << 31), u32::MAX, 0];
        ensure!(words == expected, "AS SRV slot {words:x?} != expected {expected:x?}");
        crate::XMemFree(tlas, graphics);

        // 3. Default title command list: #148 AllocateUploadMemory + #146 UAV barrier
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let l = list.as_raw();
        let slot = |i: usize| *(*(l as *const *const usize)).add(i);
        let alloc_upload: unsafe extern "system" fn(*mut c_void, usize, usize) -> *mut u8 = std::mem::transmute(slot(148));
        let uav_barrier: unsafe extern "system" fn(*mut c_void, u32, *const c_void) = std::mem::transmute(slot(146));
        let up = alloc_upload(l, 256, 256);
        ensure!(!up.is_null() && (up as usize).is_multiple_of(256), "AllocateUploadMemory (#148) returned {up:?}");
        *(up as *mut u32) = 0x5A5A_1234;
        ensure!(*(up as *const u32) == 0x5A5A_1234, "upload memory not writable");
        uav_barrier(l, 1, std::ptr::null());
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 1)?;
        alloc.Reset().map_err(e)?;
        Ok(())
    }
}

/// The console's DirectStorage as a title uses it: a file queue reads the start of the title's
/// own executable into memory, a status array entry reports it and a D3D12 fence is signalled.
pub fn direct_storage() -> Result<(), String> {
    const IID_FACTORY: windows::core::GUID = windows::core::GUID::from_u128(0xfb3b54ac_bf82_463a_b777_73195837864d);
    const IID_QUEUE: windows::core::GUID = windows::core::GUID::from_u128(0xa3f6649a_8c2c_4b3a_8b6e_10186a04b9bc);
    const IID_FILE: windows::core::GUID = windows::core::GUID::from_u128(0x43704f9f_27c6_4a9f_8815_d16527894df4);
    const IID_STATUS: windows::core::GUID = windows::core::GUID::from_u128(0xcd66d752_2786_44f7_9070_e2a6fdad3f1b);
    type Create = unsafe extern "system" fn(*mut c_void, *const u8, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
    type Open = unsafe extern "system" fn(*mut c_void, *const u16, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
    type Status = unsafe extern "system" fn(*mut c_void, u32, *const u8, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
    struct ComPtr(*mut c_void);
    impl ComPtr {
        fn new(p: *mut c_void, name: &str) -> Result<Self, String> {
            ensure!(!p.is_null(), "{name}");
            Ok(Self(p))
        }
    }
    impl Drop for ComPtr {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                        std::mem::transmute(*(*(self.0 as *const *const usize)).add(2));
                    release(self.0);
                }
            }
        }
    }
    struct PhysPage([u64; 1], bool);
    impl PhysPage {
        unsafe fn alloc() -> Result<Self, String> {
            let mut count = 1usize;
            let mut page = [0u64; 1];
            ensure!(crate::XMemAllocatePhysicalPages(0, &mut count, page.as_mut_ptr()) != 0, "XMemAllocatePhysicalPages");
            Ok(Self(page, false))
        }
        unsafe fn free(mut self) -> Result<(), String> {
            self.1 = true;
            ensure!(crate::XMemFreePhysicalPages(1, self.0.as_mut_ptr()) != 0, "XMemFreePhysicalPages");
            Ok(())
        }
    }
    impl Drop for PhysPage {
        fn drop(&mut self) {
            if !self.1 {
                unsafe {
                    let _ = crate::XMemFreePhysicalPages(1, self.0.as_mut_ptr());
                }
            }
        }
    }
    unsafe {
        let slot = |obj: *mut c_void, i: usize| *(*(obj as *const *const usize)).add(i);
        let mut f = std::ptr::null_mut();
        ensure!(DStorageGetFactory(&IID_FACTORY, &mut f) == 0, "DStorageGetFactory");
        let f = ComPtr::new(f, "DStorageGetFactory")?;
        let mut desc = [0u8; 0x18];
        desc[8..10].copy_from_slice(&16u16.to_le_bytes()); // file source, capacity 16, normal priority
        let mut q = std::ptr::null_mut();
        ensure!(std::mem::transmute::<usize, Create>(slot(f.0, 3))(f.0, desc.as_ptr(), &IID_QUEUE, &mut q) == 0, "CreateQueue");
        let q = ComPtr::new(q, "CreateQueue")?;
        let exe = std::env::current_exe().map_err(|x| x.to_string())?;
        let path: Vec<u16> = exe.to_string_lossy().encode_utf16().chain([0]).collect();
        let mut file = std::ptr::null_mut();
        ensure!(std::mem::transmute::<usize, Open>(slot(f.0, 4))(f.0, path.as_ptr(), &IID_FILE, &mut file) == 0, "OpenFile {}", exe.display());
        let file = ComPtr::new(file, "OpenFile")?;
        let mut status = std::ptr::null_mut();
        ensure!(std::mem::transmute::<usize, Status>(slot(f.0, 5))(f.0, 1, std::ptr::null(), &IID_STATUS, &mut status) == 0, "CreateStatusArray");
        let status = ComPtr::new(status, "CreateStatusArray")?;

        let mut dst = vec![0u8; 512];
        let mut req = [0u8; 0x48];
        req[8..16].copy_from_slice(&(dst.as_mut_ptr() as u64).to_le_bytes());
        req[0x18..0x1C].copy_from_slice(&512u32.to_le_bytes());
        req[0x20..0x28].copy_from_slice(&(file.0 as u64).to_le_bytes());
        req[0x30..0x34].copy_from_slice(&512u32.to_le_bytes());
        let dev = device()?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *const u8)>(slot(q.0, 3))(q.0, req.as_ptr());
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *mut c_void, u32)>(slot(q.0, 4))(q.0, status.0, 0);
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *mut c_void, u64)>(slot(q.0, 5))(q.0, fence.as_raw(), 1);
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void)>(slot(q.0, 6))(q.0);
        let ev = EventHandle::new()?;
        fence.SetEventOnCompletion(1, ev.handle()).map_err(e)?;
        ensure!(WaitForSingleObject(ev.0, 5000) == 0, "the DirectStorage fence was not signalled");
        let hr = std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, u32) -> HRESULT>(slot(status.0, 4))(status.0, 0);
        ensure!(hr == 0, "request status {hr:#x}");
        let disk = std::fs::read(&exe).map_err(|x| x.to_string())?;
        ensure!(dst[..] == disk[..512] && dst.starts_with(b"MZ"), "the read bytes are not the file's");

        // Also check physical-page destination (bit 21) and CancelRequestsWithTag.
        let mut page = PhysPage::alloc()?;
        let mut req_phys = [0u8; 0x48];
        req_phys[0..8].copy_from_slice(&(1u64 << 21).to_le_bytes()); // DestinationIsPhysicalPages
        req_phys[8..16].copy_from_slice(&(page.0.as_ptr() as u64).to_le_bytes());
        req_phys[0x10..0x12].copy_from_slice(&0x80u16.to_le_bytes()); // DestinationPageOffset
        req_phys[0x18..0x1C].copy_from_slice(&512u32.to_le_bytes());
        req_phys[0x20..0x28].copy_from_slice(&(file.0 as u64).to_le_bytes());
        req_phys[0x30..0x34].copy_from_slice(&512u32.to_le_bytes());
        let mut cancelled_dst = vec![0u8; 512];
        let mut req_cancel = [0u8; 0x48];
        req_cancel[8..16].copy_from_slice(&(cancelled_dst.as_mut_ptr() as u64).to_le_bytes());
        req_cancel[0x18..0x1C].copy_from_slice(&512u32.to_le_bytes());
        req_cancel[0x20..0x28].copy_from_slice(&(file.0 as u64).to_le_bytes());
        req_cancel[0x30..0x34].copy_from_slice(&512u32.to_le_bytes());
        req_cancel[0x38..0x40].copy_from_slice(&0xCAFE_0001u64.to_le_bytes()); // CancellationTag
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *const u8)>(slot(q.0, 3))(q.0, req_phys.as_ptr());
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *const u8)>(slot(q.0, 3))(q.0, req_cancel.as_ptr());
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, u64, u64)>(slot(q.0, 7))(q.0, 0xFFFF_0000, 0xCAFE_0000);
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *mut c_void, u32)>(slot(q.0, 4))(q.0, status.0, 0);
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, *mut c_void, u64)>(slot(q.0, 5))(q.0, fence.as_raw(), 2);
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void)>(slot(q.0, 6))(q.0);
        fence.SetEventOnCompletion(2, ev.handle()).map_err(e)?;
        ensure!(WaitForSingleObject(ev.0, 5000) == 0, "the second DirectStorage fence was not signalled");
        let hr2 = std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void, u32) -> HRESULT>(slot(status.0, 4))(status.0, 0);
        ensure!(hr2 == 0, "physical-page request status {hr2:#x}");
        let mapped = crate::XMemMapPhysicalPages(std::ptr::null_mut(), 1, page.0.as_mut_ptr());
        ensure!(!mapped.is_null(), "XMemMapPhysicalPages");
        let phys_slice = std::slice::from_raw_parts((mapped as *const u8).add(0x80), 512);
        ensure!(phys_slice == &disk[..512], "physical-page DirectStorage read mismatch");
        ensure!(cancelled_dst.iter().all(|&b| b == 0), "cancelled DirectStorage request wrote to destination");
        page.free()?;
        Ok(())
    }
}

/// A texture in the title's own memory, as a console title streams one: its layout from the
/// layout library (64KB_S_X, 2 slices x 2 mips), texels written there tiled, the texture placed
/// at that address (#44) - the GPU reads every subresource; then one subresource is written
/// again and the next ExecuteCommandLists sees the new texels.
pub fn title_texture() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let slot = |obj: *mut c_void, i: usize| *(*(obj as *const *const usize)).add(i);
        let mut desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 80,
            Height: 40,
            DepthOrArraySize: 2,
            MipLevels: 2,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.Layout = D3D12_TEXTURE_LAYOUT(0x119); // the console's 64KB_S_X
        let mut xg = std::ptr::null_mut();
        ensure!(XGCreateTextureComputer(&desc, &mut xg) == 0, "XGCreateTextureComputer");
        let size = std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> u64>(slot(xg, 3))(xg) as usize;
        type Into = unsafe extern "system" fn(*mut c_void, *mut u8, u32, u32, *const u8, u32, u32) -> HRESULT;
        let into: Into = std::mem::transmute(slot(xg, 8));
        let graphics = 1u64 << 29;
        let mem = crate::XMemAlloc(size, graphics) as *mut u8;
        ensure!(!mem.is_null(), "graphics memory");
        let dims = |mip: u32| ((80u32 >> mip).max(1), (40u32 >> mip).max(1));
        let texels = |sub: u32, salt: u8| -> Vec<u8> {
            let (w, h) = dims(sub % 2);
            (0..w * h * 4).map(|i| (i as u8).wrapping_mul(7).wrapping_add(sub as u8 * 31).wrapping_add(salt)).collect()
        };
        for sub in 0..4u32 {
            let (w, h) = dims(sub % 2);
            ensure!(into(xg, mem, 0, sub, texels(sub, 0).as_ptr(), w * 4, w * h * 4) == 0, "CopyIntoSubresource {sub}");
        }
        type PlacedX = unsafe extern "system" fn(*mut c_void, u64, *const D3D12_RESOURCE_DESC, u32, *const c_void, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
        let mut raw = std::ptr::null_mut();
        let hr = std::mem::transmute::<usize, PlacedX>(slot(dev.as_raw(), 44))(dev.as_raw(), mem as u64, &desc, 0, std::ptr::null(), &ID3D12Resource::IID, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreatePlacedResourceX texture hr={hr:#x}");
        let tex = ID3D12Resource::from_raw(raw);

        // all four subresources through the GPU
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let pc = tex.GetDesc();
        let mut fps = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 4];
        let mut total = 0u64;
        dev.GetCopyableFootprints(&pc, 0, 4, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut total));
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(e)?;
        let rb = rb.ok_or("no readback buffer")?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        list.Close().map_err(e)?;
        let mut step = 0u64;
        let mut read_back = |want: &dyn Fn(u32) -> Vec<u8>| -> Result<(), String> {
            alloc.Reset().map_err(e)?;
            list.Reset(&alloc, None).map_err(e)?;
            for (sub, fp) in fps.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&tex), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().map_err(e)?;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
            step += 1;
            wait(&queue, &fence, step)?;
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).map_err(e)?;
            let mut result = Ok(());
            for (sub, fp) in fps.iter().enumerate() {
                let (w, h) = dims(sub as u32 % 2);
                let exp = want(sub as u32);
                for y in 0..h as usize {
                    let got = std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), w as usize * 4);
                    if got != &exp[y * w as usize * 4..(y + 1) * w as usize * 4] {
                        result = Err(format!("subresource {sub} row {y} differs from the title memory"));
                        break;
                    }
                }
            }
            rb.Unmap(0, None);
            result
        };
        read_back(&|sub| texels(sub, 0))?;
        // stream new texels into slice 1 mip 0 (subresource 2) and run work: the GPU sees them
        ensure!(into(xg, mem, 0, 2, texels(2, 99).as_ptr(), 80 * 4, 80 * 40 * 4) == 0, "CopyIntoSubresource again");
        read_back(&|sub| texels(sub, if sub == 2 { 99 } else { 0 }))?;
        drop(tex);
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> u32>(slot(xg, 1))(xg);
        crate::XMemFree(mem as *mut c_void, graphics);
        Ok(())
    }
}

/// A texture placed in a streaming pool before any memory is there: address space reserved,
/// the texture placed at it (#44, its texels read as zero), then a physical page mapped in and
/// the texels written – the GPU sees them after the next ExecuteCommandLists.
pub fn streamed_texture() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let slot = |obj: *mut c_void, i: usize| *(*(obj as *const *const usize)).add(i);
        let mut desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.Layout = D3D12_TEXTURE_LAYOUT(0x105); // 4KB_S: 16 KB
        let mut xg = std::ptr::null_mut();
        ensure!(XGCreateTextureComputer(&desc, &mut xg) == 0, "XGCreateTextureComputer");
        let page = 0x1_0000usize;
        let va = crate::XMemVirtualAlloc(std::ptr::null(), page, 0x2000, 1u64 << 29, 4) as usize;
        ensure!(va != 0, "reservation");
        type PlacedX = unsafe extern "system" fn(*mut c_void, u64, *const D3D12_RESOURCE_DESC, u32, *const c_void, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
        let mut raw = std::ptr::null_mut();
        let hr = std::mem::transmute::<usize, PlacedX>(slot(dev.as_raw(), 44))(dev.as_raw(), va as u64, &desc, 0, std::ptr::null(), &ID3D12Resource::IID, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreatePlacedResourceX hr={hr:#x}");
        let tex = ID3D12Resource::from_raw(raw);
        let (mut n, mut pages) = (1usize, [0u64; 1]);
        ensure!(crate::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()) != 0, "XMemAllocatePhysicalPages");
        ensure!(crate::XMemMapPhysicalPages(va as *mut c_void, 1, pages.as_mut_ptr()) as usize == va, "XMemMapPhysicalPages");
        let texels: Vec<u8> = (0..64 * 64 * 4).map(|i: u32| (i as u8).wrapping_mul(13) ^ 0x5A).collect();
        type Into = unsafe extern "system" fn(*mut c_void, *mut u8, u32, u32, *const u8, u32, u32) -> HRESULT;
        ensure!(std::mem::transmute::<usize, Into>(slot(xg, 8))(xg, va as *mut u8, 0, 0, texels.as_ptr(), 256, 256 * 64) == 0, "CopyIntoSubresource");

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&tex.GetDesc(), 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(e)?;
        let rb = rb.ok_or("no readback buffer")?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&tex), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 1)?;
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(e)?;
        let same = (0..64usize).all(|y| std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), 256) == &texels[y * 256..(y + 1) * 256]);
        rb.Unmap(0, None);
        ensure!(same, "the texture does not have the texels mapped in after it was placed");
        drop(tex);
        ensure!(crate::XMemFreePhysicalPages(1, pages.as_mut_ptr()) != 0, "XMemFreePhysicalPages");
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> u32>(slot(xg, 1))(xg);
        crate::VirtualFree(va as *mut c_void, 0, 0x8000);
        Ok(())
    }
}

pub fn render_triangle() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let rs = root_signature(&dev)?;
        let vs = compile(
            "float4 main(uint id : SV_VertexID) : SV_Position { float2 p[3] = { float2(-0.5,-0.5), float2(0.0,0.5), float2(0.5,-0.5) }; return float4(p[id], 0, 1); }",
            "vs_5_0",
        )?;
        let ps = compile("float4 main() : SV_Target { return float4(0, 1, 0, 1); }", "ps_5_0")?;
        // console pipeline creation: #60 CreateGraphicsPipelineStateX, then #55
        // CreateDerivedPipelineStateX with a blend patch writing only green and alpha
        let slot = |obj: *mut c_void, i: usize| *(*(obj as *const *const usize)).add(i);
        type PsoX = unsafe extern "system" fn(*mut c_void, *const D3D12_GRAPHICS_PIPELINE_STATE_DESC, u32, *const c_void, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
        type DerivedX = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const u8, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
        let desc = pso_desc(&rs, bc(&vs), bc(&ps));
        let mut raw = std::ptr::null_mut();
        let hr = std::mem::transmute::<usize, PsoX>(slot(dev.as_raw(), 60))(dev.as_raw(), &desc, 0, std::ptr::null(), &ID3D12PipelineState::IID, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreateGraphicsPipelineStateX hr={hr:#x}");
        let pso = ID3D12PipelineState::from_raw(raw);
        let mut blend = desc.BlendState;
        blend.RenderTarget[0].RenderTargetWriteMask = (D3D12_COLOR_WRITE_ENABLE_GREEN.0 | D3D12_COLOR_WRITE_ENABLE_ALPHA.0) as u8;
        let mut patch = [0u8; 0x14C];
        patch[0] = 2; // blend state
        std::ptr::copy_nonoverlapping(&blend as *const _ as *const u8, patch.as_mut_ptr().add(4), std::mem::size_of::<D3D12_BLEND_DESC>());
        let derive: DerivedX = std::mem::transmute(slot(dev.as_raw(), 55));
        let mut raw = std::ptr::null_mut();
        let hr = derive(dev.as_raw(), pso.as_raw(), 1, patch.as_ptr(), &ID3D12PipelineState::IID, &mut raw);
        ensure!(hr >= 0 && !raw.is_null(), "CreateDerivedPipelineStateX hr={hr:#x}");
        let derived = ID3D12PipelineState::from_raw(raw);
        // indices in the title's graphics memory: [0 1 2 0 1 2] for the bound index buffer,
        // [0 1 2] at +16 for DrawIndexedX
        let graphics = 1u64 << 29;
        let ib = crate::XMemAlloc(0x1_0000, graphics) as *mut u16;
        ensure!(!ib.is_null(), "graphics memory");
        for (i, v) in [0u16, 1, 2, 0, 1, 2, 0, 0, 0, 1, 2].into_iter().enumerate() {
            *ib.add(i) = v;
        }
        patch[0] = 4;
        let hr = derive(dev.as_raw(), pso.as_raw(), 1, patch.as_ptr(), &ID3D12PipelineState::IID, &mut raw);
        ensure!(hr < 0, "an unknown patch part must fail (hr={hr:#x})");

        let (w, h) = (64u32, 64u32);
        let tex = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w as u64,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0, 0.0, 1.0, 1.0] } };
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &tex,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&clear),
            &mut rt,
        )
        .map_err(e)?;
        let rt = rt.ok_or("no render target")?;
        let heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
            .map_err(e)?;
        // D3D12.X: the heap start handle comes back in rax (the PC method returns it through
        // a hidden pointer) – console code calls slot 9 like this
        let start: unsafe extern "system" fn(*mut c_void) -> usize = transmute_copy(&(*(*(heap.as_raw() as *const *const usize)).add(9)));
        let rtv = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start(heap.as_raw()) };
        dev.CreateRenderTargetView(&rt, None, rtv);

        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&tex, 0, 1, 0, Some(&mut layout), None, None, Some(&mut total));
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(e)?;
        let rb = rb.ok_or("no readback buffer")?;
        // occlusion queries: 2 around one draw, 0 around two draws with 1 nested inside it
        let mut queries: Option<ID3D12QueryHeap> = None;
        dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION, Count: 3, NodeMask: 0 }, &mut queries).map_err(e)?;
        let queries = queries.ok_or("no query heap")?;
        let mut results: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC { Width: 24, ..buf_desc },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut results,
        )
        .map_err(e)?;
        let results = results.ok_or("no query results buffer")?;

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(e)?;
        // console slots (OMSetRenderTargets #50, ClearRenderTargetView #52, SetGraphicsRootSignature #34)
        xslot::<OmSetRenderTargets>(&list, X_OM_SET_RENDER_TARGETS)(list.as_raw(), 1, &rtv, 0, std::ptr::null());
        xslot::<ClearRtv>(&list, X_CLEAR_RENDER_TARGET_VIEW)(list.as_raw(), rtv, &[0.0, 0.0, 1.0, 1.0], 0, std::ptr::null());
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
        xslot::<SetRootSignature>(&list, X_SET_GRAPHICS_ROOT_SIGNATURE)(list.as_raw(), rs.as_raw());
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.DrawInstanced(3, 1, 0, 0);
        // the derived pipeline in the top-left quadrant: green over the blue clear = cyan
        list.SetPipelineState(&derived);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: (w / 2) as f32, Height: (h / 2) as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.DrawInstanced(3, 1, 0, 0);
        // console index buffers at title addresses: IASetIndexBuffer #47 binds [0 1 2 0 1 2];
        // DrawIndexedX #81 draws [0 1 2] from +16 in the top-right quadrant; the bound buffer
        // comes back for indices 3–5 in the bottom-left one
        type SetIb = unsafe extern "system" fn(*mut c_void, *const D3D12_INDEX_BUFFER_VIEW);
        type DrawIndexedX = unsafe extern "system" fn(*mut c_void, u64, DXGI_FORMAT, u32, u32);
        list.SetPipelineState(&pso);
        xslot::<SetIb>(&list, 47)(list.as_raw(), &D3D12_INDEX_BUFFER_VIEW { BufferLocation: ib as u64, SizeInBytes: 12, Format: DXGI_FORMAT_R16_UINT });
        let half = |x: u32, y: u32| D3D12_VIEWPORT { TopLeftX: x as f32, TopLeftY: y as f32, Width: (w / 2) as f32, Height: (h / 2) as f32, MinDepth: 0.0, MaxDepth: 1.0 };
        list.RSSetViewports(&[half(w / 2, 0)]);
        xslot::<DrawIndexedX>(&list, 81)(list.as_raw(), ib as u64 + 16, DXGI_FORMAT_R16_UINT, 3, 0);
        list.RSSetViewports(&[half(0, h / 2)]);
        list.DrawIndexedInstanced(3, 1, 3, 0, 0);
        // the base triangle again over the whole target (same pixels), counted by occlusion
        // queries at the console slots BeginQuery #28 / EndQuery #29 / ResolveQueryData #30
        type Query = unsafe extern "system" fn(*mut c_void, *mut c_void, D3D12_QUERY_TYPE, u32);
        type Resolve = unsafe extern "system" fn(*mut c_void, *mut c_void, D3D12_QUERY_TYPE, u32, u32, *mut c_void, u64);
        let (begin, end) = (xslot::<Query>(&list, 28), xslot::<Query>(&list, 29));
        let (l, q, occ) = (list.as_raw(), queries.as_raw(), D3D12_QUERY_TYPE_OCCLUSION);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        begin(l, q, occ, 2);
        list.DrawInstanced(3, 1, 0, 0);
        end(l, q, occ, 2);
        begin(l, q, occ, 0);
        list.DrawInstanced(3, 1, 0, 0);
        begin(l, q, occ, 1);
        list.DrawInstanced(3, 1, 0, 0);
        end(l, q, occ, 1);
        end(l, q, occ, 0);
        xslot::<Resolve>(&list, 30)(l, q, occ, 0, 3, results.as_raw(), 0);
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: transmute_copy(&rt),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        list.ResourceBarrier(&[barrier]);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(&rb),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: layout },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(&rt),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 1)?;

        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(e)?;
        let pitch = layout.Footprint.RowPitch as usize;
        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = y * pitch + x * 4;
            let s = std::slice::from_raw_parts((p as *const u8).add(o), 4);
            [s[0], s[1], s[2], s[3]]
        };
        let center = px(32, 32);
        let corner = px(1, 1);
        let quadrant = px(16, 16);
        let (top_right, bottom_left) = (px(48, 16), px(16, 48));
        rb.Unmap(0, None);
        crate::XMemFree(ib as *mut c_void, graphics);
        let mut r = std::ptr::null_mut();
        results.Map(0, None, Some(&mut r)).map_err(e)?;
        let counts = std::slice::from_raw_parts(r as *const u64, 3).to_vec();
        results.Unmap(0, None);
        ensure!(counts[2] > 0 && counts[1] == counts[2] && counts[0] > 0, "occlusion queries (outer, inner, single) = {counts:?}: expected the inner one = single and a visible outer one");
        ensure!(top_right == [0, 255, 0, 255], "top-right quadrant {top_right:?} (expected green: DrawIndexedX)");
        ensure!(bottom_left == [0, 255, 0, 255], "bottom-left quadrant {bottom_left:?} (expected green: the title's index buffer, bound again)");
        ensure!(center == [0, 255, 0, 255], "centre pixel {center:?} (expected green triangle)");
        ensure!(quadrant == [0, 255, 255, 255], "top-left quadrant {quadrant:?} (expected cyan: the derived pipeline writes green only)");
        ensure!(corner == [0, 0, 255, 255], "corner pixel {corner:?} (expected blue clear colour)");
        Ok(())
    }
}

/// Media Foundation `MFResetDXGIDeviceManagerX` with the console `d3d12_xs` device.
pub fn mfplat_reset_dxgi_device_manager() -> Result<(), String> {
    const MF_VERSION: u32 = 0x0002_0070;
    const MFSTARTUP_LITE: u32 = 1;
    const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;

    type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
    type CloseHandleFn = unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT;
    type OpenHandleFn = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
    type TestDeviceFn = unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT;

    unsafe {
        let dev = device()?;
        let hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
        ensure!(hr >= 0, "MFStartup hr={hr:#x}");

        let mut token: u32 = 0;
        let mut mgr: *mut c_void = std::ptr::null_mut();
        let hr = MFCreateDXGIDeviceManager(&mut token, &mut mgr);
        if hr < 0 || mgr.is_null() {
            let _ = MFShutdown();
            return Err(format!("MFCreateDXGIDeviceManager hr={hr:#x}"));
        }

        let vtbl = *(mgr as *const *const usize);
        let release: ReleaseFn = transmute_copy(&*vtbl.add(2));
        let close_handle: CloseHandleFn = transmute_copy(&*vtbl.add(3));
        let open_handle: OpenHandleFn = transmute_copy(&*vtbl.add(6));
        let test_device: TestDeviceFn = transmute_copy(&*vtbl.add(8));

        let res = (|| -> Result<(), String> {
            let hr_bad = MFResetDXGIDeviceManagerX(mgr, dev.as_raw(), token.wrapping_add(1));
            ensure!(hr_bad == E_INVALIDARG, "bad token expected E_INVALIDARG, got {hr_bad:#x}");

            let hr_ok = MFResetDXGIDeviceManagerX(mgr, dev.as_raw(), token);
            ensure!(hr_ok == 0, "MFResetDXGIDeviceManagerX hr={hr_ok:#x}");

            let mut handle: *mut c_void = std::ptr::null_mut();
            let hr_open = open_handle(mgr, &mut handle);
            ensure!(hr_open >= 0 && !handle.is_null(), "OpenDeviceHandle hr={hr_open:#x}");

            let hr_test = test_device(mgr, handle);
            let _ = close_handle(mgr, handle);
            ensure!(hr_test >= 0, "TestDevice hr={hr_test:#x}");
            Ok(())
        })();

        release(mgr);
        let _ = MFShutdown();
        res
    }
}

/// Dynamic depth bias via `RSSetDepthBiasX` (#94): draws a triangle at z = 0.5 against a depth
/// buffer cleared to 0.5 with `DepthFunc = LESS`, exercising native `ID3D12GraphicsCommandList9`
/// dynamic depth bias (when supported by the adapter) or the patched-PSO fallback:
/// 1. Top-left: no bias → rejected (blue).
/// 2. Top-right: fractional `RSSetDepthBiasX(-1000.5, 0, 0)` → passes (green).
/// 3. `KickoffX` (#93) while `-1000.5` is active, then bottom-right without re-calling
///    `RSSetDepthBiasX` → still passes (green).
/// 4. Bottom-left right half: `RSSetDepthBiasX(0, 0, 0)` resets bias to 0 → rejected (blue).
/// 5. Bottom-left left half: PSO with baked static `RasterizerState.DepthBias = -1000` (no
///    `RSSetDepthBiasX` override) → passes (green).
/// 6. Top-left corner: switch back to zero-bias PSO → previous PSO's static bias does not leak (blue).
pub fn dynamic_depth_bias() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let rs = root_signature(&dev)?;
        let vs = compile(
            "float4 main(uint id : SV_VertexID) : SV_Position { float2 p[3] = { float2(-1,-1), float2(-1,3), float2(3,-1) }; return float4(p[id], 0.5, 1.0); }",
            "vs_5_0",
        )?;
        let ps = compile("float4 main() : SV_Target { return float4(0, 1, 0, 1); }", "ps_5_0")?;
        let mut desc = pso_desc(&rs, bc(&vs), bc(&ps));
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.DepthStencilState.DepthEnable = true.into();
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&desc).map_err(e)?;

        let mut static_desc = desc;
        static_desc.RasterizerState.DepthBias = -1000;
        let pso_static_bias: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&static_desc).map_err(e)?;

        let (w, h) = (64u32, 64u32);
        let rt_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w as u64,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let rt_clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0, 0.0, 1.0, 1.0] } };
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &rt_desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&rt_clear),
            &mut rt,
        )
        .map_err(e)?;
        let rt = rt.ok_or("no render target")?;

        let ds_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w as u64,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let ds_clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 0.5, Stencil: 0 } },
        };
        let mut ds: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &ds_desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&ds_clear),
            &mut ds,
        )
        .map_err(e)?;
        let ds = ds.ok_or("no depth buffer")?;

        let rtv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
            .map_err(e)?;
        let dsv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() })
            .map_err(e)?;
        let heap_start = |heap: &ID3D12DescriptorHeap| -> D3D12_CPU_DESCRIPTOR_HANDLE {
            let f: unsafe extern "system" fn(*mut c_void) -> usize = transmute_copy(&(*(*(heap.as_raw() as *const *const usize)).add(9)));
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: f(heap.as_raw()) }
        };
        let rtv = heap_start(&rtv_heap);
        let dsv = heap_start(&dsv_heap);
        dev.CreateRenderTargetView(&rt, None, rtv);
        dev.CreateDepthStencilView(&ds, None, dsv);

        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&rt_desc, 0, 1, 0, Some(&mut layout), None, None, Some(&mut total));
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(e)?;
        let rb = rb.ok_or("no readback buffer")?;

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(e)?;
        let bind_pass_state = || {
            xslot::<OmSetRenderTargets>(&list, X_OM_SET_RENDER_TARGETS)(list.as_raw(), 1, &rtv, 0, &dsv);
            list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
            xslot::<SetRootSignature>(&list, X_SET_GRAPHICS_ROOT_SIGNATURE)(list.as_raw(), rs.as_raw());
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        };
        bind_pass_state();
        xslot::<ClearRtv>(&list, X_CLEAR_RENDER_TARGET_VIEW)(list.as_raw(), rtv, &[0.0, 0.0, 1.0, 1.0], 0, std::ptr::null());
        xslot::<ClearDsv>(&list, X_CLEAR_DEPTH_STENCIL_VIEW)(list.as_raw(), dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, 0, std::ptr::null());

        let vp = |x: u32, y: u32, vw: u32, vh: u32| D3D12_VIEWPORT { TopLeftX: x as f32, TopLeftY: y as f32, Width: vw as f32, Height: vh as f32, MinDepth: 0.0, MaxDepth: 1.0 };
        type KickoffX = unsafe extern "system" fn(*mut c_void);
        type SetDepthBiasX = unsafe extern "system" fn(*mut c_void, f32, f32, f32);
        let kickoff = xslot::<KickoffX>(&list, 93);
        let set_bias = xslot::<SetDepthBiasX>(&list, 94);

        // 1. Top-left: no bias (0.5 < 0.5 is false -> rejected -> blue)
        list.RSSetViewports(&[vp(0, 0, w / 2, h / 2)]);
        list.DrawInstanced(3, 1, 0, 0);

        // 2. Top-right: fractional negative depth bias pushes z < 0.5 -> passes -> green
        set_bias(list.as_raw(), -1000.5, 0.0, 0.0);
        list.RSSetViewports(&[vp(w / 2, 0, w / 2, h / 2)]);
        list.DrawInstanced(3, 1, 0, 0);

        // 3. KickoffX mid-list while depth-bias override is active: bias must persist after Reset()
        kickoff(list.as_raw());
        bind_pass_state();
        list.RSSetViewports(&[vp(w / 2, h / 2, w / 2, h / 2)]);
        list.DrawInstanced(3, 1, 0, 0);

        // 4. Bottom-left right half: reset depth bias to 0 -> rejected -> blue
        set_bias(list.as_raw(), 0.0, 0.0, 0.0);
        list.RSSetViewports(&[vp(w / 4, h / 2, w / 4, h / 2)]);
        list.DrawInstanced(3, 1, 0, 0);

        // 5. Bottom-left left half: PSO with baked static DepthBias = -1000 (no override) -> passes -> green
        list.SetPipelineState(&pso_static_bias);
        list.RSSetViewports(&[vp(0, h / 2, w / 4, h / 2)]);
        list.DrawInstanced(3, 1, 0, 0);

        // 6. Top-left corner: switch back to zero-bias PSO -> static bias must not leak -> blue
        list.SetPipelineState(&pso);
        list.RSSetViewports(&[vp(0, 0, w / 4, h / 4)]);
        list.DrawInstanced(3, 1, 0, 0);

        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: transmute_copy(&rt),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        }]);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(&rb),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: layout },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: transmute_copy(&rt),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        wait(&queue, &fence, 1)?;

        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(e)?;
        let pitch = layout.Footprint.RowPitch as usize;
        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = y * pitch + x * 4;
            let s = std::slice::from_raw_parts((p as *const u8).add(o), 4);
            [s[0], s[1], s[2], s[3]]
        };
        let (tl, tr, br, bl_reset, bl_static, tl_after) = (px(20, 20), px(48, 16), px(48, 48), px(24, 48), px(8, 48), px(8, 8));
        rb.Unmap(0, None);
        ensure!(tl == [0, 0, 255, 255], "top-left {tl:?} (expected blue: no depth bias, 0.5 < 0.5 fails)");
        ensure!(tr == [0, 255, 0, 255], "top-right {tr:?} (expected green: RSSetDepthBiasX(-1000.5) passes depth test)");
        ensure!(br == [0, 255, 0, 255], "bottom-right {br:?} (expected green: depth bias persists across KickoffX)");
        ensure!(bl_reset == [0, 0, 255, 255], "bottom-left reset {bl_reset:?} (expected blue: RSSetDepthBiasX(0) restores zero bias)");
        ensure!(bl_static == [0, 255, 0, 255], "bottom-left static {bl_static:?} (expected green: baked static DepthBias = -1000 preserved)");
        ensure!(tl_after == [0, 0, 255, 255], "top-left after static PSO {tl_after:?} (expected blue: static bias did not leak to zero-bias PSO)");
        Ok(())
    }
}

/// D3D12 buffer heaps + committed upload/default buffers on the sparse-window VRAM path:
/// persistent `Map()` pointer coherence across multiple submits, sub-64 KiB aliased placement
/// offsets, and explicit `Map(Some(range))` GPU→CPU readback.
pub fn sparse_buffer_heap_coherence() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;

        // 1. D3D12 DEFAULT buffer heap + two placed buffers (one at non-64KiB aliased offset 0x200)
        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: 0x4_0000,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0x1_0000,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        dev.CreateHeap(&heap_desc, &mut heap).map_err(e)?;
        let heap = heap.ok_or("CreateHeap")?;

        let buf_desc = |bytes: u64| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buf_a: Option<ID3D12Resource> = None;
        dev.CreatePlacedResource(&heap, 0, &buf_desc(0x1_0000), D3D12_RESOURCE_STATE_COMMON, None, &mut buf_a).map_err(e)?;
        let buf_a = buf_a.ok_or("placed buf_a")?;

        let mut buf_b: Option<ID3D12Resource> = None;
        dev.CreatePlacedResource(&heap, 0x200, &buf_desc(0x1000), D3D12_RESOURCE_STATE_COMMON, None, &mut buf_b).map_err(e)?;
        let buf_b = buf_b.ok_or("placed buf_b")?;

        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc(0x1000),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(e)?;
        let rb = rb.ok_or("readback")?;

        // Persistent Map() on buf_a: address must match GetGPUVirtualAddress(), and buf_b aliases at +0x200
        let mut ptr_a = std::ptr::null_mut();
        buf_a.Map(0, None, Some(&mut ptr_a)).map_err(e)?;
        ensure!(!ptr_a.is_null(), "buf_a Map null");
        ensure!(buf_a.GetGPUVirtualAddress() == ptr_a as u64, "buf_a VA != Map ptr");
        ensure!(buf_b.GetGPUVirtualAddress() == ptr_a as u64 + 0x200, "buf_b aliased VA mismatch");

        let mut ptr_rb = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut ptr_rb)).map_err(e)?;
        ensure!(!ptr_rb.is_null(), "rb Map null");

        // Submit 1: write pattern 1 via persistent pointer, copy buf_a[0..16] and buf_b[0..16] to rb
        let wa = ptr_a as *mut u32;
        *wa.add(0) = 0x1122_3344;
        *wa.add(1) = 0x5566_7788;
        *wa.add(0x200 / 4) = 0xCAFE_BABE;
        list.CopyBufferRegion(&rb, 0, &buf_a, 0, 16);
        list.CopyBufferRegion(&rb, 64, &buf_b, 0, 16);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 1)?;
        let wrb = ptr_rb as *const u32;
        ensure!(*wrb.add(0) == 0x1122_3344 && *wrb.add(1) == 0x5566_7788, "submit 1 buf_a readback mismatch");
        ensure!(*wrb.add(16) == 0xCAFE_BABE, "submit 1 aliased buf_b readback mismatch: {:#x}", *wrb.add(16));

        // Submit 2: mutate the SAME persistent pointer without re-mapping
        *wa.add(0) = 0xAABB_CCDD;
        *wa.add(0x200 / 4) = 0xDEAD_BEEF;
        alloc.Reset().map_err(e)?;
        list.Reset(&alloc, None).map_err(e)?;
        list.CopyBufferRegion(&rb, 0, &buf_a, 0, 16);
        list.CopyBufferRegion(&rb, 64, &buf_b, 0, 16);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 2)?;
        ensure!(*wrb.add(0) == 0xAABB_CCDD, "persistent Map mutation did not reach GPU: {:#x}", *wrb.add(0));
        ensure!(*wrb.add(16) == 0xDEAD_BEEF, "persistent Map aliased mutation did not reach GPU: {:#x}", *wrb.add(16));

        // Submit 3: committed UPLOAD buffer (sparse VRAM) -> GPU copy into buf_a -> explicit Map(read range) on buf_a
        let mut up: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc(0x1000),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut up,
        )
        .map_err(e)?;
        let up = up.ok_or("upload buf")?;
        let mut ptr_up = std::ptr::null_mut();
        up.Map(0, None, Some(&mut ptr_up)).map_err(e)?;
        *(ptr_up as *mut u32) = 0x4242_1337;
        alloc.Reset().map_err(e)?;
        list.Reset(&alloc, None).map_err(e)?;
        list.CopyBufferRegion(&buf_a, 0x100, &up, 0, 64);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 3)?;

        let mut ptr_a2 = std::ptr::null_mut();
        let read_range = D3D12_RANGE { Begin: 0x100, End: 0x140 };
        buf_a.Map(0, Some(&read_range), Some(&mut ptr_a2)).map_err(e)?;
        ensure!(ptr_a2 == ptr_a, "re-Map returned a different pointer");
        let got = *(ptr_a2 as *const u32).add(0x100 / 4);
        ensure!(got == 0x4242_1337, "explicit Map(read_range) readback mismatch: {got:#x}");
        Ok(())
    }
}

/// Coherent Window 3 headroom: general `CUSTOM + WRITE_BACK` buffer heaps and buffers must
/// route to sparse VRAM (windows 0..2), leaving Window 3 free for XCL's Arena pool and small
/// explicit readback buffers, while `COPY_DEST` buffers in a sparse `CUSTOM + WRITE_BACK` heap
/// (and oversized `READBACK` buffers that spill to sparse VRAM) still read back automatically
/// on `Map(0, NULL, ...)`.
pub fn coherent_window_headroom() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;

        let win = |va: u64| ((va >> 32) & 3) as u32;
        let buf_desc = |bytes: u64| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };

        // 1. General CUSTOM + WRITE_BACK buffer and heap (Xbox UMA pattern) must live in sparse VRAM.
        let custom_wb = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let mut custom_buf: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&custom_wb, D3D12_HEAP_FLAG_NONE, &buf_desc(0x10_0000), D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut custom_buf).map_err(e)?;
        let custom_buf = custom_buf.ok_or("custom wb committed buffer")?;
        let custom_va = custom_buf.GetGPUVirtualAddress();
        ensure!(custom_va != 0 && win(custom_va) != 3, "general CUSTOM+WRITE_BACK buffer landed in window 3: {custom_va:#x}");

        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: 0x10_0000,
            Properties: custom_wb,
            Alignment: 0x1_0000,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut wb_heap: Option<ID3D12Heap> = None;
        dev.CreateHeap(&heap_desc, &mut wb_heap).map_err(e)?;
        let wb_heap = wb_heap.ok_or("CUSTOM+WRITE_BACK heap")?;

        let mut src_buf: Option<ID3D12Resource> = None;
        dev.CreatePlacedResource(&wb_heap, 0, &buf_desc(0x1_0000), D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut src_buf).map_err(e)?;
        let src_buf = src_buf.ok_or("placed src in wb heap")?;
        ensure!(win(src_buf.GetGPUVirtualAddress()) != 3, "CUSTOM+WRITE_BACK heap landed in window 3: {:#x}", src_buf.GetGPUVirtualAddress());

        let mut dst_rb: Option<ID3D12Resource> = None;
        dev.CreatePlacedResource(&wb_heap, 0x1_0000, &buf_desc(0x1_0000), D3D12_RESOURCE_STATE_COPY_DEST, None, &mut dst_rb).map_err(e)?;
        let dst_rb = dst_rb.ok_or("placed COPY_DEST in wb heap")?;

        // 2. Oversized READBACK buffer (> 64 MiB) spills cleanly to sparse VRAM and still reads back on Map(NULL).
        let mut big_rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc(65 << 20),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut big_rb,
        )
        .map_err(e)?;
        let big_rb = big_rb.ok_or("oversized READBACK buffer")?;
        ensure!(win(big_rb.GetGPUVirtualAddress()) != 3, "oversized READBACK buffer (>64 MiB) landed in window 3: {:#x}", big_rb.GetGPUVirtualAddress());

        let mut p_src = std::ptr::null_mut();
        src_buf.Map(0, None, Some(&mut p_src)).map_err(e)?;
        *(p_src as *mut u32) = 0x5EED_CAFE;
        *((p_src as *mut u32).add(1)) = 0x1234_5678;

        list.CopyBufferRegion(&dst_rb, 0, &src_buf, 0, 64);
        list.CopyBufferRegion(&big_rb, 0, &src_buf, 0, 64);
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 1)?;

        // Map(0, NULL, ...) on both sparse readback buffers must automatically sync from VRAM.
        let mut p_dst = std::ptr::null_mut();
        dst_rb.Map(0, None, Some(&mut p_dst)).map_err(e)?;
        let w0 = *(p_dst as *const u32);
        let w1 = *((p_dst as *const u32).add(1));
        ensure!(w0 == 0x5EED_CAFE && w1 == 0x1234_5678, "CUSTOM+WRITE_BACK COPY_DEST Map(NULL) readback mismatch: {w0:#x} {w1:#x}");

        let mut p_big = std::ptr::null_mut();
        big_rb.Map(0, None, Some(&mut p_big)).map_err(e)?;
        let b0 = *(p_big as *const u32);
        let b1 = *((p_big as *const u32).add(1));
        ensure!(b0 == 0x5EED_CAFE && b1 == 0x1234_5678, "oversized READBACK Map(NULL) readback mismatch: {b0:#x} {b1:#x}");
        Ok(())
    }
}

/// Asynchronous ring-buffered sparse upload stress test: streams 80 MiB across 20 back-to-back
/// pipelined submits (wrapping both the 64 MiB staging ring and the 16 upload command slots
/// without CPU waits between submits), plus reused-tile zeroing before new CPU writes.
pub fn sparse_async_ring_upload_stress() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;

        const SLICE_BYTES: usize = 4 << 20; // 4 MiB per submit
        const SUBMITS: usize = 20; // 80 MiB total > 64 MiB staging ring, 20 > 16 upload slots
        const TOTAL_BYTES: usize = SLICE_BYTES * SUBMITS;
        let buf_desc = |bytes: u64| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };

        let mut src: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc(TOTAL_BYTES as u64),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut src,
        )
        .map_err(e)?;
        let src = src.ok_or("stress upload buffer")?;

        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc((SUBMITS * 64) as u64),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(e)?;
        let rb = rb.ok_or("stress readback buffer")?;

        let mut p_src = std::ptr::null_mut();
        src.Map(0, None, Some(&mut p_src)).map_err(e)?;
        let mut p_rb = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p_rb)).map_err(e)?;

        let mut allocs = Vec::with_capacity(SUBMITS);
        let mut lists = Vec::with_capacity(SUBMITS);
        for _ in 0..SUBMITS {
            let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
            let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).map_err(e)?;
            allocs.push(a);
            lists.push(l);
        }

        // Write a disjoint 4 MiB slice on every submit (ring-buffer style) so all 80 MiB cross
        // the 64 MiB staging ring and 16 upload slots back-to-back without waiting between submits.
        let src_u32 = p_src as *mut u32;
        let slice_words = SLICE_BYTES / 4;
        for s in 0..SUBMITS {
            let tag = 0xA500_0000u32 | (s as u32);
            let base_word = s * slice_words;
            // Touch every 64-byte chunk across this 4 MiB slice so all 4 MiB are genuinely uploaded
            for off in (0..slice_words).step_by(16) {
                *src_u32.add(base_word + off) = tag ^ (off as u32);
            }
            let list = &lists[s];
            let slice_off = (s * SLICE_BYTES) as u64;
            // Copy first 32 bytes + last 32 bytes of this 4 MiB slice into readback slot s
            list.CopyBufferRegion(&rb, (s * 64) as u64, &src, slice_off, 32);
            list.CopyBufferRegion(&rb, (s * 64 + 32) as u64, &src, slice_off + (SLICE_BYTES - 64) as u64, 32);
            list.Close().map_err(e)?;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(e)?)]);
        }

        wait(&queue, &fence, 1)?;
        let rb_u32 = p_rb as *const u32;
        let last_off = (slice_words - 16) as u32;
        for s in 0..SUBMITS {
            let tag = 0xA500_0000u32 | (s as u32);
            let got_head = *rb_u32.add(s * 16);
            let got_tail = *rb_u32.add(s * 16 + 8);
            ensure!(got_head == tag, "submit {s} head mismatch: got {got_head:#x}, want {tag:#x}");
            ensure!(got_tail == (tag ^ last_off), "submit {s} tail mismatch: got {got_tail:#x}, want {:#x}", tag ^ last_off);
        }

        // Release `src` so its tiles enter the free list, then allocate a new buffer reusing tiles
        // and verify `zero_buf` clears untouched bytes before our new write lands.
        drop(src);
        let mut reused: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buf_desc(0x2_0000),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut reused,
        )
        .map_err(e)?;
        let reused = reused.ok_or("reused tile buffer")?;
        let mut p_reused = std::ptr::null_mut();
        reused.Map(0, None, Some(&mut p_reused)).map_err(e)?;
        *(p_reused as *mut u32) = 0x7788_99AA;

        allocs[0].Reset().map_err(e)?;
        lists[0].Reset(&allocs[0], None).map_err(e)?;
        lists[0].CopyBufferRegion(&rb, 0, &reused, 0, 128);
        lists[0].Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(lists[0].cast::<ID3D12CommandList>().map_err(e)?)]);
        wait(&queue, &fence, 2)?;
        ensure!(*rb_u32.add(0) == 0x7788_99AA, "reused tile written word mismatch: {:#x}", *rb_u32.add(0));
        ensure!(*rb_u32.add(16) == 0, "reused tile untouched chunk was not zeroed: {:#x}", *rb_u32.add(16));
        Ok(())
    }
}

/// A graphics reservation larger than the two upper sparse windows (8 GiB + 4 MiB, the first
/// thing one retail title does) goes across window 0 – and the console descriptor heaps made
/// after it still get window 0's descriptor range: their start handle (in rax, the console's
/// convention) is a 32-bit-addressable console address, not a PC heap's.
pub fn descriptor_heaps_after_a_large_graphics_reservation() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let big = 0x2_0041_0000usize;
        let r = crate::XMemVirtualAlloc(std::ptr::null(), big, 0x2000, 1u64 << 29, 4);
        ensure!(!r.is_null(), "graphics reservation of {big:#x} bytes refused");
        let check = |ty: D3D12_DESCRIPTOR_HEAP_TYPE, n: u32| -> Result<(), String> {
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
                .map_err(e)?;
            let start: unsafe extern "system" fn(*mut c_void) -> u64 = transmute_copy(&(*(*(heap.as_raw() as *const *const usize)).add(9)));
            let p = start(heap.as_raw());
            ensure!(p >> 32 >= 4 && (p >> 32) & 3 == 0, "{ty:?} heap of {n} slots is not in unified memory window 0: start {p:#x}");
            ensure!(p < r as u64 || p >= r as u64 + big as u64, "{ty:?} heap at {p:#x} inside the title's reservation {r:?}");
            Ok(())
        };
        let views = check(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 442_368);
        let samplers = check(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 512);
        ensure!(crate::VirtualFree(r, 0, 0x8000) != 0, "VirtualFree of the reservation");
        views?;
        samplers
    }
}

/// Committed console textures fill the coherent window only up to the room it keeps: more of
/// them than fit (400 MB of 512x512 RGBA8, up to the room to the MB; one retail title makes
/// 2.2 GB) are still created – as PC resources, also with console-only flag bits – and a
/// readback buffer made afterwards still gets coherent memory (window 3).
pub fn committed_textures_leave_room_for_readback() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let tex = |side: u64, flags: u32| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: side,
            Height: side as u32,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAGS(flags as i32),
            ..Default::default()
        };
        let default = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut kept = Vec::new();
        for i in 0..400 {
            let mut t: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&default, D3D12_HEAP_FLAG_NONE, &tex(512, 0), D3D12_RESOURCE_STATE_COMMON, None, &mut t).map_err(|x| format!("texture {i}: {x}"))?;
            kept.push(t.ok_or("texture")?);
        }
        // past the room: a render target with a console-only flag bit (0x1000000) the PC refuses
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&default, D3D12_HEAP_FLAG_NONE, &tex(2048, 0x100_0001), D3D12_RESOURCE_STATE_COMMON, None, &mut rt)
            .map_err(|x| format!("render target with console flag bits beyond the coherent room: {x}"))?;
        kept.push(rt.ok_or("render target")?);
        let buf = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1 << 20,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb)
            .map_err(e)?;
        let rb = rb.ok_or("readback buffer")?;
        let va = rb.GetGPUVirtualAddress();
        ensure!((va >> 32) & 3 == 3, "readback buffer after the textures is not in the coherent window 3: {va:#x}");
        drop(kept);
        Ok(())
    }
}
