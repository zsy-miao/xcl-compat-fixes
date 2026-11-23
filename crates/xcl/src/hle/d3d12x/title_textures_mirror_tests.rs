use super::*;
use crate::hle::kernelx::{self, mirrors};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};
use windows_sys::Win32::System::Memory as mem;

static TEST_LOCK: Mutex<()> = Mutex::new(());
// Larger than all four 4 GiB UMA windows together: the public allocation hook MUST
// take its Windows fallback. This reserves address space, not 24 GiB of RAM.
const RESERVATION: usize = 24usize << 30;
const OFFSET: usize = 0x2_0000;
const PAGE: usize = 0x1_0000;

struct Reservation(usize);
impl Drop for Reservation {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe { kernelx::VirtualFree(self.0 as *mut c_void, 0, mem::MEM_RELEASE); }
        }
    }
}

unsafe fn reserve(flags: u32, protect: u32) -> Reservation {
    let at = kernelx::VirtualAlloc(std::ptr::null(), RESERVATION, flags, protect) as usize;
    assert_ne!(at, 0, "reserve-only fallback must succeed");
    let r = Reservation(at);
    let u = crate::gpu::uma::get().expect("UMA initialized");
    assert!(!u.contains(at), "test must use the Windows fallback, not a GPU window");
    assert!(kernelx::in_graphics_reservation(at), "fallback must be write-watched");
    r
}

#[test]
fn gpu_visible_fallback_is_forced_and_reports_cpu_writes() {
    let _lock = TEST_LOCK.lock().unwrap();
    unsafe {
        for (flags, protect) in [(0x2040_2000, mem::PAGE_READWRITE), (0x2000_2000, 0x2404), (kernelx::MEM_GRAPHICS | mem::MEM_RESERVE, mem::PAGE_READWRITE)] {
            let r = reserve(flags, protect);
            let at = r.0 + OFFSET;
            assert_eq!(kernelx::VirtualAlloc(at as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, at);
            let mut pages = [std::ptr::null_mut(); 32];
            let (mut count, mut granularity) = (pages.len(), 0);
            assert_eq!(mem::GetWriteWatch(1, at as *const c_void, PAGE, pages.as_mut_ptr(), &mut count, &mut granularity), 0);
            ((at + 0x1234) as *mut u8).write_volatile(7);
            count = pages.len();
            assert_eq!(mem::GetWriteWatch(1, at as *const c_void, PAGE, pages.as_mut_ptr(), &mut count, &mut granularity), 0);
            assert_eq!(&pages[..count], &[(at + 0x1000) as *mut c_void]);
            let base = r.0;
            drop(r);
            assert!(!kernelx::in_graphics_reservation(base), "release removes the watched range");
        }
    }
}

#[test]
fn gpu_mirror_pairing_preserves_offsets_and_release() {
    let _lock = TEST_LOCK.lock().unwrap();
    let (cpu, gpu, other, size) = (0x6A00_0000_0000, 0x6A10_0000_0000, 0x6A20_0000_0000, 0x80_0000);
    mirrors::reserved(cpu, size, 0x2040_2000, mem::PAGE_READWRITE);
    mirrors::reserved(other, size / 2, 0x2040_2000, mem::PAGE_READWRITE);
    mirrors::reserved(gpu, size, 0x2040_2000, 0x801);
    assert_eq!(mirrors::source(gpu), Some(cpu));
    assert_eq!(mirrors::source(gpu + OFFSET), Some(cpu + OFFSET));
    assert_eq!(mirrors::source(gpu + size), None);
    assert_eq!(mirrors::source(cpu), None);
    mirrors::released(gpu);
    assert_eq!(mirrors::source(gpu), None);
    mirrors::reserved(gpu, size, 0x2040_2000, 0x801);
    assert_eq!(mirrors::source(gpu), Some(cpu));
    mirrors::released(cpu);
    assert_eq!(mirrors::source(gpu), None, "releasing either view removes the pair");
    mirrors::released(other);
}

unsafe fn read_r8(dev: &ID3D12Device, queue: &ID3D12CommandQueue, res: &ID3D12Resource) -> Vec<u8> {
    own_work(|| {
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        dev.GetCopyableFootprints(&res.GetDesc(), 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let mut rb = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() },
            D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb,
        ).unwrap();
        let rb: ID3D12Resource = rb.unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let barrier = |a, b| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(res), Subresource: 0, StateBefore: a, StateAfter: b }) },
            ..Default::default()
        };
        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, dev, None, 5_000).unwrap();
        let mut data = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut data)).unwrap();
        let mut out = Vec::with_capacity(64 * 64);
        for y in 0..64 {
            out.extend_from_slice(std::slice::from_raw_parts((data as *const u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), 64));
        }
        rb.Unmap(0, None);
        out
    })
}

#[test]
fn gpu_mirror_texture_reads_cpu_view_and_refreshes_after_cpu_write() {
    let _lock = TEST_LOCK.lock().unwrap();
    unsafe {
        let raw_device = super::super::real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&raw_device).unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let cpu = reserve(0x2040_2000, mem::PAGE_READWRITE);
        let gpu = reserve(0x2040_2000, 0x801);
        let (src, va) = (cpu.0 + OFFSET, gpu.0 + OFFSET);
        assert_eq!(mirrors::source(va), Some(src));
        assert_eq!(mirror_source(va), src);
        assert_eq!(kernelx::VirtualAlloc(src as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, src);
        let mut console = [0u8; 0x38];
        console[0..4].copy_from_slice(&3u32.to_le_bytes());
        console[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        console[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        console[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        console[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        console[0x20..0x24].copy_from_slice(&(DXGI_FORMAT_R8_UNORM.0 as u32).to_le_bytes());
        console[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        console[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes()); // linear console layout
        assert!(xg::Texture::new(&console).unwrap().size() <= PAGE as u64);
        let mut texture = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() },
            D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
        ).unwrap();
        let texture: ID3D12Resource = texture.unwrap();
        let key = texture.as_raw() as usize;
        register(raw_device, texture.as_raw(), va, &console, D3D12_RESOURCE_STATE_COMMON).unwrap();
        assert_eq!(address(key), Some(va), "the title's address remains the GPU view");
        for value in [0x3D, 0x9A] {
            std::ptr::write_bytes(src as *mut u8, value, PAGE);
            refresh(queue.as_raw());
            assert_eq!(read_r8(dev, &queue, &texture), vec![value; 64 * 64], "GPU contents must follow CPU writes through the mirror");
            assert_eq!(address(key), Some(va));
        }
        forget(key);
        assert_eq!(address(key), None);
        // A new registration with already mapped GPU memory uses that view, as in
        // PR #283. Existing entries deliberately retain their registration-time src.
        assert_eq!(kernelx::VirtualAlloc(va as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize, va);
        assert_eq!(mirror_source(va), va);
        register(raw_device, texture.as_raw(), va, &console, D3D12_RESOURCE_STATE_COMMON).unwrap();
        assert_eq!(TEXTURES.lock().unwrap().get(&key).unwrap().src, va);
        forget(key);
        drop(gpu);
        assert_eq!(mirrors::source(va), None);
        drop(cpu);
        dev.GetDeviceRemovedReason().unwrap();
    }
}
