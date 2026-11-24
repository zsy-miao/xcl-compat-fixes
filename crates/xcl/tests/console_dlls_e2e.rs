//! Real console DLLs from the game package (`XCL_SAMPLE_DLLS` = their folder; skipped when
//! unset) through XCL's loader: mapped, imports resolved (HLE / Windows / the game's own
//! DLLs), TLS + DllMain run – then their code is called.

use std::path::PathBuf;
use std::sync::OnceLock;

use xcl::modules::MappedModule;
use xcl::pe::ExportTarget;

fn setup() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("XCL_SAMPLE_DLLS")?);
        let gb: usize = std::env::var("XCL_TEST_MEM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        cap_memory(gb << 30);
        xcl::stubs::init_db(&Default::default());
        xcl::hle::init(&xcl::config::Config::default());
        xcl::resolver::set_search_dirs(vec![dir.clone()]);
        let _ = xcl::eh::install_pc_hook();
        xcl::eh::install_crash_handler();
        Some(dir)
    })
    .as_ref()
}

/// Limit this test process's committed memory: a runaway shader compile (DXC / driver) on
/// a big translated program once exhausted the machine's commit and took the user's session
/// down – with the cap only the test fails.
fn cap_memory(bytes: usize) {
    use windows_sys::Win32::System::JobObjects::*;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = bytes;
        let ok = SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32) != 0
            && AssignProcessToJobObject(job, GetCurrentProcess()) != 0;
        if !ok {
            eprintln!("cap_memory: could not set the memory limit");
        }
    }
}

fn load(dir: &std::path::Path, name: &str) -> std::sync::Arc<MappedModule> {
    xcl::loader::load_game_dll(&dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn proc(m: &MappedModule, name: &str) -> usize {
    match m.exports.by_name.get(name) {
        Some(ExportTarget::Rva(r)) => m.base + *r as usize,
        other => panic!("{}!{name}: {other:?}", m.name),
    }
}

#[test]
fn console_crt_dlls_load_and_run() {
    let Some(dir) = setup() else { return };
    unsafe {
        let vcrt = load(dir, "vcruntime140.dll");
        let memcpy: extern "C" fn(*mut u8, *const u8, usize) -> *mut u8 = std::mem::transmute(proc(&vcrt, "memcpy"));
        let src: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut dst = vec![0u8; 1000];
        memcpy(dst.as_mut_ptr(), src.as_ptr(), 1000);
        assert_eq!(dst, src);

        let cpp = load(dir, "msvcp140.dll");
        let hw: extern "C" fn() -> u32 = std::mem::transmute(proc(&cpp, "_Thrd_hardware_concurrency"));
        let cores = std::thread::available_parallelism().unwrap().get() as u32;
        assert_eq!(hw(), cores, "console msvcp140 _Thrd_hardware_concurrency");
        // std::mutex storage (80 B on x64), plain mutex
        let init: extern "C" fn(*mut u8, i32) = std::mem::transmute(proc(&cpp, "_Mtx_init_in_situ"));
        let lock: extern "C" fn(*mut u8) -> i32 = std::mem::transmute(proc(&cpp, "_Mtx_lock"));
        let unlock: extern "C" fn(*mut u8) -> i32 = std::mem::transmute(proc(&cpp, "_Mtx_unlock"));
        let destroy: extern "C" fn(*mut u8) = std::mem::transmute(proc(&cpp, "_Mtx_destroy_in_situ"));
        let mut mtx = vec![0u8; 256];
        init(mtx.as_mut_ptr(), 1);
        // two threads bump a counter under the console mutex
        let (m, counter) = (mtx.as_mut_ptr() as usize, Box::leak(Box::new(0u64)) as *mut u64 as usize);
        let workers: Vec<_> = (0..2)
            .map(|_| {
                std::thread::spawn(move || {
                    for _ in 0..20000 {
                        lock(m as *mut u8);
                        *(counter as *mut u64) += 1;
                        unlock(m as *mut u8);
                    }
                })
            })
            .collect();
        workers.into_iter().for_each(|w| w.join().unwrap());
        assert_eq!(*(counter as *const u64), 40000);
        destroy(mtx.as_mut_ptr());

        let omp = load(dir, "vcomp140.dll");
        let procs: extern "C" fn() -> i32 = std::mem::transmute(proc(&omp, "omp_get_num_procs"));
        assert_eq!(procs() as u32, cores, "console vcomp140 omp_get_num_procs");

        let concrt = load(dir, "concrt140.dll");
        let n: extern "C" fn() -> u32 = std::mem::transmute(proc(&concrt, "?GetProcessorCount@Concurrency@@YAIXZ"));
        assert_eq!(n(), cores, "console concrt140 GetProcessorCount");
    }
}

#[test]
fn other_console_dlls_map_and_initialise() {
    let Some(dir) = setup() else { return };
    for name in ["msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vccorlib140.dll", "vcruntime140_1.dll", "Bink2Scarlett.dll", "XCurl.dll", "libHttpClient.GDK.dll", "amd_fidelityfx_loader_xbox.dll", "amd_fidelityfx_upscaler_xbox.dll", "XboxBvhBuilder.dll"] {
        let m = load(dir, name);
        assert!(m.initialized.load(std::sync::atomic::Ordering::Acquire), "{name}: DllMain");
    }
}

/// libHttpClient.GDK initialises: its XGameRuntime thunks (statically linked, as in every
/// GDK title) load xgameruntime.dll and go through InitializeApiImpl* / QueryApiImpl.
#[test]
fn libhttpclient_initialises_through_the_xgameruntime_thunks() {
    let Some(dir) = setup() else { return };
    let m = load(dir, "libHttpClient.GDK.dll");
    unsafe {
        let init: extern "system" fn(*mut std::ffi::c_void) -> i32 = std::mem::transmute(proc(&m, "HCInitialize"));
        let hr = init(std::ptr::null_mut());
        println!("HCInitialize → {:#010x}", hr as u32);
        assert!(hr >= 0, "HCInitialize: {:#010x}", hr as u32);
        let cleanup: extern "system" fn() = std::mem::transmute(proc(&m, "HCCleanup"));
        cleanup();
    }
}

#[repr(C)]
struct GeometryDesc {
    ty: u32,
    flags: u32,
    transform: u64,
    index_format: u32,
    vertex_format: u32,
    index_count: u32,
    vertex_count: u32,
    index_buffer: u64,
    vertex_buffer: u64,
    vertex_stride: u64,
}

#[repr(C)]
struct BuildInputs {
    ty: u32,
    flags: u32,
    num_descs: u32,
    layout: u32,
    descs: u64,
}

/// Worst-case result size the console builder reports, as measured: a 256-byte header,
/// n leaves, max(n - 1, 1) box32 nodes (128 B), a 4-byte parent pointer per 64 bytes of
/// nodes, rounded up to 1 KB. Leaves: triangle 64 B (bottom level), instance 128 B (top).
fn console_bvh_size(n: u64, leaf: u64) -> u64 {
    let nodes = n * leaf + (n.max(2) - 1) * 128;
    (256 + nodes + nodes / 64 * 4).div_ceil(1024) * 1024
}

/// The console's own BVH builder sizes acceleration structures on the CPU: result / scratch
/// bytes for bottom levels of n triangles and top levels of n instances (XCL_PRINT=1 prints).
/// The sizes pin down the node sizes of the console format (`console_bvh_size`).
#[test]
fn xbox_bvh_builder_prebuild_sizes() {
    let Some(dir) = setup() else { return };
    let m = load(dir, "XboxBvhBuilder.dll");
    let f: extern "system" fn(*mut u8, *const BuildInputs, *const u8, *mut [u64; 3]) = unsafe {
        std::mem::transmute(proc(&m, "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z"))
    };
    // the method only reads settings from the object; a zeroed one = defaults
    let mut this = vec![0u8; 0x10000];
    let print = std::env::var_os("XCL_PRINT").is_some();
    let mut size = |inputs: &BuildInputs| {
        let mut out = [0u64; 3];
        f(this.as_mut_ptr(), inputs, std::ptr::null(), &mut out);
        out
    };
    for n in [1u32, 2, 3, 4, 5, 8, 16, 64, 256, 1024, 4096] {
        let g = GeometryDesc { ty: 0, flags: 1, transform: 0, index_format: 0, vertex_format: 6, index_count: 0, vertex_count: 3 * n, index_buffer: 0, vertex_buffer: 0x1000, vertex_stride: 12 };
        let r = size(&BuildInputs { ty: 1, flags: 0, num_descs: 1, layout: 0, descs: &g as *const GeometryDesc as u64 });
        if print {
            println!("BLAS {n:5} tri: result {:8} scratch {:8} update {:8}  ({:.1} B/tri)", r[0], r[1], r[2], r[0] as f64 / n as f64);
        }
        assert_eq!(r[0], console_bvh_size(n as u64, 64), "bottom level, {n} triangles");
    }
    for n in [1u32, 2, 4, 16, 256, 4096] {
        let r = size(&BuildInputs { ty: 0, flags: 0, num_descs: n, layout: 0, descs: 0x10000 });
        if print {
            println!("TLAS {n:5} inst: result {:8} scratch {:8} update {:8}  ({:.1} B/inst)", r[0], r[1], r[2], r[0] as f64 / n as f64);
        }
        assert_eq!(r[0], console_bvh_size(n as u64, 128), "top level, {n} instances");
    }
}

/// The console's own Bink 2 library decodes a movie from the game (`XCL_SAMPLE_BK2`) on the
/// CPU: open, decode every frame to 32-bit pixels, the picture must change and not be flat.
/// `XCL_BINK_DUMP=<file.ppm>` saves the middle frame.
#[test]
fn bink2_console_dll_decodes_a_game_movie() {
    let Some(dir) = setup() else { return };
    let Some(movie) = std::env::var_os("XCL_SAMPLE_BK2") else { return };
    let m = load(dir, "Bink2Scarlett.dll");
    unsafe {
        let open: extern "C" fn(*const u8, u32) -> *mut u32 = std::mem::transmute(proc(&m, "BinkOpen"));
        let error: extern "C" fn() -> *const std::ffi::c_char = std::mem::transmute(proc(&m, "BinkGetError"));
        let do_frame: extern "C" fn(*mut u32) -> i32 = std::mem::transmute(proc(&m, "BinkDoFrame"));
        let next: extern "C" fn(*mut u32) = std::mem::transmute(proc(&m, "BinkNextFrame"));
        let copy: extern "C" fn(*mut u32, *mut u8, i32, u32, u32, u32, u32) -> i32 = std::mem::transmute(proc(&m, "BinkCopyToBuffer"));
        let close: extern "C" fn(*mut u32) = std::mem::transmute(proc(&m, "BinkClose"));
        let path = format!("{}\0", movie.to_string_lossy());
        let b = open(path.as_ptr(), 0);
        assert!(!b.is_null(), "BinkOpen: {:?}", std::ffi::CStr::from_ptr(error()));
        let (w, h, frames) = (*b, *b.add(1), *b.add(2));
        assert!(w >= 16 && h >= 16 && frames > 0, "{w}x{h}, {frames} frames");
        let mut px = vec![0u8; (w * h * 4) as usize];
        let (mut changes, mut prev, mut mid) = (0, vec![], vec![]);
        for f in 0..frames {
            do_frame(b);
            copy(b, px.as_mut_ptr(), (w * 4) as i32, h, 0, 0, 3); // BINKSURFACE32
            if f > 0 && px != prev {
                changes += 1;
            }
            if f == frames / 2 {
                mid = px.clone();
            }
            prev.clone_from(&px);
            if f + 1 < frames {
                next(b);
            }
        }
        close(b);
        // the middle frame has a picture: many distinct colours, not one flat fill
        let mut colours: Vec<u32> = mid.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], 0])).collect();
        colours.sort_unstable();
        colours.dedup();
        if let Some(out) = std::env::var_os("XCL_BINK_DUMP") {
            let mut ppm = format!("P6 {w} {h} 255\n").into_bytes();
            mid.chunks(4).for_each(|c| ppm.extend([c[2], c[1], c[0]]));
            std::fs::write(out, ppm).unwrap();
        }
        println!("Bink: {w}x{h}, {frames} frames, changes between frames {changes}, colours in the middle one {}", colours.len());
        assert!(changes > frames / 4, "frames change: {changes} of {frames}");
        assert!(colours.len() > 500, "middle frame: {} colours", colours.len());
    }
}

#[repr(C)]
struct BuildDesc {
    dest: u64,
    inputs: BuildInputs,
    source: u64,
    scratch: u64,
}

/// Page-aligned zeroed memory (the builder uses aligned vector loads).
fn pages(n: usize) -> &'static mut [u8] {
    use windows_sys::Win32::System::Memory::*;
    unsafe {
        let p = VirtualAlloc(std::ptr::null(), n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u8;
        assert!(!p.is_null());
        std::slice::from_raw_parts_mut(p, n)
    }
}

/// Game geometry from `core_bvh4.rpf` (`XCL_SAMPLE_BVH4` = the package): one physical chunk
/// (zlib at 0xa4e26070) holds at +0x100 a compressed bottom level (tag 0x81020006: tree,
/// boxes and primitive pairing, no vertices). The console builder's own CPU `Rehydrate…X`
/// expands it with a synthetic vertex buffer (vertex i = (i, -i, i / 2)); the output must be the
/// GPURT gfx10 layout XCL emulates: root @0x30, box16 / box32 nodes, triangle pairs
/// (v0..v3, primitives @52 / @56, triangle id @60) with every primitive exactly once.
#[test]
fn xbox_bvh_builder_rehydrates_game_geometry() {
    let Some(dir) = setup() else { return };
    let Some(pack) = std::env::var_os("XCL_SAMPLE_BVH4") else { return };
    let m = load(dir, "XboxBvhBuilder.dll");
    let chunk = {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(pack).unwrap();
        f.seek(SeekFrom::Start(0xa4e26070)).unwrap();
        let mut z = vec![0u8; 1 << 20];
        f.read_exact(&mut z).unwrap();
        let mut d = miniz_oxide::inflate::core::DecompressorOxide::new();
        let mut out = vec![0u8; 1 << 20];
        let flags = miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER | miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
        let (st, _, n) = miniz_oxide::inflate::core::decompress(&mut d, &z, &mut out, 0, flags);
        assert_eq!(st, miniz_oxide::inflate::TINFLStatus::Done);
        out.truncate(n);
        out
    };
    assert_eq!(&chunk[0x100..0x104], &[0x06, 0x00, 0x02, 0x81], "compressed bottom level tag");
    let src = pages(chunk.len() + 0x1000);
    src[..chunk.len()].copy_from_slice(&chunk);
    let hdr = |at: usize| u32::from_le_bytes(src[0x100 + at..0x104 + at].try_into().unwrap());
    let (prims, tri_nodes) = (hdr(4), hdr(0xC) as usize);
    let scale = [0, 1, 2].map(|k| (2.0f64).powi(src[0x100 + 0x3C + k] as i32 - 127));
    let dest = pages(16 << 20);
    const VERTS: usize = 1 << 16;
    let vb = pages(VERTS * 12);
    for i in 0..VERTS {
        for (k, c) in [i as f32, -(i as f32), i as f32 / 2.0].iter().enumerate() {
            vb[12 * i + 4 * k..12 * i + 4 * k + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
    // primitives carry their geometry's index; identical geometries cover any of them
    let geoms: Vec<GeometryDesc> = (0..1024)
        .map(|_| GeometryDesc { ty: 0, flags: 1, transform: 0, index_format: 0, vertex_format: 6, index_count: 0, vertex_count: VERTS as u32, index_buffer: 0, vertex_buffer: vb.as_ptr() as u64, vertex_stride: 12 })
        .collect();
    let desc = BuildDesc {
        dest: dest.as_ptr() as u64,
        inputs: BuildInputs { ty: 1, flags: 0, num_descs: geoms.len() as u32, layout: 0, descs: geoms.as_ptr() as u64 },
        source: src.as_ptr() as u64 + 0x100,
        scratch: 0,
    };
    let mut this = vec![0u8; 0x10000];
    let rehydrate: extern "system" fn(*mut u8, *mut u8, *const BuildDesc) -> i32 = unsafe {
        std::mem::transmute(proc(&m, "?RehydrateRaytracingAccelerationStructureX@XboxBvhBuilder@@QEAAJPEAUID3D12Device11@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z"))
    };
    assert_eq!(rehydrate(this.as_mut_ptr(), std::ptr::null_mut(), &desc), 0, "Rehydrate HRESULT");
    if let Some(out) = std::env::var_os("XCL_BVH_DUMP") {
        let used = dest.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
        std::fs::write(out, &dest[..used]).unwrap();
    }
    let word = |at: usize| u32::from_le_bytes(dest[at..at + 4].try_into().unwrap());
    assert_eq!(word(0) >> 24 & 1, 0, "rehydrated: compressed bit cleared");
    let root = word(0x30);
    assert!(matches!(root & 7, 4 | 5), "root {root:#x} is a box");
    // vertex index from the synthetic position (positions come out scaled per axis)
    let f = |at: usize| f32::from_bits(word(at)) as f64;
    let vertex = |at: usize| -> usize {
        let i = f(at) / scale[0];
        let node = at & !63;
        assert!(i.fract() == 0.0 && f(at + 4) == -i * scale[1] && f(at + 8) == i / 2.0 * scale[2], "vertex @{at:#x}: node {:?} words {:x?}", (0..12).map(|k| f(node + 4 * k) / scale[k % 3]).collect::<Vec<_>>(), (12..16).map(|k| word(node + 4 * k)).collect::<Vec<_>>());
        i as usize
    };
    // @48 = geometry << 8 | second triangle << 7 | opaque; primitive indices count within their
    // geometry, so (geometry, primitive) is unique
    let (mut seen_prims, mut seen_nodes, mut boxes) = (std::collections::BTreeMap::<(u32, usize), u32>::new(), std::collections::BTreeSet::new(), [0usize; 2]);
    let mut todo = vec![root];
    while let Some(n) = todo.pop() {
        let at = ((n & !7) as usize) << 3;
        match n & 7 {
            4 | 5 => {
                boxes[(n & 7) as usize - 4] += 1;
                todo.extend((0..4).map(|k| word(at + 4 * k)).filter(|&c| c != u32::MAX));
            }
            t @ 0..=1 => {
                seen_nodes.insert(at);
                let id = word(at + 60) >> (8 * t);
                let (isrc, jsrc) = ((id & 3) as usize, (id >> 2 & 3) as usize);
                if t == 0 {
                    // @48 bit 7: a second triangle – the game's shader then tests the node again
                    // as type 1 (`v76 = [node+48] >> 7`, `next = node | 1`)
                    if word(at + 48) & 0x80 != 0 {
                        todo.push(n | 1);
                    }
                    let p = word(at + 52) as usize;
                    *seen_prims.entry((word(at + 48) >> 8, p)).or_default() += 1;
                    let v = [vertex(at), vertex(at + 12), vertex(at + 24)];
                    let mut sorted = v;
                    sorted.sort_unstable();
                    assert_eq!(sorted, [3 * p, 3 * p + 1, 3 * p + 2], "triangle 0 of node {n:#x}");
                    assert_eq!((v[isrc], v[jsrc]), (3 * p + 1, 3 * p + 2), "I / J source of node {n:#x}");
                } else {
                    let q = word(at + 56) as usize;
                    *seen_prims.entry((word(at + 48) >> 8, q)).or_default() += 1;
                    // triangle 1 = (v1, v3, v2): only v3 is its own vertex
                    let v3 = vertex(at + 36);
                    assert!((3 * q..3 * q + 3).contains(&v3), "triangle 1 of node {n:#x}");
                    if v3 == 3 * q + 1 {
                        assert_eq!(isrc, 1, "I source of node {n:#x}");
                    }
                    if v3 == 3 * q + 2 {
                        assert_eq!(jsrc, 1, "J source of node {n:#x}");
                    }
                }
            }
            t => panic!("unexpected node type {t} ({n:#x})"),
        }
    }
    println!("Rehydrate: {} triangle nodes, box32 {}, box16 {}", seen_nodes.len(), boxes[1], boxes[0]);
    assert_eq!(seen_nodes.len(), tri_nodes, "triangle nodes = header @0x0C");
    assert!(seen_prims.values().all(|&c| c == 1), "every (geometry, primitive) exactly once");
    // every geometry's primitives 0..n are all there
    let mut per_geometry = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    for &(g, p) in seen_prims.keys() {
        per_geometry.entry(g).or_default().push(p);
    }
    for (g, ps) in &per_geometry {
        assert!(ps.iter().enumerate().all(|(i, &p)| i == p), "geometry {g}: primitives 0..{} without gaps", ps.len());
    }
    let counts: Vec<usize> = per_geometry.values().map(Vec::len).collect();
    println!("geometries: {} with triangles {counts:?} (total {}; header field @0x04 = {prims})", counts.len(), seen_prims.len());
}

/// Probe: the console BVH builder constructed on XCL's D3D12.X device (real backend) – what
/// it asks the device for (root signatures, compute pipelines from its console shaders) and
/// whether it survives. `XCL_PROBE_BVH_DEVICE=1` runs it.
#[test]
fn xbox_bvh_builder_on_xcl_device_probe() {
    let Some(dir) = setup() else { return };
    if std::env::var_os("XCL_PROBE_BVH_DEVICE").is_none() {
        return;
    }
    let m = load(dir, "XboxBvhBuilder.dll");
    unsafe {
        // IID_ID3D12Device
        let iid: [u8; 16] = [0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7];
        let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let hr = xcl::hle::d3d12x::D3D12XboxCreateDevice(std::ptr::null_mut(), std::ptr::null(), iid.as_ptr() as _, &mut dev);
        println!("D3D12XboxCreateDevice {hr:#x} {dev:?}");
        assert!(hr >= 0 && !dev.is_null());
        let ctor: extern "system" fn(*mut u8, *mut std::ffi::c_void, bool) -> *mut u8 = std::mem::transmute(proc(&m, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z"));
        let this = pages(0x10000);
        ctor(this.as_mut_ptr(), dev, false);
        println!("XboxBvhBuilder constructed");
        if std::env::var_os("XCL_PROBE_BVH_RECORD").is_none() {
            return;
        }
        // record (not execute) a bottom-level build of 64 triangles into a command list made
        // on XCL's device; buffers in the unified memory (console GPU address = CPU address)
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D12::*;
        let device = ID3D12Device::from_raw_borrowed(&dev).unwrap();
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let uma = xcl::gpu::uma::get().expect("UMA");
        let n = 64u32;
        let vb = uma.alloc(n as usize * 36, 256).unwrap();
        for (i, v) in std::slice::from_raw_parts_mut(vb as *mut f32, n as usize * 9).iter_mut().enumerate() {
            *v = (i % 9) as f32 * 0.1 + (i / 9) as f32;
        }
        let geom = GeometryDesc { ty: 0, flags: 1, transform: 0, index_format: 0, vertex_format: 6, index_count: 0, vertex_count: 3 * n, index_buffer: 0, vertex_buffer: vb as u64, vertex_stride: 12 };
        let inputs = BuildInputs { ty: 1, flags: 0, num_descs: 1, layout: 0, descs: &geom as *const GeometryDesc as u64 };
        let prebuild: extern "system" fn(*mut u8, *const BuildInputs, *const u8, *mut [u64; 3]) = std::mem::transmute(proc(&m, "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z"));
        let mut sizes = [0u64; 3];
        prebuild(this.as_mut_ptr(), &inputs, std::ptr::null(), &mut sizes);
        let dest = uma.alloc(sizes[0] as usize, 256).unwrap();
        let scratch = uma.alloc(sizes[1] as usize, 256).unwrap();
        println!("sizes: result {} B, scratch {} B; dest {dest:?}, scratch {scratch:?}", sizes[0], sizes[1]);
        let desc = BuildDesc { dest: dest as u64, inputs, source: 0, scratch: scratch as u64 };
        let build: extern "system" fn(*mut u8, *mut std::ffi::c_void, *const BuildDesc, u32, *const u8) = std::mem::transmute(proc(
            &m,
            "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
        ));
        build(this.as_mut_ptr(), list.as_raw(), &desc, 0, std::ptr::null());
        list.Close().unwrap();
        println!("BLAS build recorded on the command list (not executed)");
    }
}

/// The memory cap of `setup` holds: committing 9 GB in this process fails.
#[test]
fn memory_cap_is_active() {
    if setup().is_none() {
        return;
    }
    use windows_sys::Win32::System::Memory::*;
    unsafe {
        let p = VirtualAlloc(std::ptr::null(), 9 << 30, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if !p.is_null() {
            VirtualFree(p, 0, MEM_RELEASE);
        }
        assert!(p.is_null(), "a 9 GB commit went through – the job memory cap is not active");
    }
}

// ---------------------------------------------------------------------------------------
// Fake console command list: what D3D12.X code inlined into a title writes into the list
// object and which methods it calls (`XCL_PROBE_BVH_FAKELIST=1`).
// ---------------------------------------------------------------------------------------

static FAKE_SLOT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static FAKE_CALLS: std::sync::Mutex<Vec<(u32, [usize; 6])>> = std::sync::Mutex::new(Vec::new());
static FAKE_BUMP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

extern "system" fn fake_method(this: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> usize {
    let slot = FAKE_SLOT.load(std::sync::atomic::Ordering::Relaxed);
    FAKE_CALLS.lock().unwrap().push((slot, [this, a, b, c, d, e]));
    match slot {
        // AllocateUploadMemory(size, alignment) → CPU pointer (UMA: = GPU address)
        148 => {
            let align = b.max(16);
            let at = FAKE_BUMP.load(std::sync::atomic::Ordering::Relaxed).div_ceil(align) * align;
            FAKE_BUMP.store(at + a, std::sync::atomic::Ordering::Relaxed);
            at
        }
        _ => 0,
    }
}

/// vtable of `n` thunks: `mov dword [FAKE_SLOT], slot ; jmp fake_method`
fn fake_vtable(n: usize) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::*;
    let per = 32usize;
    unsafe {
        let code = VirtualAlloc(std::ptr::null(), n * per, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
        (0..n)
            .map(|i| {
                let mut c = vec![0x48, 0xB8];
                c.extend_from_slice(&(&FAKE_SLOT as *const _ as u64).to_le_bytes()); // mov rax, &FAKE_SLOT
                c.extend_from_slice(&[0xC7, 0x00]);
                c.extend_from_slice(&(i as u32).to_le_bytes()); // mov dword [rax], i
                c.extend_from_slice(&[0x48, 0xB8]);
                c.extend_from_slice(&(fake_method as *const () as usize as u64).to_le_bytes()); // mov rax, fake_method
                c.extend_from_slice(&[0xFF, 0xE0]); // jmp rax
                assert!(c.len() <= per);
                let p = code.add(i * per);
                std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
                p as usize
            })
            .collect()
    }
}

#[test]
fn xbox_bvh_builder_records_into_a_fake_console_list() {
    let Some(dir) = setup() else { return };
    if std::env::var_os("XCL_PROBE_BVH_FAKELIST").is_none() {
        return;
    }
    let m = load(dir, "XboxBvhBuilder.dll");
    unsafe {
        let iid: [u8; 16] = [0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7];
        let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
        assert!(xcl::hle::d3d12x::D3D12XboxCreateDevice(std::ptr::null_mut(), std::ptr::null(), iid.as_ptr() as _, &mut dev) >= 0);
        let ctor: extern "system" fn(*mut u8, *mut std::ffi::c_void, bool) -> *mut u8 = std::mem::transmute(proc(&m, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z"));
        let this = pages(0x10000);
        ctor(this.as_mut_ptr(), dev, false);
        let uma = xcl::gpu::uma::get().expect("UMA");
        // the fake list: vtable, then a zeroed console-sized body; +0x18 / +0x30 = record buffer
        let vt = Box::leak(fake_vtable(600).into_boxed_slice());
        let list = pages(0x10000);
        let records = uma.alloc(1 << 20, 256).unwrap() as usize;
        let lp = list.as_mut_ptr();
        let w = |at: usize, v: usize| (lp.add(at) as *mut usize).write(v);
        w(0, vt.as_ptr() as usize);
        w(0x18, records);
        w(0x30, records + (1 << 20));
        FAKE_BUMP.store(uma.alloc(4 << 20, 4096).unwrap() as usize, std::sync::atomic::Ordering::Relaxed);
        let snapshot = list.to_vec();
        // a bottom level of 64 triangles
        let n = 64u32;
        let vb = uma.alloc(n as usize * 36, 256).unwrap();
        for (i, v) in std::slice::from_raw_parts_mut(vb as *mut f32, n as usize * 9).iter_mut().enumerate() {
            *v = (i % 9) as f32 * 0.1 + (i / 9) as f32;
        }
        let geom = GeometryDesc { ty: 0, flags: 1, transform: 0, index_format: 0, vertex_format: 6, index_count: 0, vertex_count: 3 * n, index_buffer: 0, vertex_buffer: vb as u64, vertex_stride: 12 };
        let inputs = BuildInputs { ty: 1, flags: 0, num_descs: 1, layout: 0, descs: &geom as *const GeometryDesc as u64 };
        let dest = uma.alloc(13312, 256).unwrap();
        let scratch = uma.alloc(23872, 256).unwrap();
        let desc = BuildDesc { dest: dest as u64, inputs, source: 0, scratch: scratch as u64 };
        println!("vb {vb:?} dest {dest:?} scratch {scratch:?} records {records:#x} upload from {:#x}", FAKE_BUMP.load(std::sync::atomic::Ordering::Relaxed));
        let build: extern "system" fn(*mut u8, *mut u8, *const BuildDesc, u32, *const u8) = std::mem::transmute(proc(
            &m,
            "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
        ));
        build(this.as_mut_ptr(), list.as_mut_ptr(), &desc, 0, std::ptr::null());
        println!("== list method calls");
        for (slot, a) in FAKE_CALLS.lock().unwrap().iter() {
            println!("  #{slot:<4} {:x?}", &a[1..]);
        }
        println!("== changed fields of the list object (offset: before → after)");
        for k in (0..0x10000).step_by(8) {
            let (b, a) = (u64::from_le_bytes(snapshot[k..k + 8].try_into().unwrap()), u64::from_le_bytes(list[k..k + 8].try_into().unwrap()));
            if a != b {
                println!("  +{k:#06x}: {b:#x} → {a:#x}");
            }
        }
        let end = (list.as_ptr().add(0x18) as *const usize).read();
        // which interface the objects are: #33's argument and the first pointer of pipeline records
        let kind = |p: usize| -> &'static str {
            use windows::Win32::Graphics::Direct3D12::*;
            let raw = p as *mut std::ffi::c_void;
            let Some(u) = <windows::core::IUnknown as windows::core::Interface>::from_raw_borrowed(&raw) else { return "?" };
            if windows::core::Interface::cast::<ID3D12PipelineState>(u).is_ok() {
                "PSO"
            } else if windows::core::Interface::cast::<ID3D12RootSignature>(u).is_ok() {
                "RootSignature"
            } else {
                "other"
            }
        };
        let args33: std::collections::BTreeSet<usize> = FAKE_CALLS.lock().unwrap().iter().filter(|c| c.0 == 33).map(|c| c.1[1]).collect();
        for p in &args33 {
            println!("  #33 argument {p:#x}: {}", kind(*p));
        }
        println!("== device {dev:?}");
        let mut seen = std::collections::BTreeSet::new();
        let rec64 = std::slice::from_raw_parts(records as *const u32, (end - records) / 4);
        for w in rec64.windows(2) {
            let p = (w[0] as usize) | (w[1] as usize) << 32;
            if (0x1_0000_0000..0x7FFF_0000_0000).contains(&p) && w[1] != 4 && seen.insert(p) {
                let vt = (p as *const usize).read();
                println!("  pointer {p:#x}: vtable {vt:#x} = {} ({})", xcl::modules::describe_address(vt), if xcl::modules::describe_address(vt).contains(".dll") { "?" } else { kind(p) });
            }
        }
        println!("== record buffer: {} B", end - records);
        let rec = std::slice::from_raw_parts(records as *const u32, (end - records) / 4);
        for (i, ch) in rec.chunks(8).enumerate().take(64) {
            println!("  {:#06x}: {:08x?}", i * 32, ch);
        }
    }
}

/// The console BVH builder builds a bottom level ON THE GPU through XCL: its D3D12.X inline
/// code records into a console-shaped list (`console_list`), XCL turns the records into PC
/// commands running the translated console shaders, and the result is read back with XCL's
/// own node decoder (`bvh::Bvh::trace`) against brute force.
/// `XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_GPU=1` runs it.
#[test]
fn xbox_bvh_builder_builds_on_the_gpu() {
    let Some(dir) = setup() else { return };
    if std::env::var_os("XCL_PROBE_BVH_GPU").is_none() || !xcl::hle::d3d12x::console_list::enabled() {
        return;
    }
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::*;
    let m = load(dir, "XboxBvhBuilder.dll");
    unsafe {
        let iid: [u8; 16] = [0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7];
        let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
        assert!(xcl::hle::d3d12x::D3D12XboxCreateDevice(std::ptr::null_mut(), std::ptr::null(), iid.as_ptr() as _, &mut dev) >= 0);
        let ctor: extern "system" fn(*mut u8, *mut std::ffi::c_void, bool) -> *mut u8 = std::mem::transmute(proc(&m, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z"));
        let this = pages(0x10000);
        ctor(this.as_mut_ptr(), dev, false);
        let device = ID3D12Device::from_raw_borrowed(&dev).unwrap();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let pc: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let console = xcl::hle::d3d12x::console_list::create_with_allocator(pc.clone().into_raw(), alloc.as_raw());
        // console code drives the list through its vtable: Close #9, Reset #10; the queue gets
        // the console pointer itself
        let vt = *(console as *const *const usize);
        let close: extern "system" fn(*mut std::ffi::c_void) -> i32 = std::mem::transmute(*vt.add(9));
        let reset: extern "system" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void) -> i32 = std::mem::transmute(*vt.add(10));
        let q_vt = *(queue.as_raw() as *const *const usize);
        let execute: extern "system" fn(*mut std::ffi::c_void, u32, *const *mut std::ffi::c_void) = std::mem::transmute(*q_vt.add(10));
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let uma = xcl::gpu::uma::get().expect("UMA");
        let prebuild: extern "system" fn(*mut u8, *const BuildInputs, *const u8, *mut [u64; 3]) = std::mem::transmute(proc(&m, "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z"));
        let build: extern "system" fn(*mut u8, *mut std::ffi::c_void, *const BuildDesc, u32, *const u8) = std::mem::transmute(proc(
            &m,
            "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
        ));
        let mut seed = 12345u64;
        let mut rnd = |lo: f32, hi: f32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((seed >> 40) as f32 / (1u64 << 24) as f32)
        };
        // two rounds on the same list: record → Close → Execute → check; then allocator and
        // list Reset and a different set of triangles
        for (round, n) in [(1u64, 64usize), (2, 200)] {
            if round > 1 {
                alloc.Reset().unwrap();
                assert_eq!(reset(console, alloc.as_raw(), std::ptr::null_mut()), 0, "list Reset");
            }
            let tris: Vec<[[f32; 3]; 3]> = (0..n)
                .map(|_| {
                    let c = [rnd(-8.0, 8.0), rnd(-8.0, 8.0), rnd(-8.0, 8.0)];
                    [0; 3].map(|_| [0, 1, 2].map(|a| c[a] + rnd(-1.5, 1.5)))
                })
                .collect();
            let vb = uma.alloc(n * 36, 256).unwrap();
            std::ptr::copy_nonoverlapping(tris.as_ptr() as *const u8, vb, n * 36);
            let geom = GeometryDesc { ty: 0, flags: 1, transform: 0, index_format: 0, vertex_format: 6, index_count: 0, vertex_count: 3 * n as u32, index_buffer: 0, vertex_buffer: vb as u64, vertex_stride: 12 };
            let inputs = BuildInputs { ty: 1, flags: 0, num_descs: 1, layout: 0, descs: &geom as *const GeometryDesc as u64 };
            let mut sizes = [0u64; 3];
            prebuild(this.as_mut_ptr(), &inputs, std::ptr::null(), &mut sizes);
            let dest = uma.alloc(sizes[0] as usize, 256).unwrap();
            let scratch = uma.alloc(sizes[1] as usize, 256).unwrap();
            std::ptr::write_bytes(dest, 0, sizes[0] as usize);
            std::ptr::write_bytes(scratch, 0, sizes[1] as usize);
            let desc = BuildDesc { dest: dest as u64, inputs, source: 0, scratch: scratch as u64 };
            build(this.as_mut_ptr(), console, &desc, 0, std::ptr::null());
            assert_eq!(close(console), 0, "Close");
            execute(queue.as_raw(), 1, [console].as_ptr());
            queue.Signal(&fence, round).unwrap();
            let start = std::time::Instant::now();
            while fence.GetCompletedValue() < round {
                assert!(start.elapsed().as_secs() < 10, "GPU did not finish within 10 s (device removal: {:?})", device.GetDeviceRemovedReason());
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let data = std::slice::from_raw_parts(dest, sizes[0] as usize).to_vec();
            if let Some(out) = std::env::var_os("XCL_BVH_DUMP") {
                std::fs::write(format!("{}.{round}", out.to_string_lossy()), &data).unwrap();
            }
            let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
            let bvh = xcl::gpu::bvh::Bvh { root: word(0x30), data: data.clone() };
            let mut hits = 0;
            for _ in 0..500 {
                let o = [rnd(-12.0, 12.0), rnd(-12.0, 12.0), rnd(-12.0, 12.0)];
                let to = [rnd(-6.0, 6.0), rnd(-6.0, 6.0), rnd(-6.0, 6.0)];
                let d = [to[0] - o[0], to[1] - o[1], to[2] - o[2]];
                let brute = tris
                    .iter()
                    .enumerate()
                    .filter_map(|(i, tri)| xcl::gpu::bvh::triangle_hit(tri, o, d).map(|h| (h[0] / h[1], i as u32)))
                    .filter(|h| (0.0..100.0).contains(&h.0))
                    .min_by(|a, b| a.0.total_cmp(&b.0));
                assert_eq!(bvh.trace(o, d, 100.0).map(|g| g.1), brute.map(|b| b.1), "round {round}: ray {o:?} → {to:?}");
                hits += brute.is_some() as u32;
            }
            println!("round {round}: {n} triangles built on the GPU by the console builder – 500 rays match brute force ({hits} hits)");
            assert!(hits > 20);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct InstanceDesc {
    transform: [[f32; 4]; 3],
    id_mask: u32,
    contribution_flags: u32,
    blas: u64,
}

/// Top level built on the GPU by the console builder (two bottom levels + instances, some
/// rotated / scaled, some only moved) and traced with XCL's instance-node reading
/// (bvh::Bvh::trace_scene, layout taken from the game's RT shader) against brute force in
/// world space. XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_TLAS=1 runs it.
#[test]
fn xbox_bvh_builder_builds_a_tlas_on_the_gpu() {
    let Some(dir) = setup() else { return };
    if std::env::var_os("XCL_PROBE_BVH_TLAS").is_none() || !xcl::hle::d3d12x::console_list::enabled() {
        return;
    }
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::*;
    let m = load(dir, "XboxBvhBuilder.dll");
    unsafe {
        let iid: [u8; 16] = [0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7];
        let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
        assert!(xcl::hle::d3d12x::D3D12XboxCreateDevice(std::ptr::null_mut(), std::ptr::null(), iid.as_ptr() as _, &mut dev) >= 0);
        let ctor: extern "system" fn(*mut u8, *mut std::ffi::c_void, bool) -> *mut u8 = std::mem::transmute(proc(&m, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z"));
        let this = pages(0x10000);
        ctor(this.as_mut_ptr(), dev, false);
        let device = ID3D12Device::from_raw_borrowed(&dev).unwrap();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let pc: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let console = xcl::hle::d3d12x::console_list::create_with_allocator(pc.into_raw(), alloc.as_raw());
        let vt = *(console as *const *const usize);
        let close: extern "system" fn(*mut std::ffi::c_void) -> i32 = std::mem::transmute(*vt.add(9));
        let reset: extern "system" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void) -> i32 = std::mem::transmute(*vt.add(10));
        let q_vt = *(queue.as_raw() as *const *const usize);
        let execute: extern "system" fn(*mut std::ffi::c_void, u32, *const *mut std::ffi::c_void) = std::mem::transmute(*q_vt.add(10));
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let uma = xcl::gpu::uma::get().expect("UMA");
        let prebuild: extern "system" fn(*mut u8, *const BuildInputs, *const u8, *mut [u64; 3]) = std::mem::transmute(proc(&m, "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z"));
        let build: extern "system" fn(*mut u8, *mut std::ffi::c_void, *const BuildDesc, u32, *const u8) = std::mem::transmute(proc(
            &m,
            "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
        ));
        let mut round = 0u64;
        // record one build, run it, return (result address, size)
        let mut run = |inputs: &BuildInputs| -> (u64, usize) {
            if round > 0 {
                alloc.Reset().unwrap();
                assert_eq!(reset(console, alloc.as_raw(), std::ptr::null_mut()), 0);
            }
            round += 1;
            let mut sizes = [0u64; 3];
            prebuild(this.as_mut_ptr(), inputs, std::ptr::null(), &mut sizes);
            let dest = uma.alloc(sizes[0] as usize, 256).unwrap();
            let scratch = uma.alloc(sizes[1].max(256) as usize, 256).unwrap();
            std::ptr::write_bytes(dest, 0, sizes[0] as usize);
            std::ptr::write_bytes(scratch, 0, sizes[1] as usize);
            let desc = BuildDesc { dest: dest as u64, inputs: BuildInputs { ..*inputs }, source: 0, scratch: scratch as u64 };
            build(this.as_mut_ptr(), console, &desc, 0, std::ptr::null());
            assert_eq!(close(console), 0, "Close");
            execute(queue.as_raw(), 1, [console].as_ptr());
            queue.Signal(&fence, round).unwrap();
            let start = std::time::Instant::now();
            while fence.GetCompletedValue() < round {
                assert!(start.elapsed().as_secs() < 10, "GPU did not finish within 10 s ({:?})", device.GetDeviceRemovedReason());
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            (dest as u64, sizes[0] as usize)
        };
        let mut seed = 777u64;
        let mut rnd = |lo: f32, hi: f32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((seed >> 40) as f32 / (1u64 << 24) as f32)
        };
        // two bottom levels (small objects around the origin)
        let mut blases = Vec::new();
        for n in [40usize, 60] {
            let tris: Vec<[[f32; 3]; 3]> = (0..n)
                .map(|_| {
                    let c = [rnd(-2.0, 2.0), rnd(-2.0, 2.0), rnd(-2.0, 2.0)];
                    [0; 3].map(|_| [0, 1, 2].map(|a| c[a] + rnd(-0.6, 0.6)))
                })
                .collect();
            let vb = uma.alloc(n * 36, 256).unwrap();
            std::ptr::copy_nonoverlapping(tris.as_ptr() as *const u8, vb, n * 36);
            let geom = GeometryDesc { ty: 0, flags: 1, transform: 0, index_format: 0, vertex_format: 6, index_count: 0, vertex_count: 3 * n as u32, index_buffer: 0, vertex_buffer: vb as u64, vertex_stride: 12 };
            let (addr, size) = run(&BuildInputs { ty: 1, flags: 0, num_descs: 1, layout: 0, descs: &geom as *const GeometryDesc as u64 });
            blases.push((addr, std::slice::from_raw_parts(addr as *const u8, size).to_vec(), tris));
        }
        // instances: every 3rd only moved, the others rotated + scaled + moved
        let count = 16;
        let insts: Vec<InstanceDesc> = (0..count)
            .map(|i| {
                let k = i % 2;
                let t = [rnd(-15.0, 15.0), rnd(-15.0, 15.0), rnd(-15.0, 15.0)];
                let m3 = if i % 3 == 0 {
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                } else {
                    let q = [rnd(-1.0, 1.0), rnd(-1.0, 1.0), rnd(-1.0, 1.0), rnd(-1.0, 1.0)];
                    let l = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let [w, x, y, z] = q.map(|v| v / l);
                    let s = [rnd(0.5, 2.0), rnd(0.5, 2.0), rnd(0.5, 2.0)];
                    let r = [
                        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
                    ];
                    [0, 1, 2].map(|row| [r[row][0] * s[0], r[row][1] * s[1], r[row][2] * s[2]])
                };
                InstanceDesc {
                    transform: [0, 1, 2].map(|row| [m3[row][0], m3[row][1], m3[row][2], t[row]]),
                    id_mask: (100 + i as u32) | 0xFF << 24,
                    contribution_flags: 0,
                    blas: blases[k].0,
                }
            })
            .collect();
        let ib = uma.alloc(count * 64, 256).unwrap();
        std::ptr::copy_nonoverlapping(insts.as_ptr() as *const u8, ib, count * 64);
        let (tlas, tsize) = run(&BuildInputs { ty: 0, flags: 0, num_descs: count as u32, layout: 0, descs: ib as u64 });
        let data = std::slice::from_raw_parts(tlas as *const u8, tsize).to_vec();
        if let Some(out) = std::env::var_os("XCL_BVH_DUMP") {
            std::fs::write(format!("{}.tlas", out.to_string_lossy()), &data).unwrap();
        }
        let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        println!("TLAS: {tsize} B, header {:x?}", (0..16).map(|k| word(4 * k)).collect::<Vec<_>>());
        let top = xcl::gpu::bvh::Bvh { root: word(0x30), data: data.clone() };
        let lookup = |a: u64| blases.iter().find(|b| b.0 == a).map(|b| b.1.as_slice());
        let mut hits = 0;
        for _ in 0..800 {
            let o = [rnd(-25.0, 25.0), rnd(-25.0, 25.0), rnd(-25.0, 25.0)];
            let to = [rnd(-15.0, 15.0), rnd(-15.0, 15.0), rnd(-15.0, 15.0)];
            let d = [to[0] - o[0], to[1] - o[1], to[2] - o[2]];
            let brute = insts
                .iter()
                .flat_map(|inst| {
                    let b = blases.iter().find(|b| b.0 == inst.blas).unwrap();
                    b.2.iter().enumerate().filter_map(move |(p, tri)| {
                        let m = &inst.transform;
                        let w = tri.map(|v| [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3]));
                        xcl::gpu::bvh::triangle_hit(&w, o, d).map(|h| (h[0] / h[1], p as u32, inst.id_mask & 0xFF_FFFF))
                    })
                })
                .filter(|h| (0.0..100.0).contains(&h.0))
                .min_by(|a, b| a.0.total_cmp(&b.0));
            let got = top.trace_scene(&lookup, o, d, 100.0);
            assert_eq!(got.map(|g| (g.1, g.2)), brute.map(|b| (b.1, b.2)), "ray {o:?} → {to:?}");
            hits += brute.is_some() as u32;
        }
        println!("TLAS built on the GPU by the console builder: 800 rays match brute force ({hits} hits)");
        assert!(hits > 30);
    }
}

// ---------------------------------------------------------------------------------------
// FSR4 (console upscaler) for learning how console code binds textures / descriptor tables
// ---------------------------------------------------------------------------------------

#[repr(C)]
struct FfxHeader {
    ty: u64,
    next: *mut FfxHeader,
}

#[repr(C)]
struct FfxCreateUpscale {
    header: FfxHeader,
    flags: u32,
    max_render: [u32; 2],
    max_upscale: [u32; 2],
    message: usize,
}

#[repr(C)]
struct FfxBackend {
    header: FfxHeader,
    device: *mut std::ffi::c_void,
}

/// Probe: the console FSR4 upscaler creates its context on XCL's device. With
/// XCL_D3D12_TRACE=1 the device methods it calls are logged. `XCL_PROBE_FSR=1` runs it.
#[test]
fn fsr4_console_upscaler_creates_a_context_probe() {
    let Some(dir) = setup() else { return };
    if std::env::var_os("XCL_PROBE_FSR").is_none() {
        return;
    }
    let m = load(dir, "amd_fidelityfx_upscaler_xbox.dll");
    unsafe {
        let iid: [u8; 16] = [0xf1, 0x19, 0x98, 0x18, 0xb6, 0x1d, 0x57, 0x4b, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7];
        let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
        assert!(xcl::hle::d3d12x::D3D12XboxCreateDevice(std::ptr::null_mut(), std::ptr::null(), iid.as_ptr() as _, &mut dev) >= 0);
        let mut backend = FfxBackend { header: FfxHeader { ty: 0x0100_0002, next: std::ptr::null_mut() }, device: dev };
        let mut desc = FfxCreateUpscale {
            header: FfxHeader { ty: 0x0001_0000, next: &mut backend.header },
            // the only configuration in the game's package (the DLL says so on its debug output):
            // HDR | inverted depth | auto exposure, 1280x720 -> 1920x1080
            flags: 1 | 1 << 3 | 1 << 5,
            max_render: [1280, 720],
            max_upscale: [1920, 1080],
            message: 0,
        };
        let create: extern "C" fn(*mut usize, *mut FfxHeader, *const u8) -> u32 = std::mem::transmute(proc(&m, "ffxCreateContext"));
        let mut ctx = 0usize;
        let rc = create(&mut ctx, &mut desc.header, std::ptr::null());
        println!("ffxCreateContext: code {rc:#x}, context {ctx:#x}");
        if rc != 0 || std::env::var_os("XCL_PROBE_FSR_DISPATCH").is_none() {
            return;
        }
        // inputs at 1280x720, output 1920x1080, made on XCL's device
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let device = ID3D12Device::from_raw_borrowed(&dev).unwrap();
        let tex = |w: u32, h: u32, f: DXGI_FORMAT, uav: bool| -> ID3D12Resource {
            let mut r: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: w as u64,
                        Height: h,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: f,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
                        ..Default::default()
                    },
                    if uav { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { D3D12_RESOURCE_STATE_COMMON },
                    None,
                    &mut r,
                )
                .unwrap();
            r.unwrap()
        };
        let color = tex(1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
        let depth = tex(1280, 720, DXGI_FORMAT_R32_FLOAT, false);
        let motion = tex(1280, 720, DXGI_FORMAT_R16G16_FLOAT, false);
        let output = tex(1920, 1080, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
        // FfxApiResource: resource, {type, format, width, height, depth, mips, flags, usage}, state
        let res = |r: &ID3D12Resource, fmt: u32, w: u32, h: u32, usage: u32, state: u32| -> [u32; 12] {
            let p = r.as_raw() as usize;
            [p as u32, (p >> 32) as u32, 2, fmt, w, h, 1, 1, 0, usage, state, 0]
        };
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let pc: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let console = xcl::hle::d3d12x::console_list::create_with_allocator(pc.into_raw(), alloc.as_raw());
        let mut d = vec![0u32; 432 / 4];
        let put = |d: &mut Vec<u32>, at: usize, w: &[u32]| d[at / 4..at / 4 + w.len()].copy_from_slice(w);
        put(&mut d, 0, &[0x0001_0001, 0, 0, 0]); // header: FFX_API_DISPATCH_DESC_TYPE_UPSCALE, no next
        put(&mut d, 16, &[console as usize as u32, (console as usize >> 32) as u32]);
        put(&mut d, 24, &res(&color, 4, 1280, 720, 0, 4));
        put(&mut d, 72, &res(&depth, 28, 1280, 720, 0, 4));
        put(&mut d, 120, &res(&motion, 18, 1280, 720, 0, 4));
        put(&mut d, 312, &res(&output, 4, 1920, 1080, 2, 2));
        put(&mut d, 376, &[1280, 720, 1920, 1080]); // renderSize, upscaleSize
        put(&mut d, 368, &[1280f32.to_bits(), 720f32.to_bits()]); // motion vector scale
        put(&mut d, 400, &[16.6f32.to_bits(), 1f32.to_bits()]); // frameTimeDelta, preExposure
        put(&mut d, 408, &[1]); // reset
        put(&mut d, 412, &[0.1f32.to_bits(), 1000f32.to_bits(), 1.0f32.to_bits(), 1f32.to_bits()]);
        // `XCL_PROBE_FSR_EXECUTE`: the inputs get content first (colour: a horizontal 0..1 grey
        // gradient, depth 0.5, no motion) and the recorded frame runs on the GPU below
        let execute = std::env::var_os("XCL_PROBE_FSR_EXECUTE").is_some();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let mut fence_value = 0u64;
        let mut run = |list: &ID3D12GraphicsCommandList| {
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            fence_value += 1;
            queue.Signal(&fence, fence_value).unwrap();
            let t0 = std::time::Instant::now();
            while fence.GetCompletedValue() < fence_value && t0.elapsed().as_secs() < 20 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let removed = device.GetDeviceRemovedReason();
            assert!(removed.is_ok() && fence.GetCompletedValue() != u64::MAX, "device removed: {removed:?}");
            assert!(fence.GetCompletedValue() >= fence_value, "GPU work did not finish in 20 s");
        };
        let buffer = |bytes: u64, heap: D3D12_HEAP_TYPE| -> ID3D12Resource {
            let mut r: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: bytes,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    if heap == D3D12_HEAP_TYPE_READBACK { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_GENERIC_READ },
                    None,
                    &mut r,
                )
                .unwrap();
            r.unwrap()
        };
        let footprint = |r: &ID3D12Resource| -> (D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64) {
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            device.GetCopyableFootprints(&r.GetDesc(), 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
            (fp, total)
        };
        if execute {
            let h = |x: f32| -> u16 {
                // f32 -> f16 for values in 0..1 (normal or zero)
                if x <= 0.0 { 0 } else { let b = x.to_bits(); ((((b >> 23) & 0xFF) as i32 - 127 + 15) as u16) << 10 | ((b >> 13) & 0x3FF) as u16 }
            };
            let prep_alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let prep: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &prep_alloc, None).unwrap();
            let mut uploads = Vec::new();
            for (r, texel) in [
                (&color, &(|x: u32, _y: u32| { let g = h(x as f32 / 1279.0); [g, g, g, h(1.0)].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>() }) as &dyn Fn(u32, u32) -> Vec<u8>),
                (&depth, &|_x: u32, _y: u32| 0.5f32.to_le_bytes().to_vec()),
                (&motion, &|_x: u32, _y: u32| vec![0u8; 4]),
            ] {
                let (fp, total) = footprint(r);
                let up = buffer(total, D3D12_HEAP_TYPE_UPLOAD);
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                up.Map(0, None, Some(&mut p)).unwrap();
                let base = p as *mut u8;
                for y in 0..fp.Footprint.Height {
                    let mut row = Vec::new();
                    for x in 0..fp.Footprint.Width {
                        row.extend(texel(x, y));
                    }
                    std::ptr::copy_nonoverlapping(row.as_ptr(), base.add(fp.Offset as usize + (y * fp.Footprint.RowPitch) as usize), row.len());
                }
                up.Unmap(0, None);
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                prep.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                uploads.push(up);
            }
            prep.Close().unwrap();
            run(&prep);
            println!("FSR inputs uploaded (colour gradient, depth 0.5, no motion)");
        }
        let dispatch: extern "C" fn(*mut usize, *const u32) -> u32 = std::mem::transmute(proc(&m, "ffxDispatch"));
        let rc = dispatch(&mut ctx, d.as_ptr());
        println!("ffxDispatch: code {rc:#x}");
        assert_eq!(rc, 0);
        // descriptor tables in the stream: 32-bit addresses of slots in XCL's console heaps;
        // the 2D images there must be the inputs (formats 71 / 22 / 29) in the XCL ABI form
        use xcl::hle::d3d12x::{console_heap, console_list};
        let words = console_list::recorded_words(console);
        let mut tables = 0;
        let mut formats = std::collections::BTreeSet::new();
        let (mut buffers, mut buffers_in_uma) = (0, 0);
        let uma = xcl::gpu::uma::get().unwrap();
        for &w in &words {
            if console_heap::words_at(w).is_none() {
                continue;
            }
            tables += 1;
            for k in 0..24 {
                if let Some(d) = console_heap::words_at(w.wrapping_add(32 * k)) {
                    if d[3] >> 28 == 9 {
                        assert_eq!(d[0] % 2, 0, "word 0 = SRV index in the PC heap (even)");
                        formats.insert((d[1] >> 20) & 0x1FF);
                    } else if d[3] & 0xFFF == 0xFAC && d[3] >> 30 == 0 {
                        // V#: base address in UMA, num_records set
                        buffers += 1;
                        let a = d[0] as usize | ((d[1] & 0xFFFF) as usize) << 32;
                        if uma.contains(a) && d[2] != 0 {
                            buffers_in_uma += 1;
                        }
                    }
                }
            }
        }
        println!(
            "FSR stream: {} words, {tables} table pointers, 2D image formats in the tables: {formats:?}, V#s in the tables: {buffers} (in UMA: {buffers_in_uma}), console buffers in UMA: {}",
            words.len(),
            xcl::hle::d3d12x::real::console_buffers()
        );
        assert!(xcl::hle::d3d12x::real::console_buffers() > 0, "FSR buffers in UMA");
        assert_eq!(buffers, buffers_in_uma, "every V# in the tables points at a buffer in UMA");
        assert!(tables > 0);
        for f in [71, 22, 29] {
            assert!(formats.contains(&f), "format {f} (FSR input) in the tables");
        }
        let t = console_list::close(console).unwrap();
        println!("console list → PC: {t:?}");
        assert!(t.unknown.is_none(), "all FSR records recognised");
        assert!(t.clears > 0 && t.copies > 0 && t.skipped == 0, "UAV clears and copies carried over to the PC");
        if !execute {
            return;
        }
        assert_eq!(t.skipped_dispatches, 0, "every FSR4 pass translated");
        let pc_raw = console_list::pc_list(console).expect("the console list's PC list");
        let pc = ID3D12GraphicsCommandList::from_raw_borrowed(&pc_raw).unwrap().clone();
        let t0 = std::time::Instant::now();
        run(&pc);
        println!("FSR4 frame ran on the GPU in {:.1} ms (with submission)", t0.elapsed().as_secs_f64() * 1000.0);
        // the output back to the CPU
        let (fp, total) = footprint(&output);
        let rb = buffer(total, D3D12_HEAP_TYPE_READBACK);
        let rb_alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let rb_list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).unwrap();
        let barrier = |before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(&output), Subresource: 0, StateBefore: before, StateAfter: after }),
            },
            ..Default::default()
        };
        rb_list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&output), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        rb_list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        rb_list.Close().unwrap();
        run(&rb_list);
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).unwrap();
        let half = |v: u16| -> f32 {
            let (s, e, m) = ((v >> 15) as u32, ((v >> 10) & 0x1F) as i32, (v & 0x3FF) as f32);
            let x = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else if e == 31 { if m == 0.0 { f32::INFINITY } else { f32::NAN } } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
            if s == 1 { -x } else { x }
        };
        let (mut finite, mut nonzero, mut n) = (0u64, 0u64, 0u64);
        let (mut left, mut right) = (0f64, 0f64);
        for y in (0..1080u32).step_by(8) {
            let row = (p as *const u8).add(fp.Offset as usize + (y * fp.Footprint.RowPitch) as usize) as *const u16;
            for x in (0..1920u32).step_by(8) {
                let r = half(*row.add(4 * x as usize));
                n += 1;
                if r.is_finite() {
                    finite += 1;
                    if r != 0.0 {
                        nonzero += 1;
                    }
                    if x < 480 {
                        left += r as f64;
                    } else if x >= 1440 {
                        right += r as f64;
                    }
                }
            }
        }
        rb.Unmap(0, None);
        println!("FSR4 output: {n} samples, {finite} finite, {nonzero} non-zero; red mean left quarter {:.3}, right quarter {:.3}", left / (n as f64 / 4.0), right / (n as f64 / 4.0));
        assert_eq!(finite, n, "no NaN / inf in FSR4's output");
        assert!(nonzero > n / 2, "FSR4 wrote its output");
        assert!(right > left, "the input's left-to-right gradient survives the upscale");
    }
}

/// No GPU: every compute shader inside XboxBvhBuilder.dll (found by its DXBC header) against
/// what XCL does with its root arguments. The shader's stage descriptor says how many user
/// SGPRs it has; its RTS0 part gives the argument layout. Arguments must not be put into
/// registers the shader does not have – with the layout's rule alone that happens for the
/// builder's shaders (four user SGPRs, s4 = the group id, every argument read from the table),
/// and the builder then computes nothing.
#[test]
fn xbox_bvh_builder_shaders_get_no_argument_in_a_register_they_do_not_have() {
    use xcl::hle::d3d12x::console_list::{inline_for, layout_from_blob};
    let Some(dir) = setup() else { return };
    let dll = std::fs::read(dir.join("XboxBvhBuilder.dll")).unwrap();
    let word = |at: usize| u32::from_le_bytes(dll[at..at + 4].try_into().unwrap()) as usize;
    let (mut shaders, mut by_rule_alone, mut wrong) = (0, 0, Vec::new());
    let mut at = 0;
    while let Some(i) = dll[at..].windows(4).position(|w| w == b"DXBC") {
        let start = at + i;
        at = start + 4;
        if start + 32 > dll.len() || word(start + 20) != 1 {
            continue;
        }
        let Some(blob) = dll.get(start..start + word(start + 24)) else { continue };
        let (Some(user), Some(l)) = (xcl::gpu::recomp::pso::compute_user_sgprs(blob), layout_from_blob(blob)) else { continue };
        shaders += 1;
        let has = 4 * user.saturating_sub(4);
        by_rule_alone += (l.inline > has) as u32;
        if inline_for(&l, Some(user)) > has {
            wrong.push(format!("+{start:#x}: {user} user SGPRs, {} bytes of arguments in registers", inline_for(&l, Some(user))));
        }
    }
    println!("XboxBvhBuilder.dll: {shaders} compute shaders; by the layout's rule alone {by_rule_alone} would get arguments in registers they do not have");
    assert!(shaders >= 40, "only {shaders} compute shaders found in the DLL");
    assert!(wrong.is_empty(), "{} of {shaders} shaders: {:?}", wrong.len(), &wrong[..wrong.len().min(4)]);
}
