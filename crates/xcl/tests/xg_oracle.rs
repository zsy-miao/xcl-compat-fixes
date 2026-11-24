//! The console's texture layout library as a reference (`XCL_SAMPLE_XG` = the console's own
//! `xg_xs.dll`, from the user's console files – skipped when unset): loaded through XCL's
//! loader and asked the same questions as XCL's implementation.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use xcl::modules::MappedModule;
use xcl::pe::ExportTarget;

fn console() -> Option<&'static Arc<MappedModule>> {
    static M: OnceLock<Option<Arc<MappedModule>>> = OnceLock::new();
    M.get_or_init(|| {
        let path = PathBuf::from(std::env::var_os("XCL_SAMPLE_XG")?);
        xcl::stubs::init_db(&Default::default());
        xcl::hle::init(&xcl::config::Config::default());
        let _ = xcl::eh::install_pc_hook();
        Some(xcl::loader::load_game_dll(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
    })
    .as_ref()
}

fn proc(m: &MappedModule, name: &str) -> usize {
    match m.exports.by_name.get(name) {
        Some(ExportTarget::Rva(r)) => m.base + *r as usize,
        other => panic!("{}!{name}: {other:?}", m.name),
    }
}

/// D3D12_RESOURCE_DESC as the console's (0x38 bytes)
#[allow(clippy::too_many_arguments)]
fn desc(dim: u32, width: u64, height: u32, depth: u16, mips: u16, format: u32, layout: u32, flags: u32) -> [u8; 0x38] {
    let mut d = [0u8; 0x38];
    d[0..4].copy_from_slice(&dim.to_le_bytes());
    d[0x10..0x18].copy_from_slice(&width.to_le_bytes());
    d[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
    d[0x1C..0x1E].copy_from_slice(&depth.to_le_bytes());
    d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
    d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
    d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
    d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
    d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    d
}

type CreateFn = unsafe extern "system" fn(*const u8, *mut *mut std::ffi::c_void) -> i32;

/// (HRESULT, the computer's resource layout) of a texture computer from `create`
unsafe fn layout(create: usize, d: &[u8; 0x38]) -> (i32, Vec<u8>) {
    let mut c = std::ptr::null_mut();
    let hr = std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c);
    let mut out = vec![0u8; 0x1B40];
    if hr >= 0 && !c.is_null() {
        let vt = *(c as *const *const usize);
        let get: unsafe extern "system" fn(*mut std::ffi::c_void, *mut u8) -> i32 = std::mem::transmute(*vt.add(2));
        get(c, out.as_mut_ptr());
        let release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32 = std::mem::transmute(*vt.add(1));
        release(c);
    }
    (hr, out)
}

#[test]
#[ignore]
fn print_console_layouts() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for (name, d) in [
        ("rgba8 256x256 full chain, 64KB_S (0x109)", desc(3, 256, 256, 1, 0, 28, 0x109, 0)),
        ("bc1 1024x512 11 mips, 64KB_S", desc(3, 1024, 512, 1, 11, 71, 0x109, 0)),
        ("rgba8 100x60 1 mip linear", desc(3, 100, 60, 1, 1, 28, 1, 0)),
        ("rgba8 256x256 optimal", desc(3, 256, 256, 1, 0, 28, 0, 0)),
        ("rgba8 3x5 full chain linear", desc(3, 3, 5, 1, 0, 28, 1, 0)),
        ("rgba8 100x60 3 mips linear", desc(3, 100, 60, 1, 3, 28, 1, 0)),
        ("rgba8 256x256 64KB_Z (0x108)", desc(3, 256, 256, 1, 0, 28, 0x108, 0)),
    ] {
        let (hr, l) = unsafe { layout(create, &d) };
        let q = |o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
        let w = |o: usize| u32::from_le_bytes(l[o..o + 4].try_into().unwrap());
        println!("{name}: hr {hr:#x} size {:#x} align {:#x} mips {} planes {} @0x1b38 {}", q(0), q(8), w(0x10), w(0x14), w(0x1B38));
        let p = 0x18;
        println!("  plane: usage {} size {:#x} slice {:#x} +0x18 {:#x} align {:#x} bpe {} tail {} {}", w(p), q(p + 8), q(p + 0x10), q(p + 0x18), q(p + 0x20), w(p + 0x28), w(p + 0x2C), w(p + 0x30));
        for i in 0..(w(0x10) as usize).min(15) {
            let r = p + 0x38 + i * 0x70;
            let row: Vec<String> = (0..0x70).step_by(4).map(|k| format!("{:x}", w(r + k))).collect();
            println!("  mip {i}: {}", row.join(" "));
        }
    }
}

/// Field name of an XG_RESOURCE_LAYOUT offset (for mismatch reports).
fn field(o: usize) -> String {
    if o < 0x18 {
        return format!("header+{o:#x}");
    }
    if o >= 0x1B38 {
        return format!("+{o:#x}");
    }
    let (plane, r) = ((o - 0x18) / 0x6C8, (o - 0x18) % 0x6C8);
    if r < 0x38 {
        format!("plane{plane}+{r:#x}")
    } else {
        format!("plane{plane} mip{}+{:#x}", (r - 0x38) / 0x70, (r - 0x38) % 0x70)
    }
}

#[test]
fn layouts_match_the_console_library() {
    let Some(m) = console() else { return };
    let theirs = proc(m, "XGCreateTextureComputer");
    let ours = xcl::hle::xg::XGCreateTextureComputer as *const () as usize;
    let formats = [28u32, 71, 77, 98, 10, 41, 61, 49, 2, 80, 83, 24, 87, 40, 55];
    let sizes = [(1u64, 1u32), (3, 5), (16, 16), (100, 60), (256, 256), (1000, 333), (1024, 512), (2048, 2048), (4096, 1), (1, 4096), (640, 360), (129, 65), (257, 129), (65, 33), (1025, 513)];
    // (layout, D3D12 flags): explicit modes, and the library's own choice for textures and UAVs
    // (4); render targets / depth (metadata planes) are not modelled
    let mut layouts: Vec<(u32, u32)> = [1u32, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109, 0x10A, 0x10B, 0x114, 0x115, 0x118, 0x119, 0x11A, 0x11B].map(|l| (l, 0)).to_vec();
    layouts.extend([(0, 0), (0, 4), (2, 0), (2, 4)]);
    let (mut cases, mut same, mut both_fail) = (0, 0, 0);
    let mut fields: std::collections::BTreeMap<String, (usize, String)> = Default::default();
    for &f in &formats {
        for &(w, h) in &sizes {
            for &mips in &[0u16, 1, 3] {
                for &array in &[1u16, 6] {
                    for &(lay, flags) in &layouts {
                        let d = desc(3, w, h, array, mips, f, lay, flags);
                        let (hc, lc) = unsafe { layout(theirs, &d) };
                        let (ho, lo) = unsafe { layout(ours, &d) };
                        cases += 1;
                        if hc < 0 && ho < 0 {
                            both_fail += 1;
                            continue;
                        }
                        if (hc < 0) != (ho < 0) {
                            let e = fields.entry(format!("HRESULT {hc:#x} vs ours {ho:#x}")).or_default();
                            e.0 += 1;
                            if e.1.is_empty() {
                                e.1 = format!("fmt {f} {w}x{h} mips {mips} array {array} layout {lay:#x} flags {flags:#x}");
                            }
                            continue;
                        }
                        let diff: Vec<usize> = (0..0x1B40).step_by(4).filter(|&o| lc[o..o + 4] != lo[o..o + 4]).collect();
                        if diff.is_empty() {
                            same += 1;
                        }
                        for o in diff.iter().take(4) {
                            let e = fields.entry(field(*o)).or_default();
                            e.0 += 1;
                            if e.1.is_empty() {
                                let v = |l: &[u8]| u32::from_le_bytes(l[*o..*o + 4].try_into().unwrap());
                                e.1 = format!("fmt {f} {w}x{h} mips {mips} array {array} layout {lay:#x} flags {flags:#x}: console {:#x}, ours {:#x}", v(&lc), v(&lo));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{cases} cases: {same} identical, {both_fail} refused by both");
    for (k, (n, ex)) in fields.iter().take(40) {
        println!("  {k}: {n}× – e.g. {ex}");
    }
    assert_eq!(same + both_fail, cases, "layouts differ from the console library");
}

#[test]
#[ignore]
fn print_accepted_modes() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for (bpe, f) in [(1, 61u32), (2, 49), (4, 28), (8, 10), (16, 2), (8, 71), (16, 77)] {
        let ok: Vec<u32> = (1..32).filter(|&mode| unsafe { layout(create, &desc(3, 256, 256, 1, 1, f, 0x100 + mode, 0)).0 } >= 0).collect();
        println!("bpe {bpe} format {f}: {ok:?}");
    }
}

#[test]
#[ignore]
fn print_z_tails() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for f in [61u32, 49, 28, 10] {
        for (w, h) in [(100u64, 60u32), (128, 128), (128, 256), (200, 200), (256, 256), (64, 64)] {
            for lay in [0x118u32, 0x119] {
                let (hr, l) = unsafe { layout(create, &desc(3, w, h, 1, 0, f, lay, 0)) };
                let u = |o: usize| u32::from_le_bytes(l[o..o + 4].try_into().unwrap());
                let mip0 = 0x18 + 0x38;
                println!("fmt {f} {w}x{h} {lay:#x}: hr {hr:#x} first tail {} mips {} mip0 padded {}x{} size {:#x}", u(0x18 + 0x2C), u(0x10), u(mip0 + 0x24), u(mip0 + 0x28), u(0));
            }
        }
    }
}

/// Texel offsets (GetTexelElementOffsetBytes) of the console's library against XCL's for every
/// mode, element size and pipe / packer configuration: the configuration that matches every
/// sampled texel of the XOR modes is the console's.
#[test]
#[ignore]
fn texel_offsets_and_the_console_pipe_configuration() {
    use xcl::gpu::tiling::PipeConfig;
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    let configs: Vec<PipeConfig> = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 1), (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (5, 3), (4, 4), (5, 4)]
        .map(|(p, k)| PipeConfig { pipes_log2: p, pkrs_log2: k })
        .to_vec();
    let mut seed = 0x1234_5678u32;
    let mut rnd = |n: u32| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) % n.max(1)
    };
    for mode in [1u32, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27] {
        for f in [61u32, 49, 28, 10, 2, 71] {
            let d = desc(3, 300, 200, 2, 0, f, 0x100 + mode, 0);
            let mut c = std::ptr::null_mut();
            if unsafe { std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c) } < 0 {
                continue;
            }
            type Offset = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
            let off: Offset = unsafe { std::mem::transmute(*(*(c as *const *const usize)).add(6)) };
            let ok: Vec<String> = configs
                .iter()
                .filter(|cfg| {
                    let t = xcl::hle::xg::Texture::with_pipes(&d, **cfg).unwrap();
                    let mut good = true;
                    for i in 0..t.texels().unwrap().chain.levels.len() {
                        let l = t.texels().unwrap().chain.levels[i];
                        for k in 0..24 {
                            let (x, y, s) = match k {
                                0 => (0, 0, 0),
                                1 => (l.width - 1, l.height - 1, 1),
                                _ => (rnd(l.width as u32) as usize, rnd(l.height as u32) as usize, rnd(2) as usize),
                            };
                            let mut bit = 0u32;
                            let theirs = unsafe { off(c, 0, i as u32, x as u64, y as u32, s as u32, 0, &mut bit) };
                            if theirs != t.element_offset(i, x, y, s) {
                                good = false;
                            }
                        }
                    }
                    good
                })
                .map(|c| format!("{}/{}", c.pipes_log2, c.pkrs_log2))
                .collect();
            println!("mode {mode:2} format {f:2}: matching pipes/packers {ok:?}");
            unsafe {
                let release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32 = std::mem::transmute(*(*(c as *const *const usize)).add(1));
                release(c);
            }
        }
    }
}

#[test]
#[ignore]
fn print_xor_bits() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for (mode, base) in [(25u32, 9u32), (21, 5), (27, 9)] {
        for f in [28u32, 61] {
            let get = |layout: u32| {
                let d = desc(3, 2048, 2048, 1, 1, f, layout, 0);
                let mut c = std::ptr::null_mut();
                assert!(unsafe { std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c) } >= 0);
                c
            };
            type Offset = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
            let (cx, cs) = (get(0x100 + mode), get(0x100 + base));
            let off = |c: *mut std::ffi::c_void, x: u64, y: u32| unsafe {
                let f: Offset = std::mem::transmute(*(*(c as *const *const usize)).add(6));
                let mut b = 0;
                f(c, 0, 0, x, y, 0, 0, &mut b)
            };
            let xs: Vec<String> = (0..11).map(|k| format!("{:x}", off(cx, 1 << k, 0) ^ off(cs, 1 << k, 0))).collect();
            let ys: Vec<String> = (0..11).map(|k| format!("{:x}", off(cx, 0, 1 << k) ^ off(cs, 0, 1 << k))).collect();
            println!("mode {mode} vs {base} format {f}: x bits {xs:?} | y bits {ys:?}");
        }
    }
}

#[test]
#[ignore]
fn print_xor_64kb_parts() {
    use xcl::gpu::tiling::PipeConfig;
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for cfg in [PipeConfig { pipes_log2: 5, pkrs_log2: 3 }, PipeConfig { pipes_log2: 5, pkrs_log2: 4 }] {
        for mode in [25u32, 27] {
            let d = desc(3, 512, 512, 2, 0, 28, 0x100 + mode, 0);
            let mut c = std::ptr::null_mut();
            assert!(unsafe { std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c) } >= 0);
            type Offset = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
            let off: Offset = unsafe { std::mem::transmute(*(*(c as *const *const usize)).add(6)) };
            let t = xcl::hle::xg::Texture::with_pipes(&d, cfg).unwrap();
            let mut bad = std::collections::BTreeMap::<String, (usize, String)>::new();
            for i in 0..t.texels().unwrap().chain.levels.len() {
                let l = t.texels().unwrap().chain.levels[i];
                for s in 0..2usize {
                    for y in (0..l.height).step_by(7) {
                        for x in (0..l.width).step_by(5) {
                            let mut b = 0;
                            let theirs = unsafe { off(c, 0, i as u32, x as u64, y as u32, s as u32, 0, &mut b) };
                            let ours = t.element_offset(i, x, y, s);
                            if theirs != ours {
                                let k = format!("mip {i} slice {s}");
                                let e = bad.entry(k).or_default();
                                e.0 += 1;
                                if e.1.is_empty() {
                                    e.1 = format!("({x},{y}) console {theirs:#x} ours {ours:#x} xor {:#x}", theirs ^ ours);
                                }
                            }
                        }
                    }
                }
            }
            println!("cfg {}/{} mode {mode}: {} groups differ", cfg.pipes_log2, cfg.pkrs_log2, bad.len());
            for (k, (n, ex)) in bad.iter().take(6) {
                println!("  {k}: {n} – {ex}");
            }
        }
    }
}

#[test]
#[ignore]
fn print_hardware_versions() {
    use xcl::gpu::tiling::PipeConfig;
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    let set: unsafe extern "system" fn(u32) -> i32 = unsafe { std::mem::transmute(proc(m, "XGSetHardwareVersion")) };
    let configs: Vec<PipeConfig> = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 1), (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (5, 3), (4, 4), (5, 4)]
        .map(|(p, k)| PipeConfig { pipes_log2: p, pkrs_log2: k })
        .to_vec();
    for v in 0..8u32 {
        let hr = unsafe { set(v) };
        let d = desc(3, 512, 512, 2, 1, 28, 0x100 + 27, 0);
        let mut c = std::ptr::null_mut();
        if unsafe { std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c) } < 0 {
            println!("version {v}: XGSetHardwareVersion {hr:#x}, create failed");
            continue;
        }
        type Offset = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
        let off: Offset = unsafe { std::mem::transmute(*(*(c as *const *const usize)).add(6)) };
        let ok: Vec<String> = configs
            .iter()
            .filter(|cfg| {
                let t = xcl::hle::xg::Texture::with_pipes(&d, **cfg).unwrap();
                (0..512usize).step_by(13).all(|y| (0..512usize).step_by(11).all(|x| (0..2).all(|s| unsafe { off(c, 0, 0, x as u64, y as u32, s as u32, 0, std::ptr::null_mut()) } == t.element_offset(0, x, y, s))))
            })
            .map(|c| format!("{}/{}", c.pipes_log2, c.pkrs_log2))
            .collect();
        let (_, l) = unsafe { layout(create, &d) };
        println!("version {v}: XGSetHardwareVersion {hr:#x}; 64KB_R_X matches {ok:?}; mip0 +0x50/+0x58 {:#x}/{:#x}", u32::from_le_bytes(l[0x18 + 0x38 + 0x50..][..4].try_into().unwrap()), u32::from_le_bytes(l[0x18 + 0x38 + 0x58..][..4].try_into().unwrap()));
    }
    unsafe { set(0) };
}

/// CopyIntoSubresource / CopyFromSubresource: the console's library and XCL tile the same
/// rows to the same bytes, and read them back.
#[test]
fn subresource_copies_match_the_console_library() {
    let Some(m) = console() else { return };
    let theirs = proc(m, "XGCreateTextureComputer");
    let ours = xcl::hle::xg::XGCreateTextureComputer as *const () as usize;
    type Copy = unsafe extern "system" fn(*mut std::ffi::c_void, *mut u8, u32, u32, *const u8, u32, u32) -> i32;
    let mut seed = 0x9E37_79B9u32;
    let mut cases = 0;
    for &(f, w, h, array, layout) in &[
        (28u32, 300u64, 200u32, 2u16, 0x105u32),
        (28, 300, 200, 2, 0x119),
        (71, 512, 256, 3, 0x109),
        (10, 129, 65, 1, 0x11A),
        (61, 1000, 333, 2, 0x115),
        (98, 256, 256, 6, 0x11B),
        (28, 100, 60, 2, 1),
    ] {
        let d = desc(3, w, h, array, 0, f, layout, 0);
        let make = |create: usize| unsafe {
            let mut c = std::ptr::null_mut();
            assert!(std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c) >= 0, "{layout:#x}");
            c
        };
        let (c, o) = (make(theirs), make(ours));
        let t = xcl::hle::xg::Texture::new(&d).unwrap();
        let size = t.size() as usize;
        let (mut a, mut b) = (vec![0u8; size], vec![0u8; size]);
        let mips = t.texels().unwrap().chain.levels.len();
        for s in 0..array as usize {
            for i in 0..mips {
                let l = t.texels().unwrap().chain.levels[i];
                let row = l.width * t.texels().unwrap().bpe as usize;
                let pitch = row + 16;
                let lin: Vec<u8> = (0..pitch * l.height)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 24) as u8
                    })
                    .collect();
                let sub = (i + s * mips) as u32;
                unsafe {
                    let into = |p: *mut std::ffi::c_void| -> Copy { std::mem::transmute(*(*(p as *const *const usize)).add(8)) };
                    assert_eq!(into(c)(c, a.as_mut_ptr(), 0, sub, lin.as_ptr(), pitch as u32, (pitch * l.height) as u32), 0);
                    assert_eq!(into(o)(o, b.as_mut_ptr(), 0, sub, lin.as_ptr(), pitch as u32, (pitch * l.height) as u32), 0);
                    // ours reads the console's bytes back
                    let from: Copy = std::mem::transmute(*(*(o as *const *const usize)).add(9));
                    let mut back = vec![0u8; pitch * l.height];
                    assert_eq!(from(o, back.as_mut_ptr(), 0, sub, a.as_ptr(), pitch as u32, 0), 0);
                    for y in 0..l.height {
                        assert_eq!(back[y * pitch..y * pitch + row], lin[y * pitch..y * pitch + row], "layout {layout:#x} format {f} mip {i} slice {s} row {y}");
                    }
                }
                cases += 1;
            }
        }
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "layout {layout:#x} format {f}: {differ} bytes differ from the console's tiling");
    }
    println!("{cases} subresources tiled identically");
}

#[test]
#[ignore]
fn print_odd_offsets() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    let d = desc(3, 129, 65, 1, 0, 10, 0x11A, 0);
    let mut c = std::ptr::null_mut();
    assert!(unsafe { std::mem::transmute::<usize, CreateFn>(create)(d.as_ptr(), &mut c) } >= 0);
    type Offset = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
    let off: Offset = unsafe { std::mem::transmute(*(*(c as *const *const usize)).add(6)) };
    let t = xcl::hle::xg::Texture::new(&d).unwrap();
    let mut n = 0;
    for y in 0..65usize {
        for x in 0..129usize {
            let a = unsafe { off(c, 0, 0, x as u64, y as u32, 0, 0, std::ptr::null_mut()) };
            let b = t.element_offset(0, x, y, 0);
            if a != b {
                if n < 6 {
                    println!("({x},{y}): console {a:#x} ours {b:#x}");
                }
                n += 1;
            }
        }
    }
    println!("{n} texels of mip 0 differ; pitch {} block {}x{}", t.texels().unwrap().chain.levels[0].pitch, t.texels().unwrap().chain.layout.block_w, t.texels().unwrap().chain.layout.block_h);
}

#[test]
#[ignore]
fn print_planes() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for (name, d) in [
        ("rgba8 1x1 RT+DS optimal", desc(3, 1, 1, 1, 1, 28, 0, 3)),
        ("rgba8 256x256 RT+DS optimal", desc(3, 256, 256, 1, 1, 28, 0, 3)),
        ("d32 256x256 RT+DS optimal", desc(3, 256, 256, 1, 1, 40, 0, 3)),
        ("d24s8 256x256 RT+DS optimal", desc(3, 256, 256, 1, 1, 45, 0, 3)),
        ("bc1 256x256 RT+DS optimal", desc(3, 256, 256, 1, 1, 71, 0, 3)),
        ("bc1 256x256 DS optimal", desc(3, 256, 256, 1, 1, 71, 0, 2)),
        ("rgba8 256x256 DS 0x119", desc(3, 256, 256, 1, 1, 28, 0x119, 2)),
        ("rgba8 256x256 DS 0x118", desc(3, 256, 256, 1, 1, 28, 0x118, 2)),
        ("rgba8 256x256 DS optimal", desc(3, 256, 256, 1, 1, 28, 0, 2)),
        ("rgba8 1x1 DS optimal", desc(3, 1, 1, 1, 0, 28, 0, 2)),
        ("bc1 256x256 RT optimal", desc(3, 256, 256, 1, 1, 71, 0, 1)),
        ("d24s8 256x256 RT optimal", desc(3, 256, 256, 1, 1, 45, 0, 1)),
        ("d32s8 256x256 DS optimal", desc(3, 256, 256, 1, 1, 20, 0, 2)),
        ("r32f 256x256 DS optimal", desc(3, 256, 256, 1, 1, 41, 0, 2)),
        ("rgba8 256x256 RT optimal", desc(3, 256, 256, 1, 1, 28, 0, 1)),
        ("rgba8 1000x333 RT optimal", desc(3, 1000, 333, 1, 1, 28, 0, 1)),
        ("rgba8 256x256 RT 0x119", desc(3, 256, 256, 1, 1, 28, 0x119, 1)),
        ("rgba8 256x256 RT optimal 3 mips", desc(3, 256, 256, 1, 3, 28, 0, 1)),
        ("rgba8 256x256 RT optimal array 4", desc(3, 256, 256, 4, 1, 28, 0, 1)),
        ("d32 256x256 depth", desc(3, 256, 256, 1, 1, 40, 0, 2)),
        ("d32 1000x333 depth", desc(3, 1000, 333, 1, 1, 40, 0, 2)),
        ("d24s8 256x256 depth", desc(3, 256, 256, 1, 1, 45, 0, 2)),
        ("d16 256x256 depth", desc(3, 256, 256, 1, 1, 55, 0, 2)),
        ("d24s8 256x256 no flags", desc(3, 256, 256, 1, 1, 45, 0x109, 0)),
    ] {
        let (hr, l) = unsafe { layout(create, &d) };
        let q = |o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
        let w = |o: usize| u32::from_le_bytes(l[o..o + 4].try_into().unwrap());
        println!("{name}: hr {hr:#x} size {:#x} align {:#x} mips {} planes {}", q(0), q(8), w(0x10), w(0x14));
        for p in 0..w(0x14).min(4) as usize {
            let b = 0x18 + p * 0x6C8;
            let head: Vec<String> = (0..0x38).step_by(4).map(|k| format!("{:x}", w(b + k))).collect();
            println!("  plane {p}: {}", head.join(" "));
            let r = b + 0x38;
            let row: Vec<String> = (0..0x70).step_by(4).map(|k| format!("{:x}", w(r + k))).collect();
            println!("    mip0: {}", row.join(" "));
        }
    }
}

#[test]
#[ignore]
fn print_htile_sizes() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for &(w, h) in &[(64u64, 64u32), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (1920, 1080), (3840, 2160), (1000, 333), (16384, 16), (16, 16384)] {
        for &(array, mips) in &[(1u16, 1u16), (6, 1), (1, 3)] {
            for f in [40u32, 45] {
                let (hr, l) = unsafe { layout(create, &desc(3, w, h, array, mips, f, 0, 2)) };
                let q = |o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
                let wd = |o: usize| u32::from_le_bytes(l[o..o + 4].try_into().unwrap());
                let planes: Vec<String> = (0..wd(0x14).min(4) as usize).map(|p| { let b = 0x18 + p * 0x6C8; format!("u{} size {:#x} off {:#x} al {:#x}", wd(b), q(b + 8), q(b + 0x18), q(b + 0x20)) }).collect();
                println!("fmt {f} {w}x{h} a{array} m{mips}: hr {hr:#x} total {:#x} | {}", q(0), planes.join(" | "));
            }
        }
    }
}

/// Render targets (CMASK), depth buffers (HTILE, depth, stencil) and the depth + stencil
/// formats against the console library.
#[test]
fn render_target_and_depth_layouts_match() {
    let Some(m) = console() else { return };
    let theirs = proc(m, "XGCreateTextureComputer");
    let ours = xcl::hle::xg::XGCreateTextureComputer as *const () as usize;
    let mut bad = std::collections::BTreeMap::<String, (usize, String)>::new();
    let (mut cases, mut same) = (0, 0);
    for f in [28u32, 71, 40, 55, 45, 20, 41, 39, 44, 118, 119, 120] {
        for (w, h) in [(1u64, 1u32), (100, 60), (256, 256), (1920, 1080)] {
            for mips in [1u16, 3] {
                for array in [1u16, 6] {
                    for (lay, flags) in [(0u32, 1u32), (0, 2), (0x118, 2), (0x119, 2), (0x119, 1), (0x111, 1), (2, 1), (2, 2), (0, 5), (0x109, 2), (0, 3), (0, 4), (0, 0)] {
                        let d = desc(3, w, h, array, mips, f, lay, flags);
                        let (hc, lc) = unsafe { layout(theirs, &d) };
                        let (ho, lo) = unsafe { layout(ours, &d) };
                        cases += 1;
                        let k = if (hc < 0) != (ho < 0) {
                            Some((format!("HRESULT {hc:#x} vs ours {ho:#x}"), String::new()))
                        } else if hc >= 0 {
                            (0..0x1B40).step_by(4).find(|&o| lc[o..o + 4] != lo[o..o + 4]).map(|o| {
                                let v = |l: &[u8]| u32::from_le_bytes(l[o..o + 4].try_into().unwrap());
                                (field(o), format!(": console {:#x}, ours {:#x}", v(&lc), v(&lo)))
                            })
                        } else {
                            None
                        };
                        match k {
                            None => same += 1,
                            Some((name, val)) => {
                                let e = bad.entry(name).or_default();
                                e.0 += 1;
                                if e.1.is_empty() {
                                    e.1 = format!("fmt {f} {w}x{h} mips {mips} array {array} layout {lay:#x} flags {flags:#x}{val}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{cases} render target / depth cases: {same} the same");
    for (k, (n, ex)) in &bad {
        println!("  {k}: {n}× – e.g. {ex}");
    }
    assert!(bad.is_empty());
}

#[test]
#[ignore]
fn print_volumes_lines_and_samples() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    let samples = |mut d: [u8; 0x38], n: u32| {
        d[0x24..0x28].copy_from_slice(&n.to_le_bytes());
        d
    };
    for (name, d) in [
        ("3D rgba8 64x64x16 optimal", desc(4, 64, 64, 16, 1, 28, 0, 0)),
        ("3D rgba8 64x64x16 0x109", desc(4, 64, 64, 16, 1, 28, 0x109, 0)),
        ("3D rgba8 64x64x16 0x119", desc(4, 64, 64, 16, 1, 28, 0x119, 0)),
        ("3D rgba8 64x64x16 0x11A", desc(4, 64, 64, 16, 1, 28, 0x11A, 0)),
        ("3D rgba8 64x64x16 0x115", desc(4, 64, 64, 16, 1, 28, 0x115, 0)),
        ("3D rgba8 64x64x16 0x116", desc(4, 64, 64, 16, 1, 28, 0x116, 0)),
        ("3D rgba8 64x64x16 linear", desc(4, 64, 64, 16, 1, 28, 1, 0)),
        ("3D rgba8 64x64x16 4 mips", desc(4, 64, 64, 16, 4, 28, 0, 0)),
        ("3D r8 32x32x32 optimal", desc(4, 32, 32, 32, 1, 61, 0, 0)),
        ("3D rgba8 64x64x16 UAV", desc(4, 64, 64, 16, 1, 28, 0, 4)),
        ("1D rgba8 256 optimal", desc(2, 256, 1, 1, 1, 28, 0, 0)),
        ("1D rgba8 256 0x119", desc(2, 256, 1, 1, 1, 28, 0x119, 0)),
        ("1D rgba8 256 linear", desc(2, 256, 1, 1, 1, 28, 1, 0)),
        ("1D rgba8 256 array 4 9 mips", desc(2, 256, 1, 4, 0, 28, 0, 0)),
        ("2D rgba8 256x256 4x RT", samples(desc(3, 256, 256, 1, 1, 28, 0, 1), 4)),
        ("2D rgba8 256x256 4x", samples(desc(3, 256, 256, 1, 1, 28, 0, 0), 4)),
        ("2D d32 256x256 4x DS", samples(desc(3, 256, 256, 1, 1, 40, 0, 2), 4)),
        ("2D rgba8 256x256 2x RT", samples(desc(3, 256, 256, 1, 1, 28, 0, 1), 2)),
        ("2D rgba8 256x256 8x RT", samples(desc(3, 256, 256, 1, 1, 28, 0, 1), 8)),
    ] {
        let (hr, l) = unsafe { layout(create, &d) };
        let q = |o: usize| u64::from_le_bytes(l[o..o + 8].try_into().unwrap());
        let w = |o: usize| u32::from_le_bytes(l[o..o + 4].try_into().unwrap());
        println!("{name}: hr {hr:#x} size {:#x} align {:#x} mips {} planes {} dim {}", q(0), q(8), w(0x10), w(0x14), w(0x1B38));
        for p in 0..w(0x14).min(4) as usize {
            let b = 0x18 + p * 0x6C8;
            let head: Vec<String> = (0..0x38).step_by(4).map(|k| format!("{:x}", w(b + k))).collect();
            println!("  plane {p}: {}", head.join(" "));
            for i in 0..w(0x10).min(4) as usize {
                let r = b + 0x38 + i * 0x70;
                let row: Vec<String> = (0..0x70).step_by(4).map(|k| format!("{:x}", w(r + k))).collect();
                println!("    mip{i}: {}", row.join(" "));
            }
        }
    }
}

#[test]
#[ignore]
fn print_line_modes() {
    let Some(m) = console() else { return };
    let create = proc(m, "XGCreateTextureComputer");
    for f in [28u32, 71, 2, 61] {
        let ok: Vec<String> = [0u32, 1, 2].into_iter().chain(0x100..0x120).filter(|&lay| unsafe { layout(create, &desc(2, 256, 1, 1, 1, f, lay, 0)).0 } >= 0).map(|l| format!("{l:#x}")).collect();
        let uav: Vec<String> = [0u32, 1, 2].into_iter().filter(|&lay| unsafe { layout(create, &desc(2, 256, 1, 1, 1, f, lay, 4)).0 } >= 0).map(|l| format!("{l:#x}")).collect();
        let rt = unsafe { layout(create, &desc(2, 256, 1, 1, 1, f, 0, 1)).0 };
        let h2 = unsafe { layout(create, &desc(2, 256, 2, 1, 1, f, 0, 0)).0 };
        println!("fmt {f}: layouts {ok:?}, UAV {uav:?}, RT hr {rt:#x}, height 2 hr {h2:#x}");
    }
}

/// 3D textures: the layouts (size, alignment, levels outside the mip tail) and the texel offsets
/// of every level are the console library's.
#[test]
fn volume_layouts_and_texels_match_the_console_library() {
    let Some(m) = console() else { return };
    let theirs = proc(m, "XGCreateTextureComputer");
    let ours = xcl::hle::xg::XGCreateTextureComputer as *const () as usize;
    let (mut cases, mut same, mut refused, mut texels) = (0, 0, 0, 0);
    let mut fields: std::collections::BTreeMap<String, (usize, String)> = Default::default();
    let mut note = |k: String, ex: String| {
        let e = fields.entry(k).or_default();
        e.0 += 1;
        if e.1.is_empty() {
            e.1 = ex;
        }
    };
    for f in [28u32, 61, 49, 10, 2, 71, 98] {
        for (w, h, d) in [(1u64, 1u32, 1u16), (4, 4, 4), (16, 16, 16), (32, 32, 32), (64, 64, 16), (33, 17, 5), (128, 64, 32), (256, 32, 8), (100, 60, 20)] {
            for mips in [1u16, 0, 3] {
                for (lay, flags) in [(0u32, 0u32), (0, 4), (1, 0), (0x105, 0), (0x109, 0), (0x115, 0), (0x119, 0), (0x11A, 0), (0x106, 0)] {
                    if mips == 3 && w.max(h as u64).max(d as u64) < 4 {
                        continue;
                    }
                    let x = desc(4, w, h, d, mips, f, lay, flags);
                    let what = format!("fmt {f} {w}x{h}x{d} mips {mips} layout {lay:#x} flags {flags}");
                    let (hc, lc) = unsafe { layout(theirs, &x) };
                    let (ho, lo) = unsafe { layout(ours, &x) };
                    cases += 1;
                    if hc < 0 && ho < 0 {
                        refused += 1;
                        continue;
                    }
                    if (hc < 0) != (ho < 0) {
                        note(format!("HRESULT {hc:#x} vs ours {ho:#x}"), what);
                        continue;
                    }
                    let t = xcl::hle::xg::Texture::new(&x).unwrap();
                    // the fields of the levels in a thick tail are not modelled
                    let first_tail = t.texels().unwrap().first_tail;
                    let levels = 0x18 + 0x38..0x18 + 0x6C8;
                    let diff: Vec<usize> = (0..0x1B40).step_by(4).filter(|&o| lc[o..o + 4] != lo[o..o + 4] && !(levels.contains(&o) && (o - levels.start) / 0x70 >= first_tail)).collect();
                    same += diff.is_empty() as usize;
                    for o in diff.iter().take(3) {
                        let v = |l: &[u8]| u32::from_le_bytes(l[*o..*o + 4].try_into().unwrap());
                        note(field(*o), format!("{what}: console {:#x}, ours {:#x}", v(&lc), v(&lo)));
                    }
                    type Offset = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
                    let mut c = std::ptr::null_mut();
                    unsafe { std::mem::transmute::<usize, CreateFn>(theirs)(x.as_ptr(), &mut c) };
                    let off: Offset = unsafe { std::mem::transmute(*(*(c as *const *const usize)).add(6)) };
                    for (i, l) in t.texels().unwrap().chain.levels.iter().enumerate() {
                        for (px, py, pz) in [(0, 0, 0), (l.width - 1, l.height - 1, l.depth - 1), (l.width / 2, l.height / 3, l.depth / 2), (l.width / 3, l.height / 2, 0)] {
                            let mut bit = 0u32;
                            let a = unsafe { off(c, 0, i as u32, px as u64, py as u32, pz as u32, 0, &mut bit) };
                            let b = t.element_offset(i, px, py, pz);
                            texels += 1;
                            // (u64::MAX: the library has no answer, e.g. 64-bit elements in 64KB_D_X)
                            if a != b && a != u64::MAX {
                                note(format!("texel offset, level {}", if l.tail.is_some() { "in the tail" } else { "outside the tail" }), format!("{what} level {i} ({px},{py},{pz}): console {a:#x}, ours {b:#x}"));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{cases} cases: {same} identical, {refused} refused by both; {texels} texels");
    for (k, (n, ex)) in &fields {
        println!("  {k}: {n} - e.g. {ex}");
    }
    assert!(fields.is_empty(), "3D layouts differ from the console library");
}

/// 1D textures: linear, refused in a tiled mode, laid out as the console library does.
#[test]
fn line_layouts_match_the_console_library() {
    let Some(m) = console() else { return };
    let theirs = proc(m, "XGCreateTextureComputer");
    let ours = xcl::hle::xg::XGCreateTextureComputer as *const () as usize;
    let mut bad = Vec::new();
    for f in [28u32, 61, 10, 2, 29] {
        for w in [1u64, 5, 256, 1000, 4096] {
            for (array, mips) in [(1u16, 1u16), (1, 0), (4, 0), (3, 1)] {
                for (lay, flags) in [(0u32, 0u32), (0, 4), (1, 0), (0x100, 0), (0x105, 0), (0x109, 0), (0x119, 0), (0x11B, 4)] {
                    let x = desc(2, w, 1, array, mips, f, lay, flags);
                    let ((hc, lc), (ho, lo)) = unsafe { (layout(theirs, &x), layout(ours, &x)) };
                    if (hc < 0) != (ho < 0) {
                        bad.push(format!("fmt {f} {w} array {array} mips {mips} layout {lay:#x} flags {flags}: console {hc:#x}, ours {ho:#x}"));
                    } else if hc >= 0 {
                        if let Some(o) = (0..0x1B40).step_by(4).find(|&o| lc[o..o + 4] != lo[o..o + 4]) {
                            bad.push(format!("fmt {f} {w} array {array} mips {mips} layout {lay:#x} flags {flags}: {}", field(o)));
                        }
                    }
                }
            }
        }
    }
    println!("{} differ", bad.len());
    for b in bad.iter().take(20) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}

/// Computer method #10 (GetResourceTiling): tile counts, the packed mip info, the tile shape and
/// the subresource tilings of 2D textures as the console's library answers.
#[test]
fn resource_tiling_matches_the_console_library() {
    let Some(m) = console() else { return };
    let theirs = proc(m, "XGCreateTextureComputer");
    let ours = xcl::hle::xg::XGCreateTextureComputer as *const () as usize;
    type T = unsafe extern "system" fn(*mut std::ffi::c_void, *mut u32, *mut [u32; 3], *mut [u32; 3], *mut u32, u32, *mut [u32; 3]) -> i32;
    let run = |create: usize, x: &[u8; 0x38], first: u32, asked: u32| unsafe {
        let mut c = std::ptr::null_mut();
        if std::mem::transmute::<usize, CreateFn>(create)(x.as_ptr(), &mut c) < 0 {
            return None;
        }
        let vt = *(c as *const *const usize);
        let f: T = std::mem::transmute(*vt.add(10));
        let (mut tiles, mut packed, mut shape, mut n, mut subs) = (0xAAAAu32, [0xAAAA_AAAAu32; 3], [0xAAAA_AAAAu32; 3], asked, [[0xAAAA_AAAAu32; 3]; 32]);
        let hr = f(c, &mut tiles, &mut packed, &mut shape, &mut n, first, subs.as_mut_ptr());
        let release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32 = std::mem::transmute(*vt.add(1));
        release(c);
        // (packed: NumStandardMips and NumPackedMips are bytes – the rest of the word is padding)
        Some(if hr < 0 { (hr, 0, [0; 3], [0; 3], 0, Vec::new()) } else { (hr, tiles, [packed[0] & 0xFFFF, packed[1], packed[2]], shape, n, subs.to_vec()) })
    };
    let (mut cases, mut bad) = (0, Vec::new());
    for f in [28u32, 61, 10, 2, 71, 77, 98, 49] {
        for (w, h) in [(256u64, 256u32), (300, 200), (4096, 4096), (2048, 512), (64, 64), (1, 1), (1000, 37)] {
            for (a, mips) in [(1u16, 0u16), (1, 1), (2, 3), (6, 0), (3, 2)] {
                for lay in [0x109u32, 0x10A, 0x119, 0x11A, 0x11B, 0x105, 0x101, 0] {
                    let x = desc(3, w, h, a, mips, f, lay, 0);
                    for (first, asked) in [(0u32, 32u32), (0, 2), (1, 32), (3, 7)] {
                        let (Some(c), Some(o)) = (run(theirs, &x, first, asked), run(ours, &x, first, asked)) else { continue };
                        cases += 1;
                        if c != o {
                            bad.push(format!("fmt {f} {w}x{h} a{a} mips {mips} layout {lay:#x} first {first} asked {asked}:
   console {c:x?}
   ours    {o:x?}"));
                        }
                    }
                }
            }
        }
    }
    println!("{cases} cases, {} differ", bad.len());
    for b in bad.iter().take(6) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}
