//! The console descriptor heaps' public contract, pinned from outside: register / start /
//! slot / words_at / copy_words / unregister. The inline tests cover one happy path each; this
//! covers the arithmetic they imply (window-0 placement, slot math, T# / buffer V# / sampler
//! copy semantics, end-of-heap bounds, and release) on a non-shared heap so no GPU is needed.

use xcl::hle::d3d12x::console_heap::{self, Slot};

/// The unified memory, or a counted skip.
fn uma() -> Option<&'static xcl::gpu::uma::Uma> {
    let u = xcl::gpu::uma::get();
    if u.is_none() {
        xcl::gpu::recomp::runner::skip_test("no unified memory");
    }
    u
}

fn rel(u: &xcl::gpu::uma::Uma, p: u64) -> u32 {
    (p as usize - u.base()) as u32
}

#[test]
fn view_heap_slots_window_zero_and_release() {
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC001, 4, 0x2000, 0x5000, 32, false, false, true));
    let v = console_heap::start(0xC001).unwrap();
    assert_eq!(u.window_of(v as usize), Some(0), "view heap must live in window 0 (32-bit table addressing)");

    // slot 1: SRV index 2*1, pc_srv = base + 2k*inc, pc_uav = base + (2k+1)*inc, gpu UAV at +1
    let s1 = console_heap::slot(v + 32).unwrap();
    assert!(!s1.sampler);
    assert_eq!(s1.k, 1);
    assert_eq!(s1.index, 2);
    assert_eq!(s1.pc_srv, 0x2000 + 2 * 32);
    assert_eq!(s1.pc_uav, 0x2000 + 3 * 32);
    assert_eq!(s1.pc_gpu_uav, 0x5000 + 3 * 32, "GPU handle of the UAV is +1 slot");
    assert_eq!(s1.shared_srv, 0, "no shared heap: no shared SRV");

    // window 0 start: address arithmetic must be self-consistent
    assert_eq!(u.window_of(console_heap::slot(v).unwrap().at), Some(0));

    // end of the heap: the last slot is 3 (4 slots), past it is None
    assert!(console_heap::slot(v + 96).is_some());
    assert!(console_heap::slot(v + 4 * 32).is_none(), "slot past the last one must be None");
    assert!(console_heap::slot(v as u64 - 1).is_none(), "slot before the heap must be None");

    console_heap::unregister(0xC001);
    assert!(console_heap::start(0xC001).is_none(), "start after unregister must be None");
    assert!(console_heap::slot(v).is_none(), "slot after unregister must be None");
    assert!(console_heap::slot(v + 3 * 32).is_none(), "last slot after unregister must be None");
}

#[test]
fn view_heap_indexing_is_per_page_slot_number() {
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC002, 4, 0x3000, 0x6000, 32, false, false, true));
    let v = console_heap::start(0xC002).unwrap();
    // 4 slots: 0..3, each 32 bytes apart, k = (addr - base) / 32
    for k in 0..4u32 {
        let s = console_heap::slot(v + (k * 32) as u64).unwrap();
        assert_eq!(s.k, k);
        assert_eq!(s.index, 2 * k);
        assert_eq!(s.pc_srv, 0x3000 + (2 * k * 32) as u64);
        assert_eq!(s.pc_uav, 0x3000 + ((2 * k + 1) * 32) as u64);
        assert_eq!(s.pc_gpu_uav, 0x6000 + ((2 * k + 1) * 32) as u64);
        assert_eq!(u.window_of(s.at), Some(0));
        assert_eq!(u.base() as u64 + rel(u, v + (k * 32) as u64) as u64, v + (k * 32) as u64);
    }
    console_heap::unregister(0xC002);
}

#[test]
fn sampler_heap_slots_and_copy() {
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC003, 2, 0x9000, 0, 16, true, false, true));
    let sm = console_heap::start(0xC003).unwrap();
    assert_eq!(u.window_of(sm as usize), Some(0), "sampler heap must live in window 0");

    let (q0, q1) = (console_heap::slot(sm).unwrap(), console_heap::slot(sm + 16).unwrap());
    assert!(q0.sampler && q1.sampler);
    assert_eq!(q0.index, 0);
    assert_eq!(q1.index, 1);
    assert_eq!(q0.pc_srv, 0x9000);
    assert_eq!(q1.pc_srv, 0x9000 + 16, "a sampler takes one 16-byte PC slot");
    assert_eq!(q1.pc_uav, 0, "a sampler has no UAV");
    assert_eq!(q1.shared_srv, 0);

    // S# copy: index 0 → 1, word 3 [11:0] renumbered, rest verbatim
    console_heap::write(&q0, &[1, 2, 3, 0xABC0_0000]);
    console_heap::copy_words(&q0, &q1);
    let w = console_heap::words_at(rel(u, sm + 16)).unwrap();
    assert_eq!(&w[..4], &[1, 2, 3, 0xABC0_0001]);
    assert!(w[4..].iter().all(|&x| x == 0), "only the first 4 words of a sampler slot are used");

    // end of the sampler heap
    assert!(console_heap::slot(sm + 32).is_none());

    console_heap::unregister(0xC003);
    assert!(console_heap::slot(sm).is_none());
}

#[test]
fn image_copy_renumbers_word_0_and_buffer_copy_keeps_address() {
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC004, 4, 0x1000, 0x5000, 32, false, false, true));
    let v = console_heap::start(0xC004).unwrap();
    let s1 = console_heap::slot(v + 32).unwrap();
    let s3 = console_heap::slot(v + 96).unwrap();

    // T# (word 3 [28:31] >= 8 selects the image path): word 0 = destination SRV index
    console_heap::write(&s1, &[2, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
    console_heap::copy_words(&s1, &s3);
    let w = console_heap::words_at(rel(u, v + 96)).unwrap();
    assert_eq!(&w[..4], &[6, 0x0470_0000, 0, 9 << 28], "image copy: word 0 = dst index 2*3");

    // buffer V#: address and stride stay as the title made them
    console_heap::write(&s1, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
    console_heap::copy_words(&s1, &s3);
    let w = console_heap::words_at(rel(u, v + 96)).unwrap();
    assert_eq!(w[0], 0x1234, "buffer copy: word 0 keeps the address");
    assert_eq!(w[1], 0);
    assert_eq!(w[2], 64);
    assert_eq!(w[3], 0x3101_6FAC);

    // the source slot is untouched by the copy
    let src = console_heap::words_at(rel(u, v + 32)).unwrap();
    assert_eq!(src[0], 0x1234, "copy_words must not modify the source");

    console_heap::unregister(0xC004);
}

#[test]
fn words_at_reads_the_console_address_space() {
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC005, 2, 0x4000, 0x7000, 32, false, false, true));
    let v = console_heap::start(0xC005).unwrap();
    let s0 = console_heap::slot(v).unwrap();
    let s1 = console_heap::slot(v + 32).unwrap();
    console_heap::write(&s0, &[1, 2, 3, 4, 5, 6, 7, 8]);
    console_heap::write(&s1, &[9, 0x0A0B_0C0D, 0, 0, 0, 0, 0, 0]);
    let w0 = console_heap::words_at(rel(u, v)).unwrap();
    let w1 = console_heap::words_at(rel(u, v + 32)).unwrap();
    assert_eq!(w0, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(w1[0], 9);
    assert_eq!(w1[1], 0x0A0B_0C0D);
    // a console address outside every heap is None
    assert!(console_heap::words_at(0xF_FFFF).is_none());
    console_heap::unregister(0xC005);
}

#[test]
fn heap_not_window_zero_is_still_registered_but_reported_elsewhere() {
    // a heap registered here is in window 0 by construction; the release path is what this
    // pins: after unregister, words_at reports None and the heap is forgotten
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC006, 1, 0x5000, 0x8000, 32, false, false, true));
    let v = console_heap::start(0xC006).unwrap();
    assert!(console_heap::words_at(rel(u, v)).is_some());
    console_heap::unregister(0xC006);
    assert!(console_heap::words_at(rel(u, v)).is_none());
    assert!(console_heap::start(0xC006).is_none());
}

#[test]
fn non_shader_visible_heap_reports_no_gpu_uav() {
    let Some(u) = uma() else { return };
    // pc_gpu = 0: the title made the heap CPU-only; slots report pc_gpu_uav == 0
    assert!(console_heap::register(0xC007, 2, 0x6000, 0, 32, false, false, false));
    let v = console_heap::start(0xC007).unwrap();
    for k in 0..2u32 {
        let s = console_heap::slot(v + (k * 32) as u64).unwrap();
        assert_eq!(s.pc_gpu_uav, 0);
        assert_eq!(u.window_of(s.at), Some(0));
    }
    console_heap::unregister(0xC007);
}

#[test]
fn copy_words_of_a_view_to_a_buffer_slot_does_not_crash_on_short_words() {
    // copy_words reads exactly the source's word count and writes exactly 8 (a view slot);
    // a source slot written with only 4 words must not be read past the slot
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC008, 2, 0x7000, 0x9000, 32, false, false, true));
    let v = console_heap::start(0xC008).unwrap();
    let s0 = console_heap::slot(v).unwrap();
    let s1 = console_heap::slot(v + 32).unwrap();
    // only 4 words written; the rest of the slot is zeros (the heap is zero-filled)
    console_heap::write(&s0, &[0x1111, 0x2222, 0x3333, 8 << 28]);
    console_heap::copy_words(&s0, &s1);
    let w = console_heap::words_at(rel(u, v + 32)).unwrap();
    assert_eq!(w[0], 2, "image path (word 3 >= 8): word 0 = dst index");
    assert_eq!(w[1], 0x2222);
    assert_eq!(w[2], 0x3333);
    assert_eq!(w[3], 8 << 28);
    console_heap::unregister(0xC008);
}

#[test]
fn start_is_stable_across_slot_calls() {
    let Some(u) = uma() else { return };
    assert!(console_heap::register(0xC009, 2, 0x8100, 0x9100, 32, false, false, true));
    let v = console_heap::start(0xC009).unwrap();
    let _s = console_heap::slot(v).unwrap();
    assert_eq!(console_heap::start(0xC009).unwrap(), v);
    assert_eq!(u.window_of(console_heap::start(0xC009).unwrap() as usize), Some(0));
    console_heap::unregister(0xC009);
}

// Keep Slot referenced (the tests use it only via inferred bindings).
#[allow(dead_code)]
fn _slot_type_is_used(_: Slot) {}
