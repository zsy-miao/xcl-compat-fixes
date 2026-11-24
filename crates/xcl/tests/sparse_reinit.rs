//! The sparse windows set up again while the state they replace holds resources of its own
//! (#344): the old state was dropped under its lock, and the release of its last buffers
//! (`real::res_release` → `sparse::forget_buffer`) took the lock again – the thread waited for
//! itself. Its own process: when the test fails, that thread stays blocked, holding the lock.

use std::time::Duration;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use xcl::hle::d3d12x::{real, sparse};

#[test]
fn the_sparse_state_is_replaced_outside_its_lock() {
    if real::init().is_err() {
        xcl::gpu::recomp::runner::skip_test("no GPU");
        return;
    }
    // no sparse windows on the GPU (WARP makes no reserved buffers that large): no state to replace
    if real::uma_gpu_va() == 0 {
        xcl::gpu::recomp::runner::skip_test("no GPU UMA");
        return;
    }
    unsafe {
        // the state in use: a block of the windows, written and read back (its readback buffer is
        // the state's own)
        let (va, len) = sparse::alloc(0x1_0000, 0x1_0000).expect("a sparse block");
        std::ptr::write_bytes(va as *mut u8, 0x5A, len);
        sparse::readback_now(va, len);
        // set up again, as two first `real::init` calls at once do
        let dev = real::device_raw() as usize;
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let dev = ID3D12Device::from_raw_borrowed(&(dev as *mut std::ffi::c_void)).expect("the device").clone();
            let _ = done.send(sparse::init(&dev).len());
        });
        let windows = wait.recv_timeout(Duration::from_secs(60)).expect("sparse::init did not return within 60 s: it waits for the lock it holds");
        assert!(windows > 0, "the sparse windows were set up again");
        assert!(sparse::free(va));
    }
}
