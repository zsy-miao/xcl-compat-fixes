//! Stable driver blob identities for translated two-pass pipelines.
use super::{payload_shader, stream_subobjects};
use windows::Win32::Graphics::Direct3D12::*;

/// Bump on any change to two-pass cache key fields or ABI root-signature layout to invalidate stale disk entries.
const TWO_PASS_CACHE_SCHEMA: u32 = 2;
/// ABI root signature layout tag for two-pass pipelines (`graphics = true`: 6 root UAVs + directly indexed heaps + IA flag).
const TWO_PASS_ABI_RS_TAG: u32 = 0x5253_0001;

pub(super) struct Fnv64(u64);

impl Fnv64 {
    pub(super) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &byte in b {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    pub(super) fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.bytes(&[v as u8]);
    }
    pub(super) fn finish(self) -> u64 {
        self.0
    }
}

fn samplers_hash_into(h: &mut Fnv64, samplers: &[crate::gpu::recomp::pso::StaticSampler]) {
    h.u64(samplers.len() as u64);
    for (w, cmp) in samplers {
        for word in w {
            h.u32(*word);
        }
        h.bool(*cmp);
    }
}

unsafe fn read_state<T: Copy>(payload: &[u8]) -> Option<T> {
    let payload = payload.get(..std::mem::size_of::<T>())?;
    Some(std::ptr::read_unaligned(payload.as_ptr().cast()))
}

unsafe fn hash_blend(h: &mut Fnv64, p: &[u8]) -> Option<()> {
    let b = read_state::<D3D12_BLEND_DESC>(p)?;
    h.u32(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND.0 as u32);
    let indep = b.IndependentBlendEnable.0 != 0;
    h.bool(b.AlphaToCoverageEnable.0 != 0);
    h.bool(indep);
    let count = if indep { 8 } else { 1 };
    for rt in &b.RenderTarget[..count] {
        h.bool(rt.BlendEnable.0 != 0);
        h.bool(rt.LogicOpEnable.0 != 0);
        h.i32(rt.SrcBlend.0);
        h.i32(rt.DestBlend.0);
        h.i32(rt.BlendOp.0);
        h.i32(rt.SrcBlendAlpha.0);
        h.i32(rt.DestBlendAlpha.0);
        h.i32(rt.BlendOpAlpha.0);
        h.i32(rt.LogicOp.0);
        h.bytes(&[rt.RenderTargetWriteMask]);
    }
    Some(())
}

unsafe fn hash_rast(h: &mut Fnv64, ty: u32, p: &[u8]) -> Option<()> {
    h.u32(ty);
    match ty {
        10 => {
            let r = read_state::<D3D12_RASTERIZER_DESC>(p)?;
            h.i32(r.FillMode.0);
            h.i32(r.CullMode.0);
            h.bool(r.FrontCounterClockwise.0 != 0);
            h.i32(r.DepthBias);
            h.u32(r.DepthBiasClamp.to_bits());
            h.u32(r.SlopeScaledDepthBias.to_bits());
            h.bool(r.DepthClipEnable.0 != 0);
            h.bool(r.MultisampleEnable.0 != 0);
            h.bool(r.AntialiasedLineEnable.0 != 0);
            h.u32(r.ForcedSampleCount);
            h.i32(r.ConservativeRaster.0);
        }
        27 => {
            let r = read_state::<D3D12_RASTERIZER_DESC1>(p)?;
            h.i32(r.FillMode.0);
            h.i32(r.CullMode.0);
            h.bool(r.FrontCounterClockwise.0 != 0);
            h.u32(r.DepthBias.to_bits());
            h.u32(r.DepthBiasClamp.to_bits());
            h.u32(r.SlopeScaledDepthBias.to_bits());
            h.bool(r.DepthClipEnable.0 != 0);
            h.bool(r.MultisampleEnable.0 != 0);
            h.bool(r.AntialiasedLineEnable.0 != 0);
            h.u32(r.ForcedSampleCount);
            h.i32(r.ConservativeRaster.0);
        }
        28 => {
            let r = read_state::<D3D12_RASTERIZER_DESC2>(p)?;
            h.i32(r.FillMode.0);
            h.i32(r.CullMode.0);
            h.bool(r.FrontCounterClockwise.0 != 0);
            h.u32(r.DepthBias.to_bits());
            h.u32(r.DepthBiasClamp.to_bits());
            h.u32(r.SlopeScaledDepthBias.to_bits());
            h.bool(r.DepthClipEnable.0 != 0);
            h.i32(r.LineRasterizationMode.0);
            h.u32(r.ForcedSampleCount);
            h.i32(r.ConservativeRaster.0);
        }
        _ => return None,
    }
    Some(())
}

unsafe fn hash_depth_stencil(h: &mut Fnv64, ty: u32, p: &[u8]) -> Option<()> {
    let op = |h: &mut Fnv64, o: D3D12_DEPTH_STENCILOP_DESC| {
        h.i32(o.StencilFailOp.0);
        h.i32(o.StencilDepthFailOp.0);
        h.i32(o.StencilPassOp.0);
        h.i32(o.StencilFunc.0);
    };
    let op1 = |h: &mut Fnv64, o: D3D12_DEPTH_STENCILOP_DESC1| {
        h.i32(o.StencilFailOp.0);
        h.i32(o.StencilDepthFailOp.0);
        h.i32(o.StencilPassOp.0);
        h.i32(o.StencilFunc.0);
        h.bytes(&[o.StencilReadMask, o.StencilWriteMask]);
    };
    h.u32(ty);
    match ty {
        11 => {
            let d = read_state::<D3D12_DEPTH_STENCIL_DESC>(p)?;
            h.bool(d.DepthEnable.0 != 0);
            h.i32(d.DepthWriteMask.0);
            h.i32(d.DepthFunc.0);
            h.bool(d.StencilEnable.0 != 0);
            h.bytes(&[d.StencilReadMask, d.StencilWriteMask]);
            op(h, d.FrontFace);
            op(h, d.BackFace);
        }
        21 => {
            let d = read_state::<D3D12_DEPTH_STENCIL_DESC1>(p)?;
            h.bool(d.DepthEnable.0 != 0);
            h.i32(d.DepthWriteMask.0);
            h.i32(d.DepthFunc.0);
            h.bool(d.StencilEnable.0 != 0);
            h.bytes(&[d.StencilReadMask, d.StencilWriteMask]);
            op(h, d.FrontFace);
            op(h, d.BackFace);
            h.bool(d.DepthBoundsTestEnable.0 != 0);
        }
        26 => {
            let d = read_state::<D3D12_DEPTH_STENCIL_DESC2>(p)?;
            h.bool(d.DepthEnable.0 != 0);
            h.i32(d.DepthWriteMask.0);
            h.i32(d.DepthFunc.0);
            h.bool(d.StencilEnable.0 != 0);
            op1(h, d.FrontFace);
            op1(h, d.BackFace);
            h.bool(d.DepthBoundsTestEnable.0 != 0);
        }
        _ => return None,
    }
    Some(())
}

/// Canonical cache key `(hash, total_shader_bytes)` for a two-pass mesh/PS pipeline stream,
/// hashing referenced shader bytecodes and normalized fixed-function state (never pointers or
/// struct padding). Shader pointers in the stream must refer to live shader bytecode. Returns `None` if an unsupported, duplicate, or unknown subobject is active.
pub(super) unsafe fn two_pass_stream_key(stream: &[u8], ps_samplers: &[crate::gpu::recomp::pso::StaticSampler], driver_id: u64, mem_windows: u32) -> Option<(u64, usize)> {
    let mut subs = stream_subobjects(stream)?;
    let total = subs.iter().map(|(ty, payload)| {
        let (_, align) = super::super::null::subobject_payload(*ty).unwrap();
        (align + payload.len() + 7) & !7
    }).sum::<usize>();
    if total != stream.len() {
        return None;
    }
    subs.sort_by_key(|s| s.0);
    if subs.windows(2).any(|w| w[0].0 == w[1].0) {
        return None;
    }
    let mut h = Fnv64::new();
    h.u32(TWO_PASS_CACHE_SCHEMA);
    h.u32(1); // stream kind
    h.u32(TWO_PASS_ABI_RS_TAG);
    h.u32(mem_windows);
    h.u64(driver_id);
    samplers_hash_into(&mut h, ps_samplers);
    let mut shader_bytes = 0usize;
    for (ty, payload) in subs {
        match ty {
            0 | 19 => {} // RS replaced by ABI RS via ps_samplers; CACHED_PSO is the lookup input
            7 | 12 | 22 => {
                if payload.iter().any(|&b| b != 0) {
                    return None;
                }
            }
            1..=6 | 24 | 25 => {
                let code = payload_shader(payload)?;
                shader_bytes = shader_bytes.checked_add(code.len())?;
                h.u32(ty);
                h.u64(code.len() as u64);
                h.bytes(code);
            }
            8 => hash_blend(&mut h, payload)?,
            9 | 13 | 14 | 16 | 17 | 18 | 20 => {
                h.u32(ty);
                h.bytes(payload);
            }
            10 | 27 | 28 => hash_rast(&mut h, ty, payload)?,
            11 | 21 | 26 => hash_depth_stencil(&mut h, ty, payload)?,
            15 => {
                let rt = read_state::<D3D12_RT_FORMAT_ARRAY>(payload)?;
                if rt.NumRenderTargets > 8 {
                    return None;
                }
                let n = rt.NumRenderTargets as usize;
                h.u32(ty);
                h.u32(rt.NumRenderTargets);
                for f in &rt.RTFormats[..n] {
                    h.i32(f.0);
                }
            }
            _ => return None,
        }
    }
    Some((h.finish(), shader_bytes))
}

pub(super) fn two_pass_cs_key(cs: &[u8], samplers: &[crate::gpu::recomp::pso::StaticSampler], driver_id: u64, mem_windows: u32) -> u64 {
    let mut h = Fnv64::new();
    h.u32(TWO_PASS_CACHE_SCHEMA);
    h.u32(2); // compute-pass kind
    h.u32(TWO_PASS_ABI_RS_TAG);
    h.u32(mem_windows);
    h.u64(driver_id);
    h.u64(cs.len() as u64);
    h.bytes(cs);
    samplers_hash_into(&mut h, samplers);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::StreamOut;
    use std::ffi::c_void;

    #[test]
    fn canonical_stream_key_distinguishes_state_samplers_and_driver() {
        let ms = [1u8, 2, 3, 4];
        let ps = [5u8, 6, 7, 8, 9];
        let rs_a: *mut c_void = 0x1000 as *mut c_void;
        let rs_b: *mut c_void = 0x2000 as *mut c_void;
        let mut s1 = StreamOut::default();
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs_a);
        s1.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s1.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &45i32);

        let mut s2 = StreamOut::default();
        // Same shaders/state, different RS pointer and extra CACHED_PSO: must keep the same key.
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &45i32);
        s2.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        s2.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs_b);
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
        let blob = [0xAAu8; 8];
        s2.put(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
            &D3D12_CACHED_PIPELINE_STATE { pCachedBlob: blob.as_ptr() as _, CachedBlobSizeInBytes: blob.len() },
        );

        unsafe {
            let k1 = two_pass_stream_key(&s1.0, &[], 0x1234, 1).unwrap();
            let k2 = two_pass_stream_key(&s2.0, &[], 0x1234, 1).unwrap();
            assert_eq!(k1, k2);
            assert_eq!(k1.1, ms.len() + ps.len());
            assert_ne!(k1, two_pass_stream_key(&s1.0, &[([1, 2, 3, 4], false)], 0x1234, 1).unwrap());
            assert_ne!(k1, two_pass_stream_key(&s1.0, &[], 0x5678, 1).unwrap());
        }
    }

    #[test]
    fn canonical_stream_key_normalizes_padding_and_rejects_pointer_subobjects() {
        let mut s1 = StreamOut::default();
        let mut rt1 = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rt1.RTFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28);
        rt1.RTFormats[7] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(999);
        let mut blend1 = D3D12_BLEND_DESC::default();
        blend1.IndependentBlendEnable = false.into();
        blend1.RenderTarget[3].RenderTargetWriteMask = 7; // unused when IndependentBlendEnable is false
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt1);
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend1);

        let mut s2 = StreamOut::default();
        let mut rt2 = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rt2.RTFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28);
        let mut blend2 = D3D12_BLEND_DESC::default();
        blend2.IndependentBlendEnable = false.into();
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt2);
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend2);

        let mut ds_a = [0; std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC1>()];
        let mut ds_b = ds_a;
        let after_masks = std::mem::offset_of!(D3D12_DEPTH_STENCIL_DESC1, StencilWriteMask) + 1;
        let first_face = std::mem::offset_of!(D3D12_DEPTH_STENCIL_DESC1, FrontFace);
        assert!(after_masks < first_face);
        ds_a[after_masks..first_face].fill(0xaa);
        ds_b[after_masks..first_face].fill(0x55);
        let depth = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1.0 as u32;
        s1.raw(depth, &ds_a, false);
        s2.raw(depth, &ds_b, false);

        unsafe {
            assert_eq!(two_pass_stream_key(&s1.0, &[], 1, 1), two_pass_stream_key(&s2.0, &[], 1, 1));
            // Unknown subobject or active pointer subobject returns None safely.
            let mut with_il = StreamOut::default();
            let il = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: 0x1234 as *const _, NumElements: 1 };
            with_il.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &il);
            assert_eq!(two_pass_stream_key(&with_il.0, &[], 1, 1), None);
        }
    }

    #[test]
    fn two_pass_keys_cover_mem_windows_invalid_counts_and_duplicates() {
        let mut s = StreamOut::default();
        let ms = [1u8, 2, 3, 4];
        s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0);
        unsafe {
            assert_ne!(two_pass_stream_key(&s.0, &[], 1, 1), two_pass_stream_key(&s.0, &[], 1, 4 | crate::gpu::recomp::HEAP_WINDOWS));
            // Duplicate subobjects fail closed.
            let mut dup = StreamOut(s.0.clone(), s.1);
            dup.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0);
            assert_eq!(two_pass_stream_key(&dup.0, &[], 1, 1), None);
            // Invalid NumRenderTargets > 8 fails closed.
            let mut bad_rt = StreamOut::default();
            let rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 9, ..Default::default() };
            bad_rt.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
            assert_eq!(two_pass_stream_key(&bad_rt.0, &[], 1, 1), None);
        }
        let cs = [10u8; 64];
        assert_ne!(two_pass_cs_key(&cs, &[], 1, 1), two_pass_cs_key(&cs, &[], 1, 4));
    }

    #[test]
    fn two_pass_cs_key_changes_with_samplers_and_driver() {
        let cs = [10u8; 64];
        let k0 = two_pass_cs_key(&cs, &[], 1, 1);
        assert_eq!(k0, two_pass_cs_key(&cs, &[], 1, 1));
        assert_ne!(k0, two_pass_cs_key(&cs, &[([0, 1, 2, 3], false)], 1, 1));
        assert_ne!(k0, two_pass_cs_key(&cs, &[([0, 1, 2, 3], true)], 1, 1));
        assert_ne!(k0, two_pass_cs_key(&cs, &[], 2, 1));
        assert_ne!(k0, two_pass_cs_key(&[11u8; 64], &[], 1, 1));
    }
}
