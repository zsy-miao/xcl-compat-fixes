//! Acceleration structures in the console's format: the RDNA2 nodes `image_bvh(64)_intersect_ray`
//! reads (as the recompiler's `bvh_intersect` emulates them) plus what the game's ray tracing
//! shaders read themselves (`CS_RTShadowTrace*`, see HANDOFF):
//! * the structure starts with a 256-byte header; dword @0x30 = root node id; node ids are
//!   (offset from the structure's start / 8) | type, nodes 64-byte aligned, root always a box;
//! * box32 (type 5), 128 B: child ids u32[4] (unused = 0xFFFFFFFF), then per child
//!   min xyz, max xyz (f32) at 16 + 24·k;
//! * box16 (type 4), 64 B: child ids, then per child min / max as half floats (12 B) – read,
//!   not built;
//! * triangle (types 0 / 1), 64 B, AMD GPURT gfx10 layout (confirmed on the console builder's
//!   own output): a pair of triangles, v0..v3 (f32 xyz), type 0 = (v0, v1, v2), type 1 =
//!   (v1, v3, v2); @48 = geometry index << 8 | triangle 1 present << 7 | opaque (box nodes
//!   point at type 0 only; the game's shader tests type 1 next when bit 7 is set and uses
//!   bit 0 for the opaque decision); @52 / @56 primitive index (within its geometry) of
//!   triangle 0 / 1;
//!   @60 triangle id: per triangle a byte, bits 1:0 / 3:2 = which vertex the I / J barycentric
//!   belongs to, bit 7 opaque. One triangle per node here (triangle id 9 = no rotation);
//! * instance (type 6), 128 B, traversed by shader code: @48 id (24 b) | mask << 24,
//!   @52 hit group contribution (24 b) | flags << 24, @56 bottom-level root id (30 b) | bit 31 =
//!   transform is a pure translation, @60 bottom-level address >> 8, @64 world → object as
//!   [t.xyz, row 0, row 1, row 2]. Assumed (not read by the shaders seen): @0 object → world
//!   3×4 as in D3D12_RAYTRACING_INSTANCE_DESC; @112..127 zero.
//!
//! The console builder reports sizes for exactly these node sizes plus 4 bytes of parent
//! pointers per 64 bytes of nodes (`tests/console_dlls_e2e.rs`); where those pointers live is
//! unknown and no shader seen reads them, so they are not written.
//!
//! Top-down build: median split of the leaf box centroids on the longest axis, twice per box
//! node (4 children), one triangle / instance per leaf.

pub const BOX32: u32 = 5;
pub const INSTANCE: u32 = 6;
pub const INVALID: u32 = u32::MAX;
pub const HEADER_BYTES: usize = 256;
/// header dword holding the root node id
pub const ROOT_AT: usize = 0x30;
const BOX_BYTES: usize = 128;
const TRI_BYTES: usize = 64;
const INSTANCE_BYTES: usize = 128;
/// triangle id byte: I = vertex 1, J = vertex 2 (no rotation)
const TRI_ID_PLAIN: u32 = 9;

pub type Triangle = [[f32; 3]; 3];
pub type Aabb = ([f32; 3], [f32; 3]);

/// A built structure: bytes to place at a 256-byte aligned GPU address (descriptor word 0 =
/// address >> 8)
pub struct Bvh {
    pub data: Vec<u8>,
    pub root: u32,
}

/// One top-level instance (D3D12_RAYTRACING_INSTANCE_DESC plus the bottom level's root and bounds)
#[derive(Clone, Debug)]
pub struct Instance {
    /// object → world, row-major 3×4
    pub transform: [[f32; 4]; 3],
    pub id: u32,
    pub mask: u8,
    pub contribution: u32,
    pub flags: u8,
    /// GPU address of the bottom level (256-byte aligned), its root id and its local bounds
    pub blas_address: u64,
    pub blas_root: u32,
    pub blas_bounds: Aabb,
}

/// Worst-case console acceleration structure size (bytes) matching `XboxBvhBuilder.dll`'s
/// `GetRaytracingAccelerationStructurePrebuildInfo`: 256-byte header, `leaf_count` leaves
/// (64 B triangles for BLAS, 128 B instances for TLAS), `max(leaf_count, 2) - 1` box32 nodes
/// (128 B), plus 4 bytes of parent pointers per 64 bytes of nodes, rounded up to 1 KiB.
pub fn prebuild_size(leaf_count: u64, top_level: bool) -> u64 {
    let leaf = if top_level { INSTANCE_BYTES as u64 } else { TRI_BYTES as u64 };
    let nodes = leaf_count * leaf + (leaf_count.max(2) - 1) * BOX_BYTES as u64;
    (HEADER_BYTES as u64 + nodes + nodes / 64 * 4).div_ceil(1024) * 1024
}

/// Whether a console BVH header has bit 24 ("compressed / dehydrated", as stored in
/// `*_bvh4.rpf` before `XboxBvhBuilder::Rehydrate*`) set in dword 0.
pub fn is_compressed_header(data: &[u8]) -> bool {
    data.len() >= 4 && ((word(data, 0) >> 24) & 1) != 0
}

/// Bottom level over `tris`; `geometry` goes to every triangle node.
pub fn build(tris: &[Triangle], geometry: u32) -> Bvh {
    let bounds: Vec<Aabb> = tris.iter().map(|t| bounds(t.iter().copied())).collect();
    let leaf = |i: usize, out: &mut [u8]| {
        let v: Vec<u32> = tris[i].iter().flatten().map(|x| x.to_bits()).collect();
        put(out, 0, &v);
        // single triangle (bit 7 clear), opaque
        put(out, 48, &[geometry << 8 | 1, i as u32, INVALID, TRI_ID_PLAIN | 1 << 7]);
        0
    };
    Builder { data: Vec::new(), bounds, leaf_bytes: TRI_BYTES, leaf: &leaf }.finish()
}

/// Top level over `instances`.
pub fn build_top(instances: &[Instance]) -> Bvh {
    let bounds: Vec<Aabb> = instances.iter().map(|i| transform_aabb(&i.transform, i.blas_bounds)).collect();
    let leaf = |i: usize, out: &mut [u8]| {
        let inst = &instances[i];
        let m = &inst.transform;
        let o2w: Vec<u32> = m.iter().flatten().map(|x| x.to_bits()).collect();
        put(out, 0, &o2w);
        let translation_only = (0..3).all(|r| (0..3).all(|c| m[r][c] == if r == c { 1.0 } else { 0.0 }));
        put(
            out,
            48,
            &[
                inst.id & 0xFF_FFFF | (inst.mask as u32) << 24,
                inst.contribution & 0xFF_FFFF | (inst.flags as u32) << 24,
                inst.blas_root & 0x3FFF_FFFF | (translation_only as u32) << 31,
                (inst.blas_address >> 8) as u32,
            ],
        );
        let w2o: Vec<u32> = world_to_object(m).iter().map(|x| x.to_bits()).collect();
        put(out, 64, &w2o);
        INSTANCE
    };
    Builder { data: Vec::new(), bounds, leaf_bytes: INSTANCE_BYTES, leaf: &leaf }.finish()
}

fn put(out: &mut [u8], at: usize, words: &[u32]) {
    for (k, w) in words.iter().enumerate() {
        out[at + 4 * k..at + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
    }
}

struct Builder<'a> {
    data: Vec<u8>,
    bounds: Vec<Aabb>,
    leaf_bytes: usize,
    /// writes leaf `i` into its zeroed node bytes, returns the node type
    leaf: &'a dyn Fn(usize, &mut [u8]) -> u32,
}

impl Builder<'_> {
    fn finish(mut self) -> Bvh {
        self.data = vec![0; HEADER_BYTES];
        let mut idx: Vec<u32> = (0..self.bounds.len() as u32).collect();
        let root = if idx.is_empty() { INVALID } else { self.node(&mut idx, true) };
        let mut data = self.data;
        put(&mut data, ROOT_AT, &[root]);
        Bvh { data, root }
    }

    fn alloc(&mut self, n: usize) -> usize {
        let at = self.data.len();
        self.data.resize(at + n, 0);
        at
    }

    fn node(&mut self, idx: &mut [u32], root: bool) -> u32 {
        if idx.len() == 1 && !root {
            let at = self.alloc(self.leaf_bytes);
            let ty = (self.leaf)(idx[0] as usize, &mut self.data[at..at + self.leaf_bytes]);
            return (at / 8) as u32 | ty;
        }
        let at = self.alloc(BOX_BYTES);
        let boxes = std::mem::take(&mut self.bounds);
        let groups: Vec<&mut [u32]> = if idx.len() <= 4 {
            idx.chunks_mut(1).collect()
        } else {
            let (a, b) = split(&boxes, idx);
            let (a1, a2) = split(&boxes, a);
            let (b1, b2) = split(&boxes, b);
            vec![a1, a2, b1, b2]
        };
        let mut children = [INVALID; 4];
        let mut ranges = vec![];
        for (k, g) in groups.into_iter().enumerate() {
            let (mn, mx) = bounds(g.iter().flat_map(|&i| [boxes[i as usize].0, boxes[i as usize].1]));
            let b: Vec<u32> = mn.iter().chain(&mx).map(|x| x.to_bits()).collect();
            put(&mut self.data, at + 16 + 24 * k, &b);
            ranges.push((k, g));
        }
        self.bounds = boxes;
        for (k, g) in ranges {
            children[k] = self.node(g, false);
        }
        put(&mut self.data, at, &children);
        (at / 8) as u32 | BOX32
    }
}

fn bounds(points: impl Iterator<Item = [f32; 3]>) -> Aabb {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            mn[a] = mn[a].min(p[a]);
            mx[a] = mx[a].max(p[a]);
        }
    }
    (mn, mx)
}

/// halves at the median box centroid along the longest centroid extent
fn split<'s>(boxes: &[Aabb], idx: &'s mut [u32]) -> (&'s mut [u32], &'s mut [u32]) {
    let c = |i: u32, a: usize| boxes[i as usize].0[a] + boxes[i as usize].1[a];
    let (mn, mx) = bounds(idx.iter().map(|&i| [0, 1, 2].map(|a| c(i, a))));
    let axis = (0..3).max_by(|&a, &b| (mx[a] - mn[a]).total_cmp(&(mx[b] - mn[b]))).unwrap();
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| c(a, axis).total_cmp(&c(b, axis)));
    idx.split_at_mut(mid)
}

fn apply(m: &[[f32; 4]; 3], p: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|r| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3])
}

fn transform_aabb(m: &[[f32; 4]; 3], (mn, mx): Aabb) -> Aabb {
    bounds((0..8).map(|k| apply(m, [if k & 1 == 0 { mn[0] } else { mx[0] }, if k & 2 == 0 { mn[1] } else { mx[1] }, if k & 4 == 0 { mn[2] } else { mx[2] }])))
}

/// inverse of an affine object → world, in the instance node's order [t, row 0, row 1, row 2]
fn world_to_object(m: &[[f32; 4]; 3]) -> [f32; 12] {
    let a = |r: usize, c: usize| m[r][c] as f64;
    let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    let inv = [
        [a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1), a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2), a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)],
        [a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2), a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0), a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)],
        [a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0), a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1), a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)],
    ]
    .map(|row| row.map(|x| x / det));
    let t = [0, 1, 2].map(|r| -(inv[r][0] * a(0, 3) + inv[r][1] * a(1, 3) + inv[r][2] * a(2, 3)));
    let mut out = [0f32; 12];
    for r in 0..3 {
        out[r] = t[r] as f32;
        for c in 0..3 {
            out[3 + 3 * r + c] = inv[r][c] as f32;
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Möller–Trumbore as the triangle node returns it: (t·det, det, i·det, j·det), `None` = miss
pub fn triangle_hit(v: &Triangle, o: [f32; 3], d: [f32; 3]) -> Option<[f32; 4]> {
    let (e1, e2) = (sub(v[1], v[0]), sub(v[2], v[0]));
    let p = cross(d, e2);
    let det = dot(e1, p);
    let sv = sub(o, v[0]);
    let q = cross(sv, e1);
    let (i, j, t) = (dot(sv, p), dot(d, q), dot(e2, q));
    let hit = if det > 0.0 {
        i >= 0.0 && j >= 0.0 && i + j <= det
    } else {
        det < 0.0 && i <= 0.0 && j <= 0.0 && i + j >= det
    };
    hit.then_some([t, det, i, j])
}

fn word(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}
fn vec3(data: &[u8], at: usize) -> [f32; 3] {
    [0, 1, 2].map(|k| f32::from_bits(word(data, at + 4 * k)))
}
fn node_at(node: u32) -> usize {
    ((node & !7) as usize) << 3
}

fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 0x1F, h & 0x3FF);
    let v = match e {
        0 => m as f32 * (2.0f32).powi(-24),
        31 => if m == 0 { f32::INFINITY } else { f32::NAN },
        _ => f32::from_bits((e + 112) << 23 | m << 13),
    };
    if s == 1 { -v } else { v }
}

/// One `image_bvh_intersect_ray` on the CPU, same results as the HLSL emulation: box32 /
/// box16 → child ids hit within `tmax` (closest first when `sort`), triangle → (t·det, det,
/// I·det, J·det) bits (barycentrics in the original vertex order) or (inf, 1, 0, 0); other
/// types → 0xFFFFFFFF ×4.
pub fn intersect_node(data: &[u8], node: u32, tmax: f32, o: [f32; 3], d: [f32; 3], inv: [f32; 3], sort: bool) -> [u32; 4] {
    let at = node_at(node);
    match node & 7 {
        4 | BOX32 => {
            let mut hits = [(f32::INFINITY, INVALID); 4];
            for (k, h) in hits.iter_mut().enumerate() {
                let child = word(data, at + 4 * k);
                let (mn, mx) = if node & 7 == BOX32 {
                    (vec3(data, at + 16 + 24 * k), vec3(data, at + 28 + 24 * k))
                } else {
                    let w = [0, 1, 2].map(|j| word(data, at + 16 + 12 * k + 4 * j));
                    ([half(w[0]), half(w[0] >> 16), half(w[1])], [half(w[1] >> 16), half(w[2]), half(w[2] >> 16)])
                };
                let valid = child != INVALID && (0..3).all(|a| mn[a] <= mx[a]);
                let (mut a, mut b) = (f32::NEG_INFINITY, f32::INFINITY);
                for axis in 0..3 {
                    let (lo, hi) = if inv[axis].is_infinite() {
                        if o[axis] >= mn[axis] && o[axis] <= mx[axis] {
                            (f32::NEG_INFINITY, f32::INFINITY)
                        } else {
                            (f32::INFINITY, f32::NEG_INFINITY)
                        }
                    } else {
                        let t1 = (mn[axis] - o[axis]) * inv[axis];
                        let t2 = (mx[axis] - o[axis]) * inv[axis];
                        (t1.min(t2), t1.max(t2))
                    };
                    a = a.max(lo);
                    b = b.min(hi);
                }
                if valid && a <= b && b >= 0.0 && a <= tmax {
                    *h = (a, child);
                }
            }
            if sort {
                for (i, j) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
                    if hits[j].0 < hits[i].0 {
                        hits.swap(i, j);
                    }
                }
            }
            hits.map(|h| h.1)
        }
        0..=3 => {
            let ty = node & 7;
            let (o0, o1, o2) = match ty {
                0 => (0, 12, 24),
                1 => (12, 36, 24),
                2 => (24, 36, 48),
                _ => (24, 48, 0),
            };
            let v = [vec3(data, at + o0), vec3(data, at + o1), vec3(data, at + o2)];
            match triangle_hit(&v, o, d) {
                Some([t, det, i, j]) => {
                    let bc = [det - i - j, i, j, 0.0];
                    let id = word(data, at + 60) >> (8 * ty);
                    [t, det, bc[(id & 3) as usize], bc[((id >> 2) & 3) as usize]].map(f32::to_bits)
                }
                None => [f32::INFINITY.to_bits(), 1.0f32.to_bits(), 0, 0],
            }
        }
        _ => [INVALID; 4],
    }
}

/// closest hit (t, primitive, instance id) below `root`; instance nodes go to `instance`
fn closest(data: &[u8], root: u32, o: [f32; 3], d: [f32; 3], tmax: f32, instance: &mut dyn FnMut(&[u8], u32, f32) -> Option<(f32, u32, u32)>) -> Option<(f32, u32, u32)> {
    let inv = d.map(|x| 1.0 / x);
    let mut best: Option<(f32, u32, u32)> = None;
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n == INVALID {
            continue;
        }
        let limit = best.map_or(tmax, |b| b.0);
        match n & 7 {
            INSTANCE => {
                if let Some(h) = instance(data, n, limit) {
                    best = Some(h);
                }
            }
            4 | BOX32 => {
                // closest child popped first
                let r = intersect_node(data, n, limit, o, d, inv, true);
                stack.extend(r.iter().rev().filter(|&&c| c != INVALID));
            }
            0..=3 => {
                // second triangle of a pair: tested as type 1, as the game's shader does
                if n & 7 == 0 && word(data, node_at(n) + 48) & 0x80 != 0 {
                    stack.push(n | 1);
                }
                let r = intersect_node(data, n, limit, o, d, inv, true).map(f32::from_bits);
                let t = r[0] / r[1];
                if r[0] != f32::INFINITY && t >= 0.0 && t < limit {
                    best = Some((t, word(data, node_at(n) + if n & 7 == 1 { 56 } else { 52 }), 0));
                }
            }
            _ => {}
        }
    }
    best
}

impl Bvh {
    /// Closest hit in a bottom level, as a shader's traversal loop finds it: (t, primitive).
    pub fn trace(&self, o: [f32; 3], d: [f32; 3], tmax: f32) -> Option<(f32, u32)> {
        closest(&self.data, self.root, o, d, tmax, &mut |_, _, _| None).map(|h| (h.0, h.1))
    }

    /// Closest hit through a top level as the game's shaders traverse it (instance node →
    /// ray to object space → bottom level found by address): (t, primitive, instance id).
    pub fn trace_scene<'b>(&self, blas: &dyn Fn(u64) -> Option<&'b [u8]>, o: [f32; 3], d: [f32; 3], tmax: f32) -> Option<(f32, u32, u32)> {
        closest(&self.data, self.root, o, d, tmax, &mut |data, n, limit| {
            let at = node_at(n);
            let (w48, w56, w60) = (word(data, at + 48), word(data, at + 56), word(data, at + 60));
            let f = |k: usize| f32::from_bits(word(data, at + 64 + 4 * k));
            let (o2, d2) = if w56 >> 31 != 0 {
                ([o[0] + f(0), o[1] + f(1), o[2] + f(2)], d)
            } else {
                let lin = |v: [f32; 3], r: usize| f(3 + 3 * r) * v[0] + f(4 + 3 * r) * v[1] + f(5 + 3 * r) * v[2];
                ([0, 1, 2].map(|r| lin(o, r) + f(r)), [0, 1, 2].map(|r| lin(d, r)))
            };
            let b = blas((w60 as u64) << 8)?;
            closest(b, w56 & 0x3FFF_FFFF, o2, d2, limit, &mut |_, _, _| None).map(|h| (h.0, h.1, w48 & 0xFF_FFFF))
        })
    }

    /// bounds of everything below the root
    pub fn bounds(&self) -> Aabb {
        let at = node_at(self.root);
        let kids = (0..4).filter(|&k| word(&self.data, at + 4 * k) != INVALID);
        bounds(kids.flat_map(|k| [vec3(&self.data, at + 16 + 24 * k), vec3(&self.data, at + 28 + 24 * k)]))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn rng(seed: u64) -> impl FnMut(f32, f32) -> f32 {
        let mut s = seed;
        move |lo, hi| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        }
    }

    pub fn random_tris(n: usize, seed: u64) -> Vec<Triangle> {
        let mut r = rng(seed);
        (0..n)
            .map(|_| {
                let c = [r(-8.0, 8.0), r(-8.0, 8.0), r(-8.0, 8.0)];
                [0; 3].map(|_| [0, 1, 2].map(|a| c[a] + r(-1.5, 1.5)))
            })
            .collect()
    }

    /// the console builder's worst-case size (tests/console_dlls_e2e.rs) without the parent pointers
    fn console_nodes_bound(n: usize, leaf: usize) -> usize {
        HEADER_BYTES + n * leaf + (n.max(2) - 1) * BOX_BYTES
    }

    #[test]
    fn trace_matches_brute_force() {
        let tris = random_tris(300, 7);
        let bvh = build(&tris, 3);
        assert_eq!(bvh.root & 7, BOX32);
        assert_eq!(word(&bvh.data, ROOT_AT), bvh.root);
        assert_eq!(node_at(bvh.root), HEADER_BYTES);
        assert_eq!(bvh.data.len() % 64, 0);
        assert!(bvh.data.len() <= console_nodes_bound(300, TRI_BYTES));
        let mut r = rng(99);
        let mut hits = 0;
        for _ in 0..2000 {
            let o = [r(-12.0, 12.0), r(-12.0, 12.0), r(-12.0, 12.0)];
            let to = [r(-6.0, 6.0), r(-6.0, 6.0), r(-6.0, 6.0)];
            let d = sub(to, o);
            let brute = tris
                .iter()
                .enumerate()
                .filter_map(|(i, t)| triangle_hit(t, o, d).map(|h| (h[0] / h[1], i as u32)))
                .filter(|&(t, _)| (0.0..100.0).contains(&t))
                .min_by(|a, b| a.0.total_cmp(&b.0));
            let got = bvh.trace(o, d, 100.0);
            assert_eq!(got.map(|g| g.1), brute.map(|b| b.1), "ray {o:?} → {to:?}");
            if let (Some(g), Some(b)) = (got, brute) {
                assert!((g.0 - b.0).abs() <= 1e-5 * b.0.max(1.0));
                hits += 1;
            }
        }
        assert!(hits > 300, "{hits} hits: the test rays should hit often");
    }

    #[test]
    fn small_inputs() {
        assert_eq!(build(&[], 0).root, INVALID);
        // a lone triangle still sits under a box root
        let one = build(&random_tris(1, 1), 9);
        assert_eq!(one.data.len(), HEADER_BYTES + BOX_BYTES + TRI_BYTES);
        assert_eq!(word(&one.data, HEADER_BYTES), ((HEADER_BYTES + BOX_BYTES) / 8) as u32);
        let tri = HEADER_BYTES + BOX_BYTES;
        assert_eq!(word(&one.data, tri + 48), 9 << 8 | 1); // geometry, single, opaque
        assert_eq!(word(&one.data, tri + 52), 0); // primitive
        assert_eq!(word(&one.data, tri + 60) & 0xFF, 0x89); // no rotation, opaque
        let three = build(&random_tris(3, 2), 0);
        assert_eq!(word(&three.data, HEADER_BYTES + 12), INVALID); // 3 children, 4th unused
    }

    fn rotation(r: &mut impl FnMut(f32, f32) -> f32) -> [[f32; 3]; 3] {
        let q = [r(-1.0, 1.0), r(-1.0, 1.0), r(-1.0, 1.0), r(-1.0, 1.0)];
        let n = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let [w, x, y, z] = q.map(|v| v / n);
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    }

    #[test]
    fn instance_node_layout() {
        let inst = Instance {
            transform: [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 6.0], [0.0, 0.0, 1.0, 7.0]],
            id: 0x123456,
            mask: 0xE5,
            contribution: 0xABCDEF,
            flags: 4,
            blas_address: 0x12_3456_7800,
            blas_root: 0x25,
            blas_bounds: ([-1.0; 3], [1.0; 3]),
        };
        let top = build_top(&[inst]);
        let at = HEADER_BYTES + BOX_BYTES;
        assert_eq!(word(&top.data, HEADER_BYTES), (at / 8) as u32 | INSTANCE);
        assert_eq!(word(&top.data, at + 48), 0xE512_3456);
        assert_eq!(word(&top.data, at + 52), 0x04AB_CDEF);
        assert_eq!(word(&top.data, at + 56), 0x8000_0025); // identity rotation: translation-only bit
        assert_eq!(word(&top.data, at + 60), 0x1234_5678);
        let w2o: Vec<f32> = (0..12).map(|k| f32::from_bits(word(&top.data, at + 64 + 4 * k))).collect();
        assert_eq!(w2o, [-5.0, -6.0, -7.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        // instance box = the bottom level's bounds moved to the instance
        assert_eq!(top.bounds(), ([4.0, 5.0, 6.0], [6.0, 7.0, 8.0]));
    }

    /// Two levels: 3 bottom levels, 24 instances (rotated + scaled + moved, some only moved),
    /// traced as the game's shader does vs. every world-space triangle by brute force.
    #[test]
    fn scene_trace_matches_brute_force() {
        let mut r = rng(31);
        let blases: Vec<(Vec<Triangle>, Bvh)> = (0..3)
            .map(|k| {
                let t: Vec<Triangle> = random_tris(40 + 10 * k, 100 + k as u64).into_iter().map(|t| t.map(|v| v.map(|x| x * 0.25))).collect();
                let b = build(&t, 0);
                (t, b)
            })
            .collect();
        let addr = |k: usize| 0x10_0000u64 * (k as u64 + 1);
        let instances: Vec<Instance> = (0..24)
            .map(|i| {
                let k = i % 3;
                let s = [r(0.5, 2.0), r(0.5, 2.0), r(0.5, 2.0)];
                let rot = if i % 4 == 0 { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] } else { rotation(&mut r) };
                let s = if i % 4 == 0 { [1.0; 3] } else { s };
                let t = [r(-20.0, 20.0), r(-20.0, 20.0), r(-20.0, 20.0)];
                let m = [0, 1, 2].map(|row| [rot[row][0] * s[0], rot[row][1] * s[1], rot[row][2] * s[2], t[row]]);
                Instance { transform: m, id: 1000 + i as u32, mask: 0xFF, contribution: 0, flags: 0, blas_address: addr(k), blas_root: blases[k].1.root, blas_bounds: blases[k].1.bounds() }
            })
            .collect();
        let top = build_top(&instances);
        assert!(top.data.len() <= console_nodes_bound(24, INSTANCE_BYTES));
        let lookup = |a: u64| (0..3).find(|&k| addr(k) == a).map(|k| blases[k].1.data.as_slice());
        let mut hits = 0;
        for _ in 0..3000 {
            let o = [r(-30.0, 30.0), r(-30.0, 30.0), r(-30.0, 30.0)];
            let to = [r(-20.0, 20.0), r(-20.0, 20.0), r(-20.0, 20.0)];
            let d = sub(to, o);
            let brute = instances
                .iter()
                .flat_map(|inst| {
                    let k = (0..3).find(|&k| addr(k) == inst.blas_address).unwrap();
                    blases[k].0.iter().enumerate().filter_map(move |(p, t)| {
                        let w = t.map(|v| apply(&inst.transform, v));
                        triangle_hit(&w, o, d).map(|h| (h[0] / h[1], p as u32, inst.id))
                    })
                })
                .filter(|h| (0.0..100.0).contains(&h.0))
                .min_by(|a, b| a.0.total_cmp(&b.0));
            let got = top.trace_scene(&lookup, o, d, 100.0);
            assert_eq!(got.map(|g| (g.1, g.2)), brute.map(|b| (b.1, b.2)), "ray {o:?} → {to:?}");
            if let (Some(g), Some(b)) = (got, brute) {
                assert!((g.0 - b.0).abs() <= 1e-3 * b.0.max(1.0), "t {} vs {}", g.0, b.0);
                hits += 1;
            }
        }
        assert!(hits > 100, "{hits} hits");
    }

    #[test]
    fn prebuild_size_and_compressed_header() {
        // matches XboxBvhBuilder.dll's GetRaytracingAccelerationStructurePrebuildInfo formula
        assert_eq!(prebuild_size(1, false), 1024);
        assert_eq!(prebuild_size(64, false), 13312);
        assert_eq!(prebuild_size(16, true), 5120);
        let compressed = 0x8102_0006u32.to_le_bytes();
        let rehydrated = 0x8002_0006u32.to_le_bytes();
        assert!(is_compressed_header(&compressed));
        assert!(!is_compressed_header(&rehydrated));
        assert!(!is_compressed_header(&[]));
    }

    #[test]
    fn grazing_axis_aligned_ray_and_nan_box() {
        let mut data = vec![0u8; 256];
        let (c0, c1, c2, c3) = (0x10u32, 0x20u32, 0x30u32, 0x40u32);
        put(&mut data, 0, &[c0, c1, c2, c3]);
        let box_f = |mn: [f32; 3], mx: [f32; 3]| [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]].map(f32::to_bits);
        // child 0: normal box [-1..1, -1..1, 5..6]; ray origin lies right on x = -1.0 with dx = 0
        put(&mut data, 16, &box_f([-1.0, -1.0, 5.0], [1.0, 1.0, 6.0]));
        // child 1: NaN-disabled box (GPURT convention)
        put(&mut data, 40, &box_f([f32::NAN, -1.0, 2.0], [1.0, 1.0, 3.0]));
        // child 2: inverted empty box (mn.x > mx.x)
        put(&mut data, 64, &box_f([2.0, -1.0, 2.0], [-2.0, 1.0, 3.0]));
        // child 3: box missed along x (x in [0..1] while o.x = -1.0)
        put(&mut data, 88, &box_f([0.0, -1.0, 5.0], [1.0, 1.0, 6.0]));

        let o = [-1.0, 0.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let inv = [f32::INFINITY, f32::INFINITY, 1.0];
        let got = intersect_node(&data, BOX32, 100.0, o, d, inv, true);
        assert_eq!(got, [c0, INVALID, INVALID, INVALID]);
    }

    #[test]
    fn triangle_hit_small_and_large_det_avoids_underflow_and_overflow() {
        let d = [0.0, 0.0, 1.0];
        // Tiny triangles where det * det underflows to subnormal (1e-40) or 0.0 (1e-48)
        for s in [1e-10f32, 1e-12f32] {
            let tri = [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0]];
            assert!(triangle_hit(&tri, [2.0 * s, 2.0 * s, -1.0], d).is_none(), "s={s}: outside ray must miss");
            assert!(triangle_hit(&tri, [0.2 * s, 0.2 * s, -1.0], d).is_some(), "s={s}: inside ray must hit");
            // Opposite winding (det > 0)
            let rev = [[0.0, 0.0, 0.0], [0.0, s, 0.0], [s, 0.0, 0.0]];
            assert!(triangle_hit(&rev, [2.0 * s, 2.0 * s, -1.0], d).is_none(), "s={s} rev: outside ray must miss");
            assert!(triangle_hit(&rev, [0.2 * s, 0.2 * s, -1.0], d).is_some(), "s={s} rev: inside ray must hit");
        }
        // Huge triangle where det * det overflows to +inf
        let big = [[0.0, 0.0, 0.0], [1e10, 0.0, 0.0], [0.0, 1e10, 0.0]];
        assert!(triangle_hit(&big, [2e10, 2e10, -1.0], d).is_none());
        assert!(triangle_hit(&big, [2e9, 2e9, -1.0], d).is_some());
    }

    #[test]
    fn triangle_node_types_0_to_3_select_expected_vertices() {
        // 5 vertices forming 4 triangles in distinct xy regions at distinct z planes:
        // v0=(0,0,1), v1=(2,0,1), v2=(1,2,1), v3=(3,2,2), v4=(1,4,3)
        let verts = [
            [0.0f32, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 1.0],
            [3.0, 2.0, 2.0],
            [1.0, 4.0, 3.0],
        ];
        let mut data = vec![0u8; 64];
        for (k, v) in verts.iter().enumerate() {
            for (c, &f) in v.iter().enumerate() {
                data[k * 12 + c * 4..k * 12 + c * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        // id dword at +60: I=1, J=2 (9) for all 4 triangle types
        data[60..64].copy_from_slice(&0x0909_0909u32.to_le_bytes());
        let d = [0.0, 0.0, 1.0];
        let inv = [f32::INFINITY, f32::INFINITY, 1.0];
        // Type 2 = (v2, v3, v4): centroid at ((1+3+1)/3, (2+2+4)/3) = (5/3, 8/3), z in 1..3
        let hit2 = intersect_node(&data, 2, 100.0, [5.0 / 3.0, 8.0 / 3.0, 0.0], d, inv, false).map(f32::from_bits);
        assert!(hit2[1] != 1.0 || hit2[0].is_finite(), "type 2 must hit (v2,v3,v4)");
        assert!(hit2[0].is_finite() && (hit2[0] / hit2[1] - 2.0).abs() < 1e-4);
        // Type 2 must miss a ray aimed at type 0's interior (1.0, 0.5)
        let miss2 = intersect_node(&data, 2, 100.0, [1.0, 0.5, 0.0], d, inv, false).map(f32::from_bits);
        assert!(miss2[0].is_infinite());
        // Type 3 = (v2, v4, v0): centroid at ((1+1+0)/3, (2+4+0)/3) = (2/3, 2.0)
        let hit3 = intersect_node(&data, 3, 100.0, [2.0 / 3.0, 2.0, 0.0], d, inv, false).map(f32::from_bits);
        assert!(hit3[0].is_finite() && (hit3[0] / hit3[1] - 5.0 / 3.0).abs() < 1e-4);
        // Type 3 must miss a ray aimed at type 0's interior (1.0, 0.5)
        let miss3 = intersect_node(&data, 3, 100.0, [1.0, 0.5, 0.0], d, inv, false).map(f32::from_bits);
        assert!(miss3[0].is_infinite());
    }
}
