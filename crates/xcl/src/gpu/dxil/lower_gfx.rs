//! Vertex and pixel shader entries for the direct translator (the recompiler's
//! `Mode::Vertex` / `Mode::Pixel`, `recomp/emit.rs` `finish` / `export_graphics`
//! / `interp`): signatures declared exactly as the HLSL entry structs declare
//! them (so they pack and link alike, also against an HLSL-path partner stage),
//! initial VGPRs from the inputs, exports into output temporaries stored once
//! at the end (unwritten components are 0, as `(XclOut)0`), `v_interp_*` as
//! the PC rasteriser's interpolated inputs, and the PS `discard` of lanes
//! without a DONE export.

use std::collections::{BTreeMap, HashMap};

use super::graphics::{CompType as GCt, GraphicsIo, Interp, SemKind, Sig, SigElement as GSig};
use super::lower::{err, LResult, Lw};
use super::module::{Pred, Value};
use crate::gpu::dxbc::SigElement;
use crate::gpu::rdna2::{self, Decoded};
use crate::gpu::recomp::{elements_at, ps_attribute_slot, ps_input_vgpr, ps_slot_attribute, ranked, vs_param_register, PsInterface, VsInterface};

/// An output element: its signature handle, components and the temporaries holding them.
pub struct GfxOut {
    pub sig: Sig,
    pub comps: u8,
    pub float: bool,
    pub vars: Vec<u32>,
}

pub struct GfxCtx {
    pub io: GraphicsIo,
    pub vs: Option<VsInterface>,
    pub ps: Option<PsInterface>,
    /// VS: SV_Position, then the params `p<k>` (`ranked(outputs, true)`)
    pub pos: Option<GfxOut>,
    pub params: Vec<GfxOut>,
    /// PS: the inputs `a<k>` (`ranked(inputs, false)`), SV_Target<i> by index, depth
    pub ps_in: Vec<(Sig, bool)>,
    pub targets: BTreeMap<u32, (GfxOut, u32)>,
    pub depth: Option<GfxOut>,
    /// PS: a DONE export happened somewhere (`xcl_live` decides the discard)
    pub ps_done: bool,
    pub live_var: u32,
}

fn gct(component_type: u32) -> GCt {
    match component_type {
        1 => GCt::U32,
        2 => GCt::I32,
        _ => GCt::F32,
    }
}

/// Components of an output element: clip / cull distances by their mask, the rest 4.
fn out_comps(e: &SigElement) -> u8 {
    if matches!(e.system_value, 2 | 3) {
        8 - e.mask.leading_zeros() as u8
    } else {
        4
    }
}

impl Lw {
    fn gfx(&mut self) -> &mut GfxCtx {
        self.gfx.as_mut().expect("graphics entry")
    }

    fn out_vars(&mut self, n: u8) -> Vec<u32> {
        let z = self.c(0);
        (0..n)
            .map(|_| {
                let t = self.tmp_var();
                self.wr(t, z);
                t
            })
            .collect()
    }

    /// Lane setup of the graphics stages (the HLSL VS / PS entry's): a wave64 program
    /// uses the hardware lane; a wave32 program in 64-lane hardware runs each half as
    /// its own wave (`xcl_w32`, `xcl_half`).
    pub fn setup_lanes_gfx(&mut self) {
        let hw = self.s.wave_lane_index();
        if self.wave64 {
            self.lane = hw;
        } else {
            let n = self.s.wave_lane_count();
            let w32 = self.cmpc(Pred::UGt, n, 32);
            let h = self.lshrc(hw, 5);
            let z = self.c(0);
            let half = self.sel(w32, h, z);
            let hnz = self.nz(half);
            self.gfx_half = Some(hnz);
            self.lane = self.andc(hw, 31);
        }
        let lane = self.lane;
        self.lane31 = self.andc(lane, 31);
        self.upper = self.cmpc(Pred::UGe, lane, 32);
        self.half = self.lshrc(lane, 5);
    }

    /// VS entry: signatures, initial registers.
    pub fn setup_vertex(&mut self, vs: &VsInterface, used_vgprs: &std::collections::BTreeSet<u64>) {
        let mut io = GraphicsIo::new(super::container::Stage::Vertex);
        let inputs: Vec<SigElement> = vs.inputs.iter().filter(|e| e.system_value == 0 && !e.name.starts_with("SV_")).cloned().collect();
        let mut attr: HashMap<u32, Sig> = HashMap::new();
        for e in &inputs {
            let s = io.input(GSig::user(&e.name, e.index, gct(e.component_type), 4));
            attr.insert(e.register, s);
        }
        let vid = io.input(GSig::new("SV_VertexID", 0, SemKind::VertexId, GCt::U32, 1));
        let iid = io.input(GSig::new("SV_InstanceID", 0, SemKind::InstanceId, GCt::U32, 1));
        let pos_sig = io.output(GSig::new("SV_Position", 0, SemKind::Position, GCt::F32, 4));
        let mut outs = Vec::new();
        for e in ranked(&vs.outputs, true) {
            let n = out_comps(e);
            let kind = match e.system_value {
                2 => SemKind::ClipDistance,
                3 => SemKind::CullDistance,
                _ => SemKind::Arbitrary,
            };
            let s = io.output(GSig::new(&e.name, e.index, kind, gct(e.component_type), n));
            outs.push((s, n, e.component_type == 3));
        }
        // initial VGPRs: v0 = vertex id, v3 = instance id, the fetch shader's attribute VGPRs
        let mut vgpr = HashMap::new();
        let vidv = io.load_input(&mut self.s, vid, 0, 0);
        let iidv = io.load_input(&mut self.s, iid, 0, 0);
        for &v in used_vgprs {
            let sv_at = vs.inputs.iter().find(|e| matches!(e.system_value, 6 | 8) && vs.input_vgpr(e.register) == Some(v));
            let val = if v == 0 {
                Some(vidv)
            } else if v == 3 {
                Some(iidv)
            } else if let Some(e) = sv_at {
                Some(if e.system_value == 6 { vidv } else { iidv })
            } else {
                inputs.iter().find_map(|e| {
                    vs.input_vgpr(e.register).filter(|&b| v >= b && v < b + 4).map(|b| (attr[&e.register], (v - b) as u8))
                }).map(|(s, c)| {
                    let x = io.load_input(&mut self.s, s, 0, c);
                    let i32 = self.s.m.i32();
                    self.s.m.bitcast(x, i32)
                })
            };
            if let Some(x) = val {
                vgpr.insert(v as u32, x);
            }
        }
        self.gfx_common_init(vgpr);
        let pos = GfxOut { sig: pos_sig, comps: 4, float: true, vars: self.out_vars(4) };
        let params = outs.into_iter().map(|(sig, n, float)| GfxOut { sig, comps: n, float, vars: self.out_vars(n) }).collect();
        self.gfx = Some(GfxCtx { io, vs: Some(vs.clone()), ps: None, pos: Some(pos), params, ps_in: Vec::new(), targets: BTreeMap::new(), depth: None, ps_done: false, live_var: 0 });
    }

    /// PS entry: signatures, initial registers (`modes`: the interpolation of each
    /// attribute from the I/J pair its `v_interp` uses, `emit::ps_attr_modes`).
    pub fn setup_pixel(&mut self, ps: &PsInterface, modes: &BTreeMap<u32, String>) {
        let mut io = GraphicsIo::new(super::container::Stage::Pixel);
        let conservative = ps.outputs.iter().any(|e| matches!(e.system_value, 67 | 68));
        let mut pos_e = GSig::new("SV_Position", 0, SemKind::Position, GCt::F32, 4);
        if conservative {
            pos_e = pos_e.interp(Interp::LinearNoperspectiveCentroid);
        }
        let pos = io.input(pos_e);
        let mut ps_in = Vec::new();
        for (k, e) in ranked(&ps.inputs, false).iter().enumerate() {
            let attr = ps_slot_attribute(ps, k);
            let m = attr.and_then(|a| modes.get(&(a as u32))).cloned().unwrap_or_default();
            let float = e.component_type == 3;
            let interp = if !float {
                Interp::Constant
            } else {
                match m.as_str() {
                    "sample" => Interp::LinearSample,
                    "centroid" => Interp::LinearCentroid,
                    "noperspective" => Interp::LinearNoperspective,
                    "noperspective centroid" => Interp::LinearNoperspectiveCentroid,
                    "noperspective sample" => Interp::LinearNoperspectiveSample,
                    "nointerpolation" => Interp::Constant,
                    _ => Interp::Linear,
                }
            };
            let s = io.input(GSig::user(&e.name, e.index, gct(e.component_type), 4).interp(interp));
            ps_in.push((s, float));
        }
        // DXIL wants SV_IsFrontFace as uint (`bool` in HLSL becomes uint 0 / 0xFFFFFFFF)
        let ff = (ps.input_ena >> 12 & 1 == 1).then(|| io.input(GSig::new("SV_IsFrontFace", 0, SemKind::IsFrontFace, GCt::U32, 1).interp(Interp::Constant)));
        let mut targets = BTreeMap::new();
        let mut depth = None;
        let mut any = false;
        for e in &ps.outputs {
            match e.system_value {
                64 => {
                    let s = io.output(GSig::new("SV_Target", e.index, SemKind::Target, gct(e.component_type), 4));
                    targets.insert(e.index, (GfxOut { sig: s, comps: 4, float: e.component_type == 3, vars: Vec::new() }, e.component_type));
                    any = true;
                }
                65 | 67 | 68 => {
                    let kind = match e.system_value {
                        65 => SemKind::Depth,
                        67 => SemKind::DepthGreaterEqual,
                        _ => SemKind::DepthLessEqual,
                    };
                    let s = io.output(GSig::new(match kind {
                        SemKind::Depth => "SV_Depth",
                        SemKind::DepthGreaterEqual => "SV_DepthGreaterEqual",
                        _ => "SV_DepthLessEqual",
                    }, 0, kind, GCt::F32, 1));
                    depth = Some(GfxOut { sig: s, comps: 1, float: true, vars: Vec::new() });
                    any = true;
                }
                _ => {}
            }
        }
        if !any {
            let s = io.output(GSig::new("SV_Target", 7, SemKind::Target, GCt::F32, 4));
            targets.insert(7, (GfxOut { sig: s, comps: 4, float: true, vars: Vec::new() }, 3));
        }
        // initial VGPRs: the system values SPI_PS_INPUT_ENA enables
        let mut vgpr = HashMap::new();
        let sys = |bit: u32| ps_input_vgpr(ps.input_ena, bit);
        for (bit, c) in [(8u32, 0u8), (9, 1), (10, 2), (11, 3)] {
            if let Some(v) = sys(bit) {
                let x = io.load_input(&mut self.s, pos, 0, c);
                let i32 = self.s.m.i32();
                vgpr.insert(v as u32, self.s.m.bitcast(x, i32));
            }
        }
        if let (Some(v), Some(ff)) = (sys(12), ff) {
            let x = io.load_input(&mut self.s, ff, 0, 0);
            let b = self.nz(x);
            let one = self.c(0x3F80_0000);
            let z = self.c(0);
            vgpr.insert(v as u32, self.sel(b, one, z));
        }
        if let Some(v) = sys(14) {
            vgpr.insert(v as u32, io.coverage(&mut self.s));
        }
        self.gfx_common_init(vgpr);
        for (o, _) in targets.values_mut() {
            o.vars = self.out_vars(4);
        }
        if let Some(d) = depth.as_mut() {
            d.vars = self.out_vars(1);
        }
        let live_var = self.tmp_var();
        let f = self.c(0);
        self.wr(live_var, f);
        self.gfx = Some(GfxCtx { io, vs: None, ps: Some(ps.clone()), pos: None, params: Vec::new(), ps_in, targets, depth, ps_done: false, live_var });
    }

    /// SGPRs from `io`, SCC 0, EXEC = `ballot(true)`.
    fn gfx_common_init(&mut self, vgpr: HashMap<u32, Value>) {
        let zero = self.c(0);
        let mut sgpr = HashMap::new();
        sgpr.insert(super::lower::VAR_SCC, zero);
        let t = self.s.m.i1c(true);
        let (lo, hi) = self.ballot(t);
        sgpr.insert(126, lo);
        sgpr.insert(127, hi);
        self.init.compute = Some(super::lower::ComputeInit { sgpr, vgpr, zero });
    }

    /// The end of a graphics stage: the PS discard, then every output component.
    pub fn finish_gfx(&mut self) {
        let mut g = self.gfx.take().expect("graphics entry");
        if g.ps_done {
            let live = self.rd(g.live_var);
            let dead = self.cmpc(Pred::Eq, live, 0);
            g.io.discard(&mut self.s, dead);
        }
        let mut all: Vec<&GfxOut> = Vec::new();
        all.extend(g.pos.iter());
        all.extend(g.params.iter());
        all.extend(g.targets.values().map(|(o, _)| o));
        all.extend(g.depth.iter());
        let stores: Vec<(Sig, u8, bool, u32)> = all.iter().flat_map(|o| o.vars.iter().enumerate().map(move |(c, &v)| (o.sig, c as u8, o.float, v))).collect();
        for (sig, c, float, var) in stores {
            let v = self.rd(var);
            let v = if float { self.f(v) } else { v };
            g.io.store_output(&mut self.s, sig, 0, c, v);
        }
        self.gfx = Some(g);
    }

    /// `out[c] = v` in active lanes.
    fn set_out(&mut self, var: u32, v: Value) {
        let a = self.act();
        let old = self.rd(var);
        let n = self.sel(a, v, old);
        self.wr(var, n);
    }

    /// Graphics exports (`export_graphics`).
    pub fn export_gfx(&mut self, d: &Decoded, en: u64, tgt: u64, compr: bool, srcs: &[Value]) -> LResult<()> {
        let done = Self::fv(d, "DONE") == 1;
        let enabled = |c: usize| if compr { en >> (c / 2 * 2) & 3 != 0 } else { en >> c & 1 == 1 };
        let comp = |lw: &mut Lw, c: usize| -> Value {
            if compr {
                let r = srcs[c / 2];
                if c % 2 == 0 {
                    lw.andc(r, 0xFFFF)
                } else {
                    lw.lshrc(r, 16)
                }
            } else {
                srcs[c]
            }
        };
        let pixel = self.gfx().ps.is_some();
        if !pixel {
            match tgt {
                12 => {
                    let vars = self.gfx().pos.as_ref().expect("pos").vars.clone();
                    for (c, &var) in vars.iter().enumerate() {
                        if enabled(c) {
                            let v = comp(self, c);
                            self.set_out(var, v);
                        }
                    }
                }
                13..=15 => {}
                32..=63 => {
                    let vs = self.gfx().vs.clone().expect("vs");
                    let k = (tgt - 32) as usize;
                    let at = vs_param_register(&vs, k).map(|r| elements_at(&vs.outputs, true, r)).unwrap_or_default();
                    for j in at {
                        let (vars, n) = {
                            let o = &self.gfx().params[j];
                            (o.vars.clone(), o.comps)
                        };
                        for c in 0..n as usize {
                            if enabled(c) {
                                let v = comp(self, c);
                                self.set_out(vars[c], v);
                            }
                        }
                    }
                }
                _ => return err(d, "export target"),
            }
            return Ok(());
        }
        match tgt {
            0..=7 => {
                let t = self.gfx().targets.get(&(tgt as u32)).map(|(o, ct)| (o.vars.clone(), *ct));
                if let Some((vars, ct)) = t {
                    for (c, &var) in vars.iter().enumerate() {
                        if !enabled(c) {
                            continue;
                        }
                        let raw = comp(self, c);
                        let v = match (ct, compr) {
                            (3, true) => {
                                let f = self.s.f16_to_f32(raw);
                                self.bits(f)
                            }
                            (2, true) => self.sx16(raw),
                            _ => raw,
                        };
                        self.set_out(var, v);
                    }
                }
            }
            8 => {
                let dv = self.gfx().depth.as_ref().map(|o| o.vars[0]);
                if let (true, Some(var)) = (en & 1 == 1, dv) {
                    self.set_out(var, srcs[0]);
                }
            }
            9 => {}
            _ => return err(d, "export target"),
        }
        if done {
            self.gfx().ps_done = true;
            let lv = self.gfx().live_var;
            let one = self.c(1);
            self.set_out(lv, one);
        }
        Ok(())
    }

    /// `v_interp_*`: the attribute comes interpolated from the PC rasteriser.
    pub fn interp_gfx(&mut self, d: &Decoded, name: &str) -> LResult<()> {
        let Some(ps) = self.gfx.as_ref().and_then(|g| g.ps.clone()) else { return err(d, "interpolation outside PS") };
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
        let (vdst, ij, attr, chan) = if d.enc.name == "ENC_VINTRP" {
            (f("VDST"), f("VSRC"), f("ATTR"), f("ATTRCHAN"))
        } else {
            let s1 = f("SRC1");
            (f("VDST"), if s1 >= 256 { s1 - 256 } else { s1 }, f("SRC0") & 0x3F, f("SRC0") >> 6 & 3)
        };
        let Some(slot) = ps_attribute_slot(&ps, attr as usize, chan as usize) else { return err(d, "attribute outside ISG1") };
        let load = |lw: &mut Lw| -> Value {
            let g = lw.gfx.as_mut().expect("gfx");
            let (sig, _) = g.ps_in[slot];
            let x = g.io.load_input(&mut lw.s, sig, 0, chan as u8);
            let i32 = lw.s.m.i32();
            lw.s.m.bitcast(x, i32)
        };
        let dst = super::lower::Dst::V(vdst);
        match name {
            "v_interp_p1_f32" | "v_interp_p1ll_f16" | "v_interp_p1lv_f16" => {}
            "v_interp_p2_f32" => {
                let v = load(self);
                self.write32(&dst, v);
            }
            "v_interp_mov_f32" => {
                if ij != 2 {
                    return err(d, "v_interp_mov p10/p20");
                }
                let v = load(self);
                self.write32(&dst, v);
            }
            "v_interp_p2_f16" => {
                let x = load(self);
                let xf = self.f(x);
                let h = self.to_f16(xf);
                let cur = self.vreg(vdst);
                let hi = rdna2::field_value(d.enc, &d.words, "OP_SEL").unwrap_or(0) >> 3 & 1 == 1;
                let v = if hi {
                    let l = self.andc(cur, 0xFFFF);
                    let hs = self.shlc(h, 16);
                    self.or(l, hs)
                } else {
                    let hh = self.andc(cur, 0xFFFF_0000);
                    self.or(hh, h)
                };
                self.write32(&dst, v);
            }
            _ => return err(d, "interpolation"),
        }
        Ok(())
    }
}
