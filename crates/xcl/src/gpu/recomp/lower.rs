//! Lowering of console programs to DXIL-shaped IR, for the direct backend (`XCL_RECOMP_DXIL`).
//!
//! The console CFG is kept: blocks become blocks, and every register dword is an SSA value
//! (`ssa`). EXEC is not a number here but each lane's own bit (`i1`): a mask branch
//! (`s_cbranch_execz`, `vccz`, SCC of a mask op) is a branch of each lane on its bit, and a
//! vector instruction keeps a guard (`select(bit, new, old)`) only where `lane_masks` could not
//! prove the bit set. An SGPR therefore has two views: its number (`Var::S`) and, when it holds
//! a mask, the lane's bit of it (`Var::M`). Writing one view leaves a marker in the other – `undef`
//! – so a number read where only a mask was written (or the reverse, across a join) is found once
//! every phi is complete, and the program is refused (it stays on the HLSL path). A bit read where
//! only a number was written is the lane's bit of that number. At the end of the program every
//! lane is back, so a mask stored as a number there is the wave's ballot of the bits.
//!
//! Kernel mode only for now (the ABI of `runner`): registers come from `io` and go back to `res`.
//! The IR is written through `Ir`, the part of the DXIL module builder this needs.

use std::collections::HashSet;

use super::lane_masks::{self, Edge, Summary};
use super::ssa::{Sink, Ssa, Var};
use super::{KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS};
use crate::gpu::rdna2::{self, Decoded, Variant};

mod wave;

type R<T> = Result<T, String>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cmp {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

/// What the lowering needs from the DXIL module builder. Blocks are numbered in the order
/// `new_block` creates them, the first being the entry; `And` / `Or` / `Xor` take `i1` or `i32`.
pub trait Ir: Sink {
    fn new_block(&mut self) -> usize;
    fn at(&mut self, block: usize);
    fn int(&mut self, v: u32) -> Self::V;
    fn boolean(&mut self, v: bool) -> Self::V;
    fn bin(&mut self, op: Bin, a: Self::V, b: Self::V) -> Self::V;
    fn icmp(&mut self, c: Cmp, a: Self::V, b: Self::V) -> Self::V;
    fn select(&mut self, c: Self::V, a: Self::V, b: Self::V) -> Self::V;
    /// `i1` → `i32` (0 / 1)
    fn zext(&mut self, b: Self::V) -> Self::V;
    /// the console lane of this thread
    fn lane(&mut self) -> Self::V;
    /// compute: the group's id, component k (x, y, z)
    fn group_id(&mut self, k: u32) -> Self::V;
    /// compute, a group of `threads`: this thread's console thread (its flat index in the group)
    /// and that thread's id (x, y, z). The console forms a wave from consecutive threads, lane l
    /// the l-th of them, so the flat index is the wave's first thread plus `lane()`; a PC driver
    /// need not, in 2-D / 3-D groups, so the threads are numbered as the waves arrive
    fn compute_ids(&mut self, threads: [u32; 3]) -> (Self::V, [Self::V; 3]);
    /// the wave's ballot of `c`: lanes 0–31, 32–63
    fn ballot(&mut self, c: Self::V) -> [Self::V; 2];
    /// `ballot` where only the word of this lane's 32-lane half is used (the other may be
    /// anything): a wave64 split into two 32-lane hardware waves needs no exchange for it
    fn ballot_half(&mut self, c: Self::V) -> [Self::V; 2];
    /// `v` as lane `lane` holds it (`WaveReadLaneAt`); `lane` may differ between lanes and is
    /// always one running this instruction too
    fn read_lane(&mut self, v: Self::V, lane: Self::V) -> Self::V;
    /// `read_lane` of a lane in this lane's 32-lane half (DPP, permlane, swizzle)
    fn read_lane_half(&mut self, v: Self::V, lane: Self::V) -> Self::V;
    /// Whether a whole-wave `ballot` / `read_lane` meets another hardware wave (the split form
    /// of a wave64 program): then it needs every lane of the console wave where it runs.
    fn whole_needs_exchange(&self) -> bool {
        false
    }
    /// Whether this IR can lower whole-wave branches (`ballot` across the full console wave,
    /// including a split wave64 whose partner half is present or spin-synced).
    fn can_wave_branch(&self) -> bool {
        !self.whole_needs_exchange()
    }
    /// the number of bits set in `v` (`countbits`)
    fn count_bits(&mut self, v: Self::V) -> Self::V;
    /// `d` lowered with another lowering's instruction semantics (#1426's `Lw::inst_guarded`):
    /// `inputs` are the registers it reads, `guard` (i1) decides this lane's register writes and
    /// memory effects, and the values of `outputs` after it come back in order. Err when there
    /// is no such lowering or it refuses `d`.
    fn foreign(&mut self, _d: &Decoded, _guard: Self::V, _inputs: &[(Var, Self::V)], _outputs: &[Var]) -> R<Vec<Self::V>> {
        Err("no foreign lowering".into())
    }
    fn io_load(&mut self, dword: Self::V) -> Self::V;
    fn res_store(&mut self, dword: Self::V, value: Self::V);
    /// the dword of the flat memory (`mem`, u2) at a byte address (a multiple of 4); out of
    /// range reads 0 and writes nothing, as D3D12 raw buffers do
    fn mem_load(&mut self, addr: Self::V) -> Self::V;
    fn mem_store(&mut self, addr: Self::V, value: Self::V);
    /// the 4 GiB memory windows the program addresses: with more than one, `mem_load` /
    /// `mem_store` (window 0 only) are not used, and memory instructions go to `foreign`
    fn mem_windows(&self) -> u32 {
        1
    }
    fn br(&mut self, target: usize);
    fn br_if(&mut self, c: Self::V, then: usize, otherwise: usize);
    fn ret(&mut self);
    /// VS / PS: the stage's start (in the DXIL adapter #1426's `setup_vertex` / `setup_pixel`),
    /// with `vregs` the VGPRs the program names. None: this IR has no graphics entry, and the
    /// program is refused.
    fn graphics_start(&mut self, vregs: &[u16]) -> Option<GraphicsStart<Self::V>> {
        let _ = vregs;
        None
    }
    /// … and its end (`finish_gfx`: the outputs, the PS discard), where every lane is back
    fn graphics_end(&mut self) {}
}

/// What a VS / PS starts with (`Ir::graphics_start`).
pub struct GraphicsStart<V> {
    /// the registers the stage sets beyond the user SGPRs from `io`: (SGPR, value), (VGPR,
    /// value); a VGPR it does not set starts as 0
    pub sgprs: Vec<(u8, V)>,
    pub vgprs: Vec<(u16, V)>,
    /// the lane's bit of the launch EXEC: false in a PS's helper lanes, which run the program too
    pub exec: V,
}

const EXEC: u8 = 126;
const VCC: u8 = 106;

/// The program's stage: where its registers come from and go.
#[derive(Clone, Copy)]
pub enum Stage<'a> {
    /// `runner`'s ABI: the whole register file from `io`, back to `res`
    Kernel,
    /// a compute program: user SGPRs from `io`, then the group ids and TG_SIZE it enables,
    /// thread ids in v0–v2; it ends without storing registers
    Compute(&'a super::CsInterface),
    /// a vertex / pixel shader: user SGPRs from `io`, the rest from `Ir::graphics_start`; it ends
    /// in `Ir::graphics_end`
    Vertex,
    Pixel,
}

struct Lower<'a, I: Ir> {
    ir: I,
    ssa: Ssa<I::V>,
    sum: &'a Summary,
    guarded: HashSet<u64>,
    wave_branches: bool,
    /// Whether SCC is live after instruction `i` (used to skip dead whole-wave SCC ballots).
    scc_live_after: Vec<bool>,
    cur_inst: usize,
    wave64: bool,
    /// the markers: a number where a mask was written, a bit where a number was
    num_marker: I::V,
    bit_marker: I::V,
    /// values read as numbers / as mask bits, checked once every phi is complete
    num_uses: Vec<I::V>,
    bit_uses: Vec<I::V>,
    lane: Option<I::V>,
    /// per IR block: not every lane that reached one of the program's lane branches gets here
    /// before they meet again (a loop that lanes leave one by one, an arm some lanes skip)
    partial: Vec<bool>,
    track_frexp_denorm: bool,
    /// stale-value ownership for blocks split by a coalesced run: backbone blocks of console
    /// block b (`blk(b)` and its continuations) share `blk(b)` for both defs and uses, while a
    /// run's inner block keeps its own `def_owner` (any def there is partial to outside reads)
    /// and `blk(b)` as its `use_owner` (reads of earlier defs in the same console block stay valid)
    def_owner: Vec<usize>,
    use_owner: Vec<usize>,
    /// inside a coalesced guarded run's own block: only lanes with EXEC set enter, so writes
    /// need no per-instruction guard
    unguarded: bool,
    /// the IR block each mask bit was written in, and the mask bits read (block, value)
    def_block: std::collections::HashMap<I::V, usize>,
    lane_uses: Vec<(usize, I::V)>,
    /// a DPP instruction's first source, read from its source lane, and its write enable
    dpp_src: Option<I::V>,
    dpp_en: Option<I::V>,
    /// survey mode (`survey_compute`): every refusal is noted here and the lowering goes on
    noted: Option<Vec<String>>,
    /// a VS / PS (`lower_graphics`): `s_wqm` only of the launch EXEC (its quads' helper lanes)
    graphics: bool,
}

/// A vector memory store: a side effect inactive lanes must not have.
fn is_store(d: &Decoded) -> bool {
    let n = d.inst.name;
    (n.starts_with("GLOBAL_STORE_") || n.starts_with("FLAT_STORE_")) && matches!(d.enc.name, "ENC_FLAT_GLBL" | "ENC_FLAT")
}

/// A PS instruction that needs all four lanes of each quad: an image sample or gather with an
/// implicit LOD (no `_L` / `_LZ`, no gradients `_D` / `_CD`), `image_get_lod`, `s_wqm`. The
/// console has them from whole-quad mode; DXIL leaves derivatives undefined where control flow is
/// not uniform over the quad.
fn whole_quad(name: &str) -> bool {
    if name == "IMAGE_GET_LOD" || name.starts_with("S_WQM_") {
        return true;
    }
    match name.strip_prefix("IMAGE_SAMPLE").or_else(|| name.strip_prefix("IMAGE_GATHER4")) {
        Some(rest) => !rest.split('_').any(|t| matches!(t, "L" | "LZ" | "D" | "CD")),
        None => false,
    }
}

/// Cross-lane and wave-level ops stay out of coalesced guarded runs (they need the parent
/// block's lane participation, e.g. fetch-inactive or whole-wave mask reads).
fn is_cross_lane(d: &Decoded) -> bool {
    let n = d.inst.name;
    matches!(d.variant, Variant::Dpp16 | Variant::Dpp8)
        || n.starts_with("V_PERMLANE")
        || n.starts_with("V_MBCNT_")
        || n.starts_with("V_READLANE")
        || n.starts_with("V_READFIRSTLANE")
        || n.starts_with("V_WRITELANE")
        || n.starts_with("DS_SWIZZLE")
        || n.starts_with("DS_BPERMUTE")
        || n.starts_with("DS_PERMUTE")
}

/// Any write to an SGPR, mask (VCC / EXEC / M0), or SCC must run for all lanes outside a
/// coalesced run (inactive lanes clear their bit in a VALU mask write rather than keeping it).
fn writes_scalar_or_scc(d: &Decoded) -> bool {
    let (_, written) = super::x64_masks::uses(d);
    let (_, explicit_w) = rdna2::sgpr_uses(d);
    let writes = explicit_w.iter().fold(written, |m, &r| m | 1u128 << r) & !(1u128 << 125);
    writes != 0
        || d.inst.name.starts_with("V_CMPX")
        || d.ienc
            .operands
            .iter()
            .any(|o| o.output && matches!(o.ty, "OPR_VCC" | "OPR_EXEC" | "OPR_SDST_EXEC" | "OPR_SDST_M0" | "OPR_SSRC_SPECIAL_SCC"))
}

/// Pure no-ops in this lowering: allowed inside a coalesced run without breaking it.
fn is_run_noop(d: &Decoded) -> bool {
    matches!(
        d.inst.name,
        "V_NOP" | "S_NOP" | "S_WAITCNT" | "S_WAITCNT_VSCNT" | "S_WAITCNT_VMCNT" | "S_WAITCNT_LGKMCNT" | "S_WAITCNT_EXPCNT" | "S_INST_PREFETCH" | "S_CLAUSE"
    )
}

/// Guarded instructions with side effects that inactive lanes must not execute even alone.
fn must_branch(d: &Decoded) -> bool {
    let n = d.inst.name;
    is_store(d)
        || n.contains("_STORE_")
        || n.starts_with("IMAGE_STORE")
        || n.contains("_ATOMIC_")
        || (n.starts_with("DS_") && !n.starts_with("DS_READ") && !n.starts_with("DS_NOP"))
        || n == "EXP"
}

fn can_coalesce(d: &Decoded, guarded: &HashSet<u64>) -> bool {
    guarded.contains(&d.pc) && !is_cross_lane(d) && !whole_quad(d.inst.name) && !writes_scalar_or_scc(d)
}

/// Exclusive end of a block's non-terminator instructions.
fn block_body_end(insts: &[Decoded], range: &std::ops::Range<usize>) -> usize {
    let mut end = range.end;
    if end > range.start {
        let d = &insts[end - 1];
        let name = d.inst.name;
        if name.starts_with("S_CBRANCH") || name == "S_BRANCH" || name.starts_with("S_ENDPGM") || d.far_target.is_some() {
            end -= 1;
        }
    }
    end
}

/// Straight-line runs of guarded instructions in one console block that benefit from a single
/// per-lane EXEC branch instead of per-write `select`s (or per-store branches): any run with
/// a `must_branch` instruction or at least two guarded instructions.
fn coalesced_runs(insts: &[Decoded], range: &std::ops::Range<usize>, guarded: &HashSet<u64>) -> Vec<std::ops::Range<usize>> {
    let end = block_body_end(insts, range);
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    let mut last_end = 0usize;
    let mut inst_count = 0usize;
    let mut branch = false;
    let mut flush = |start: &mut Option<usize>, last_end: usize, inst_count: &mut usize, branch: &mut bool| {
        if let Some(s) = start.take() {
            if *branch || *inst_count >= 2 {
                runs.push(s..last_end);
            }
            *inst_count = 0;
            *branch = false;
        }
    };
    for i in range.start..end {
        let d = &insts[i];
        if is_run_noop(d) {
            continue;
        }
        if can_coalesce(d, guarded) {
            if start.is_none() {
                start = Some(i);
            }
            last_end = i + 1;
            inst_count += 1;
            branch |= must_branch(d);
        } else {
            flush(&mut start, last_end, &mut inst_count, &mut branch);
        }
    }
    flush(&mut start, last_end, &mut inst_count, &mut branch);
    runs
}

/// IR block of analysis block b (block 0 is the prologue).
fn blk(b: usize) -> usize {
    b + 1
}

impl<'a, I: Ir> Lower<'a, I> {
    fn lane(&mut self) -> I::V {
        *self.lane.get_or_insert_with(|| self.ir.lane())
    }

    fn read(&mut self, b: usize, var: Var) -> I::V {
        self.ssa.read(&mut self.ir, b, var)
    }

    fn read_num(&mut self, b: usize, r: u8) -> R<I::V> {
        let v = self.read(b, Var::S(r));
        if self.ssa.resolve_value(v) == self.num_marker {
            return Err(format!("s{r}: a mask read as a number"));
        }
        self.num_uses.push(v);
        self.lane_uses.push((b, v));
        Ok(v)
    }

    /// The lane's bit of the mask in s<r> (wave64: the pair from s<r>).
    fn read_bit(&mut self, b: usize, r: u8) -> R<I::V> {
        let m = self.read(b, Var::M(r));
        if self.ssa.resolve_value(m) != self.bit_marker {
            self.bit_uses.push(m);
            if r != EXEC {
                self.lane_uses.push((b, m));
            }
            return Ok(m);
        }
        // only numbers written: the lane's bit of them
        let lane = self.lane();
        let word = if self.wave64 {
            let (lo, hi) = (self.read_num(b, r)?, self.read_num(b, r + 1)?);
            let c32 = self.ir.int(32);
            let upper = self.ir.icmp(Cmp::Uge, lane, c32);
            self.ir.select(upper, hi, lo)
        } else {
            self.read_num(b, r)?
        };
        let c31 = self.ir.int(31);
        let sh = self.ir.bin(Bin::And, lane, c31);
        let shifted = self.ir.bin(Bin::LShr, word, sh);
        let one = self.ir.int(1);
        let bit = self.ir.bin(Bin::And, shifted, one);
        let zero = self.ir.int(0);
        Ok(self.ir.icmp(Cmp::Ne, bit, zero))
    }

    /// SCC, written and read like an SGPR number: a lane that skipped the block writing it
    /// keeps its own older value (`stale`).
    /// `v` as written in block b: in a block some lanes skip, a value made elsewhere (an older
    /// mask copied, a shared constant) gets a copy of its own, so `stale` sees the write here.
    fn own(&mut self, b: usize, v: I::V, bit: bool) -> I::V {
        let v = match self.def_block.get(&v) {
            Some(&d) if d != b && self.partial[b] => {
                let z = if bit { self.ir.boolean(false) } else { self.ir.int(0) };
                self.ir.bin(Bin::Or, v, z)
            }
            _ => v,
        };
        self.def_block.entry(v).or_insert(b);
        v
    }

    fn write_scc(&mut self, b: usize, v: I::V) {
        let v = self.own(b, v, true);
        self.ssa.write(b, Var::Scc, v);
    }

    fn read_scc(&mut self, b: usize) -> I::V {
        let v = self.read(b, Var::Scc);
        self.lane_uses.push((b, v));
        v
    }

    fn write_num(&mut self, b: usize, r: u8, v: I::V) {
        let v = self.own(b, v, false);
        self.ssa.write(b, Var::S(r), v);
        self.ssa.write(b, Var::M(r), self.bit_marker);
        if self.wave64 && r > 0 {
            // a mask in the pair from s<r - 1> is no longer that mask
            self.ssa.write(b, Var::M(r - 1), self.bit_marker);
        }
    }

    fn write_bit(&mut self, b: usize, r: u8, bit: I::V) {
        let bit = if r != EXEC { self.own(b, bit, true) } else { bit };
        self.ssa.write(b, Var::M(r), bit);
        self.ssa.write(b, Var::S(r), self.num_marker);
        if self.wave64 {
            self.ssa.write(b, Var::S(r + 1), self.num_marker);
            self.ssa.write(b, Var::M(r + 1), self.bit_marker);
            if r > 0 {
                self.ssa.write(b, Var::M(r - 1), self.bit_marker);
            }
        }
    }

    /// Whether s<r> holds a mask here (its bit view is not the marker).
    fn holds_mask(&mut self, b: usize, r: u8) -> bool {
        let m = self.read(b, Var::M(r));
        self.ssa.resolve_value(m) != self.bit_marker
    }

    /// A scalar or vector source operand as a number.
    fn src(&mut self, b: usize, d: &Decoded, code: u64) -> R<I::V> {
        if let (233 | 234 | 250, Some(v)) = (code, self.dpp_src) {
            return Ok(v);
        }
        Ok(match code {
            0..=105 | 106 | 107 | 124 | 126 | 127 => self.read_num(b, code as u8)?,
            125 => self.ir.int(0),
            128..=192 => self.ir.int((code - 128) as u32),
            193..=208 => self.ir.int((-((code - 192) as i32)) as u32),
            240..=247 => self.ir.int([0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][(code - 240) as usize].to_bits()),
            248 => self.ir.int(0x3E22_F983),
            253 => {
                let s = self.read_scc(b);
                self.ir.zext(s)
            }
            255 => self.ir.int(d.literal.ok_or("missing literal")?),
            256..=511 => self.read(b, Var::V((code - 256) as u16)),
            _ => return Err(format!("source code {code}")),
        })
    }

    /// A scalar source operand as a mask's lane bit: a register, NULL, inline integer, or literal.
    fn src_bit(&mut self, b: usize, d: &Decoded, code: u64) -> R<I::V> {
        let bit_of_const = |this: &mut Self, lo: u32, hi_all_ones: bool| {
            let lane = this.lane();
            let c31 = this.ir.int(31);
            let sh = this.ir.bin(Bin::And, lane, c31);
            let clo = this.ir.int(lo);
            let shifted = this.ir.bin(Bin::LShr, clo, sh);
            let one = this.ir.int(1);
            let masked = this.ir.bin(Bin::And, shifted, one);
            let z = this.ir.int(0);
            let lo_bit = this.ir.icmp(Cmp::Ne, masked, z);
            if this.wave64 {
                let c32 = this.ir.int(32);
                let upper = this.ir.icmp(Cmp::Uge, lane, c32);
                let hi_bit = this.ir.boolean(hi_all_ones);
                this.ir.select(upper, hi_bit, lo_bit)
            } else {
                lo_bit
            }
        };
        match code {
            0..=105 | 106 | 107 | 126 | 127 => self.read_bit(b, code as u8),
            125 | 128 => Ok(self.ir.boolean(false)),
            193 => Ok(self.ir.boolean(true)),
            129..=192 => Ok(bit_of_const(self, (code - 128) as u32, false)),
            194..=208 => Ok(bit_of_const(self, (-((code - 192) as i32)) as u32, true)),
            255 => Ok(bit_of_const(self, d.literal.ok_or("missing literal")?, false)),
            _ => Err(format!("mask source code {code}")),
        }
    }

    fn write_v(&mut self, b: usize, d: &Decoded, r: u16, v: I::V) -> R<()> {
        let v = match self.dpp_en.take() {
            Some(en) => {
                let old = self.read(b, Var::V(r));
                self.ir.select(en, v, old)
            }
            None => v,
        };
        let v = if !self.unguarded && self.guarded.contains(&d.pc) {
            let old = self.read(b, Var::V(r));
            let e = self.read_bit(b, EXEC)?;
            self.ir.select(e, v, old)
        } else {
            v
        };
        self.ssa.write(b, Var::V(r), v);
        Ok(())
    }

    fn not(&mut self, a: I::V) -> I::V {
        let t = self.ir.boolean(true);
        self.ir.bin(Bin::Xor, a, t)
    }

    /// A bitwise mask op of SALU / saveexec on lane bits.
    fn bit_op(&mut self, op: &str, a: I::V, e: I::V) -> R<I::V> {
        Ok(match op {
            "AND" => self.ir.bin(Bin::And, a, e),
            "OR" => self.ir.bin(Bin::Or, a, e),
            "XOR" => self.ir.bin(Bin::Xor, a, e),
            "ANDN2" => {
                let ne = self.not(e);
                self.ir.bin(Bin::And, a, ne)
            }
            "ORN2" => {
                let ne = self.not(e);
                self.ir.bin(Bin::Or, a, ne)
            }
            "ANDN1" => {
                let na = self.not(a);
                self.ir.bin(Bin::And, na, e)
            }
            "ORN1" => {
                let na = self.not(a);
                self.ir.bin(Bin::Or, na, e)
            }
            "NAND" | "NOR" | "XNOR" => {
                let inner = match op {
                    "NAND" => Bin::And,
                    "NOR" => Bin::Or,
                    _ => Bin::Xor,
                };
                let x = self.ir.bin(inner, a, e);
                self.not(x)
            }
            _ => return Err(format!("mask op {op}")),
        })
    }

    /// A mask-producing scalar operation sets SCC from its full result. In the native
    /// wave CFG every lane reaches scalar instructions, including lanes outside EXEC.
    fn mask_nonzero(&mut self, bit: I::V) -> I::V {
        if !self.wave_branches {
            return bit;
        }
        let [lo, hi] = self.ir.ballot(bit);
        let word = if self.wave64 { self.ir.bin(Bin::Or, lo, hi) } else { lo };
        let zero = self.ir.int(0);
        self.ir.icmp(Cmp::Ne, word, zero)
    }

    /// Whole-wave `mask_nonzero` for an SCC write: skips the ballot (and any split-wave64
    /// exchange barrier) when no subsequent instruction reads SCC before it is overwritten.
    pub(super) fn mask_nonzero_scc(&mut self, bit: I::V) -> I::V {
        if self.wave_branches && !self.scc_live_after[self.cur_inst] {
            return self.ir.boolean(false);
        }
        self.mask_nonzero(bit)
    }

    fn salu(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f);
        let (s0, s1, sd) = (fv("SSRC0").unwrap_or(0), fv("SSRC1").unwrap_or(0), fv("SDST").unwrap_or(0) as u8);
        if self.wave_salu(b, d)? {
            return Ok(());
        }
        if name == "S_GETPC_B64" {
            // the address after it, as the program's offset (#1426's `s_getpc_b64`): read only by
            // the far jump `resolve_far_jumps` resolved
            let (lo, hi) = (self.ir.int((d.pc + d.len as u64) as u32), self.ir.int(0));
            self.write_num(b, sd, lo);
            self.write_num(b, sd + 1, hi);
            return Ok(());
        }
        let width = if self.wave64 { "_B64" } else { "_B32" };
        // s_wqm: every lane where the analysis found it so (whole-quad mode of the launch EXEC);
        // any other mask would need the lanes of the quad
        if name.strip_prefix("S_WQM") == Some(width) {
            if !self.sum.wqm_all.contains(&d.pc) {
                return Err(format!("{name} of a mask other than the launch EXEC"));
            }
            let t = self.ir.boolean(true);
            self.write_bit(b, sd, t);
            self.write_scc(b, t);
            return Ok(());
        }
        // mask-width ops: on lane bits when they write EXEC, are a saveexec, or read a mask
        if let Some(base) = name.strip_suffix(width) {
            let op = base.trim_start_matches("S_");
            let saveexec = op.strip_suffix("_SAVEEXEC").or_else(|| op.strip_suffix("_WREXEC"));
            // SOP1 has no SSRC1: only the sources the encoding has
            let srcs: Vec<u64> = std::iter::once(s0).chain(d.enc.field("SSRC1").map(|_| s1)).collect();
            let reads_mask = srcs.iter().any(|&c| c <= 127 && c != 124 && self.holds_mask(b, c as u8));
            let bitwise = matches!(op, "MOV" | "NOT" | "AND" | "OR" | "XOR" | "ANDN2" | "ORN2" | "NAND" | "NOR" | "XNOR");
            if let Some(op) = saveexec {
                let a = self.src_bit(b, d, s0)?;
                let old = self.read_bit(b, EXEC)?;
                let new = self.bit_op(op, a, old)?;
                self.write_bit(b, EXEC, new);
                if sd != 125 {
                    self.write_bit(b, sd, if base.ends_with("_WREXEC") { new } else { old });
                }
                let scc = self.mask_nonzero_scc(new);
                self.write_scc(b, scc);
                return Ok(());
            }
            if op == "CSELECT" && (sd == EXEC || reads_mask) {
                // `s_cselect_b64 s, exec, 0`: the lane's bit of the source SCC picks
                let (a, c) = (self.src_bit(b, d, s0)?, self.src_bit(b, d, s1)?);
                let scc = self.read_scc(b);
                let v = self.ir.select(scc, a, c);
                self.write_bit(b, sd, v);
                return Ok(());
            }
            if bitwise && (sd == EXEC || reads_mask) {
                let a = self.src_bit(b, d, s0)?;
                let v = match op {
                    "MOV" => a,
                    "NOT" => self.not(a),
                    _ => {
                        let c = self.src_bit(b, d, s1)?;
                        self.bit_op(op, a, c)?
                    }
                };
                self.write_bit(b, sd, v);
                if op != "MOV" {
                    let scc = self.mask_nonzero_scc(v);
                    self.write_scc(b, scc);
                }
                return Ok(());
            }
        }
        // mask compares (`s_cmp_lg_u64 exec, 0`): whether the lane's bits differ / agree
        let cmp64 = if self.wave64 { ["S_CMP_EQ_U64", "S_CMP_LG_U64"] } else { ["S_CMP_EQ_U32", "S_CMP_LG_U32"] };
        if cmp64.contains(&name) && [s0, s1].iter().any(|&c| c <= 127 && self.holds_mask(b, c as u8)) {
            let (a, c) = (self.src_bit(b, d, s0)?, self.src_bit(b, d, s1)?);
            let x = self.ir.bin(Bin::Xor, a, c);
            let x = self.mask_nonzero_scc(x);
            let v = if name == cmp64[0] { self.not(x) } else { x };
            self.write_scc(b, v);
            return Ok(());
        }
        // numbers
        let num2 = |l: &mut Self, op: Bin| -> R<I::V> {
            let (a, c) = (l.src(b, d, s0)?, l.src(b, d, s1)?);
            Ok(l.ir.bin(op, a, c))
        };
        let nonzero = |l: &mut Self, v: I::V| {
            let z = l.ir.int(0);
            l.ir.icmp(Cmp::Ne, v, z)
        };
        match name {
            "S_MOV_B32" => {
                let v = self.src(b, d, s0)?;
                self.write_num(b, sd, v);
            }
            "S_MOVK_I32" => {
                let v = self.ir.int(fv("SIMM16").unwrap_or(0) as u16 as i16 as i32 as u32);
                self.write_num(b, sd, v);
            }
            "S_MOV_B64" => {
                let (lo, hi) = self.src64(b, d, s0)?;
                self.write_num(b, sd, lo);
                self.write_num(b, sd + 1, hi);
            }
            "S_ADD_U32" | "S_SUB_U32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let (op, carry) = if name == "S_ADD_U32" { (Bin::Add, true) } else { (Bin::Sub, false) };
                let v = self.ir.bin(op, a, c);
                // carry out: the sum wrapped below a; borrow: a < b
                let scc = if carry { self.ir.icmp(Cmp::Ult, v, a) } else { self.ir.icmp(Cmp::Ult, a, c) };
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            "S_CSELECT_B32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let scc = self.read_scc(b);
                let v = self.ir.select(scc, a, c);
                self.write_num(b, sd, v);
            }
            "S_CSELECT_B64" => {
                let ((al, ah), (cl, ch)) = (self.src64(b, d, s0)?, self.src64(b, d, s1)?);
                let scc = self.read_scc(b);
                let (lo, hi) = (self.ir.select(scc, al, cl), self.ir.select(scc, ah, ch));
                self.write_num(b, sd, lo);
                self.write_num(b, sd + 1, hi);
            }
            "S_ADDC_U32" | "S_SUBB_U32" => {
                // with SCC as the carry in / borrow in, and the carry / borrow out in SCC
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let cin = self.read_scc(b);
                let cin = self.ir.zext(cin);
                let (v, out) = if name == "S_ADDC_U32" {
                    let t = self.ir.bin(Bin::Add, a, c);
                    let v = self.ir.bin(Bin::Add, t, cin);
                    let (c1, c2) = (self.ir.icmp(Cmp::Ult, t, a), self.ir.icmp(Cmp::Ult, v, t));
                    (v, self.ir.bin(Bin::Or, c1, c2))
                } else {
                    let t = self.ir.bin(Bin::Sub, a, c);
                    let v = self.ir.bin(Bin::Sub, t, cin);
                    let (b1, b2) = (self.ir.icmp(Cmp::Ult, a, c), self.ir.icmp(Cmp::Ult, t, cin));
                    (v, self.ir.bin(Bin::Or, b1, b2))
                };
                self.write_num(b, sd, v);
                self.write_scc(b, out);
            }
            "S_ADD_I32" | "S_SUB_I32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let op = if name == "S_ADD_I32" { Bin::Add } else { Bin::Sub };
                let v = self.ir.bin(op, a, c);
                // signed overflow: the operands' signs agree (add) / differ (sub) and the result's differs
                let (av, cv) = (self.ir.bin(Bin::Xor, a, v), self.ir.bin(Bin::Xor, c, v));
                let both = if op == Bin::Add {
                    self.ir.bin(Bin::And, av, cv)
                } else {
                    let ac = self.ir.bin(Bin::Xor, a, c);
                    self.ir.bin(Bin::And, ac, av)
                };
                let z = self.ir.int(0);
                let scc = self.ir.icmp(Cmp::Slt, both, z);
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            "S_MUL_I32" => {
                let v = num2(self, Bin::Mul)?;
                self.write_num(b, sd, v);
            }
            "S_AND_B32" | "S_OR_B32" | "S_XOR_B32" => {
                let op = match name {
                    "S_AND_B32" => Bin::And,
                    "S_OR_B32" => Bin::Or,
                    _ => Bin::Xor,
                };
                let v = num2(self, op)?;
                let scc = nonzero(self, v);
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            "S_LSHL_B32" | "S_LSHR_B32" | "S_ASHR_I32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let c31 = self.ir.int(31);
                let sh = self.ir.bin(Bin::And, c, c31);
                let op = match name {
                    "S_LSHL_B32" => Bin::Shl,
                    "S_LSHR_B32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let v = self.ir.bin(op, a, sh);
                let scc = nonzero(self, v);
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            n if n.starts_with("S_CMP_") && (n.ends_with("_U32") || n.ends_with("_I32")) => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let signed = n.ends_with("_I32");
                let p = match &n[6..8] {
                    "EQ" => Cmp::Eq,
                    "LG" => Cmp::Ne,
                    "GT" => if signed { Cmp::Sgt } else { Cmp::Ugt },
                    "GE" => if signed { Cmp::Sge } else { Cmp::Uge },
                    "LT" => if signed { Cmp::Slt } else { Cmp::Ult },
                    "LE" => if signed { Cmp::Sle } else { Cmp::Ule },
                    _ => return Err(format!("unsupported {n}")),
                };
                let v = self.ir.icmp(p, a, c);
                self.write_scc(b, v);
            }
            "S_NOP" | "S_WAITCNT" | "S_WAITCNT_VSCNT" | "S_WAITCNT_VMCNT" | "S_WAITCNT_LGKMCNT" | "S_WAITCNT_EXPCNT" | "S_INST_PREFETCH" | "S_CLAUSE" => {}
            "S_SETREG_IMM32_B32" | "S_SETREG_B32" => {
                let s = fv("SIMM16").unwrap_or(0);
                return Err(format!("unsupported {name} hwreg {} offset {} size {}", s & 0x3F, (s >> 6) & 0x1F, ((s >> 11) & 0x1F) + 1));
            }
            n => return Err(format!("unsupported {n}")),
        }
        Ok(())
    }

    /// A 64-bit scalar source: a register pair or a constant.
    fn src64(&mut self, b: usize, d: &Decoded, code: u64) -> R<(I::V, I::V)> {
        Ok(match code {
            0..=105 | 106 | 126 => (self.read_num(b, code as u8)?, self.read_num(b, code as u8 + 1)?),
            125 => (self.ir.int(0), self.ir.int(0)),
            128..=192 => (self.ir.int((code - 128) as u32), self.ir.int(0)),
            193..=208 => (self.ir.int((-((code - 192) as i32)) as u32), self.ir.int(u32::MAX)),
            255 => (self.ir.int(d.literal.ok_or("missing literal")?), self.ir.int(0)),
            _ => return Err(format!("64-bit source code {code}")),
        })
    }

    fn valu(&mut self, b: usize, d: &Decoded) -> R<()> {
        if d.inst.name == "V_NOP" {
            return Ok(());
        }
        if matches!(d.variant, Variant::Dpp16 | Variant::Dpp8) {
            self.dpp(b, d)?;
        }
        let r = self.valu_op(b, d);
        (self.dpp_src, self.dpp_en) = (None, None);
        r
    }

    fn valu_op(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        if d.variant == Variant::Sdwa {
            return Err(format!("unsupported {name}: SDWA"));
        }
        if self.wave_valu(b, d)? {
            return Ok(());
        }
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f);
        let vop3 = d.enc.name.starts_with("ENC_VOP3") || d.enc.name.starts_with("VOP3");
        if vop3 && ["ABS", "NEG", "CLAMP", "OMOD", "OP_SEL"].iter().any(|f| fv(f).unwrap_or(0) != 0) {
            return Err(format!("unsupported {name}: VOP3 modifiers"));
        }
        // sources: VOP1 / VOP2 / VOPC put src1 in VSRC1 (a VGPR); VOP3 in SRC1
        let s0 = fv("SRC0").unwrap_or(0);
        let s1 = if vop3 { fv("SRC1").unwrap_or(0) } else { fv("VSRC1").map(|v| v + 256).unwrap_or(0) };
        let vdst = fv("VDST").unwrap_or(0);
        if let Some(rest) = name.strip_prefix("V_CMP") {
            let (x, cond) = match rest.strip_prefix("X_") {
                Some(c) => (true, c),
                None => (false, rest.trim_start_matches('_')),
            };
            let (pred, ty) = cond.split_once('_').ok_or_else(|| format!("unsupported {name}"))?;
            let signed = match ty {
                "U32" => false,
                "I32" => true,
                _ => return Err(format!("unsupported {name}")),
            };
            let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
            let c = match pred {
                "EQ" => self.ir.icmp(Cmp::Eq, a, c),
                "NE" => self.ir.icmp(Cmp::Ne, a, c),
                "GT" => self.ir.icmp(if signed { Cmp::Sgt } else { Cmp::Ugt }, a, c),
                "GE" => self.ir.icmp(if signed { Cmp::Sge } else { Cmp::Uge }, a, c),
                "LT" => self.ir.icmp(if signed { Cmp::Slt } else { Cmp::Ult }, a, c),
                "LE" => self.ir.icmp(if signed { Cmp::Sle } else { Cmp::Ule }, a, c),
                "F" => self.ir.boolean(false),
                "T" => self.ir.boolean(true),
                _ => return Err(format!("unsupported {name}")),
            };
            // the lane's result, 0 in inactive lanes
            let e = self.read_bit(b, EXEC)?;
            let bit = self.ir.bin(Bin::And, c, e);
            let dst = if x { EXEC } else if vop3 { vdst as u8 } else { VCC };
            self.write_bit(b, dst, bit);
            return Ok(());
        }
        let v = match name {
            "V_MOV_B32" => self.src(b, d, s0)?,
            "V_CNDMASK_B32" => {
                let sel = if vop3 { fv("SRC2").unwrap_or(0) } else { VCC as u64 };
                let m = self.src_bit(b, d, sel)?;
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                self.ir.select(m, c, a)
            }
            _ => {
                // the opcode first: an unsupported VOP1 has no SRC1, which would read as s0
                const TWO_SOURCE: &[&str] = &[
                    "V_ADD_NC_U32", "V_ADD_U32", "V_SUB_NC_U32", "V_SUB_U32", "V_SUBREV_NC_U32", "V_SUBREV_U32", "V_AND_B32", "V_OR_B32", "V_XOR_B32",
                    "V_MUL_LO_U32", "V_LSHL_ADD_U32", "V_ADD_LSHL_U32", "V_LSHL_OR_B32", "V_AND_OR_B32", "V_OR3_B32", "V_XOR3_B32", "V_ADD3_U32",
                    "V_LSHLREV_B32", "V_LSHRREV_B32", "V_ASHRREV_I32", "V_MIN_U32", "V_MAX_U32", "V_MIN_I32", "V_MAX_I32",
                ];
                if !TWO_SOURCE.contains(&name) {
                    return Err(format!("unsupported {name}"));
                }
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let c31 = self.ir.int(31);
                match name {
                    "V_ADD_NC_U32" | "V_ADD_U32" => self.ir.bin(Bin::Add, a, c),
                    "V_SUB_NC_U32" | "V_SUB_U32" => self.ir.bin(Bin::Sub, a, c),
                    "V_SUBREV_NC_U32" | "V_SUBREV_U32" => self.ir.bin(Bin::Sub, c, a),
                    "V_AND_B32" => self.ir.bin(Bin::And, a, c),
                    "V_OR_B32" => self.ir.bin(Bin::Or, a, c),
                    "V_XOR_B32" => self.ir.bin(Bin::Xor, a, c),
                    "V_MUL_LO_U32" => self.ir.bin(Bin::Mul, a, c),
                    // three-source integer ops (VOP3): shifts take the low 5 bits of their amount
                    "V_LSHL_ADD_U32" | "V_ADD_LSHL_U32" | "V_LSHL_OR_B32" | "V_AND_OR_B32" | "V_OR3_B32" | "V_XOR3_B32" | "V_ADD3_U32" => {
                        let e = self.src(b, d, fv("SRC2").unwrap_or(0))?;
                        match name {
                            "V_LSHL_ADD_U32" | "V_LSHL_OR_B32" => {
                                let sh = self.ir.bin(Bin::And, c, c31);
                                let x = self.ir.bin(Bin::Shl, a, sh);
                                self.ir.bin(if name == "V_LSHL_ADD_U32" { Bin::Add } else { Bin::Or }, x, e)
                            }
                            "V_ADD_LSHL_U32" => {
                                let x = self.ir.bin(Bin::Add, a, c);
                                let sh = self.ir.bin(Bin::And, e, c31);
                                self.ir.bin(Bin::Shl, x, sh)
                            }
                            "V_AND_OR_B32" => {
                                let x = self.ir.bin(Bin::And, a, c);
                                self.ir.bin(Bin::Or, x, e)
                            }
                            _ => {
                                let op = match name {
                                    "V_OR3_B32" => Bin::Or,
                                    "V_XOR3_B32" => Bin::Xor,
                                    _ => Bin::Add,
                                };
                                let x = self.ir.bin(op, a, c);
                                self.ir.bin(op, x, e)
                            }
                        }
                    }
                    "V_LSHLREV_B32" | "V_LSHRREV_B32" | "V_ASHRREV_I32" => {
                        let sh = self.ir.bin(Bin::And, a, c31);
                        let op = match name {
                            "V_LSHLREV_B32" => Bin::Shl,
                            "V_LSHRREV_B32" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.ir.bin(op, c, sh)
                    }
                    "V_MIN_U32" | "V_MAX_U32" | "V_MIN_I32" | "V_MAX_I32" => {
                        let pred = match name {
                            "V_MIN_U32" => Cmp::Ult,
                            "V_MAX_U32" => Cmp::Ugt,
                            "V_MIN_I32" => Cmp::Slt,
                            _ => Cmp::Sgt,
                        };
                        let cond = self.ir.icmp(pred, a, c);
                        self.ir.select(cond, a, c)
                    }
                    n => return Err(format!("unsupported {n}")),
                }
            }
        };
        self.write_v(b, d, vdst as u16, v)
    }

    /// `s_load_dword{,x2,x4,x8,x16}`: dwords of the flat memory into SGPRs, from
    /// `s[sbase] + offset + soffset` (one memory window, as `emit::smem` with `mem_windows` 1).
    fn smem(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        if self.ir.mem_windows() > 1 {
            return Err(format!("unsupported {name}: several memory windows"));
        }
        let count = match name.strip_prefix("S_LOAD_") {
            Some("DWORD") => 1,
            Some("DWORDX2") => 2,
            Some("DWORDX4") => 4,
            Some("DWORDX8") => 8,
            Some("DWORDX16") => 16,
            _ => return Err(format!("unsupported {name}")),
        };
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f);
        let (sdata, sbase) = (fv("SDATA") as u8, fv("SBASE") as u8);
        let imm = (((fv("OFFSET") as u32) << 11) as i32 >> 11) as u32; // 21-bit signed
        let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
        let base = self.read_num(b, sbase)?;
        let imm = self.ir.int(imm);
        let mut a = self.ir.bin(Bin::Add, base, imm);
        if soff != 125 {
            let s = self.src(b, d, soff)?;
            a = self.ir.bin(Bin::Add, a, s);
        }
        let m = self.ir.int(!3);
        let a = self.ir.bin(Bin::And, a, m);
        let vals: Vec<I::V> = (0..count)
            .map(|i| {
                let o = self.ir.int(4 * i as u32);
                let ai = self.ir.bin(Bin::Add, a, o);
                self.ir.mem_load(ai)
            })
            .collect();
        for (i, v) in vals.into_iter().enumerate() {
            self.write_num(b, sdata + i as u8, v);
        }
        Ok(())
    }

    /// `global_` / `flat_` `load` / `store_dword{,x2,x3,x4}` on the flat memory. A standalone
    /// guarded load keeps the old register in lanes without their EXEC bit (`write_v`); guarded
    /// stores and coalesced runs are lowered unguarded in a block only those lanes enter (`splits`).
    fn vmem(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        if self.ir.mem_windows() > 1 {
            return Err(format!("unsupported {name}: several memory windows"));
        }
        let op = name.trim_start_matches("GLOBAL_").trim_start_matches("FLAT_");
        let (store, count) = match op {
            "LOAD_DWORD" => (false, 1),
            "LOAD_DWORDX2" => (false, 2),
            "LOAD_DWORDX3" => (false, 3),
            "LOAD_DWORDX4" => (false, 4),
            "STORE_DWORD" => (true, 1),
            "STORE_DWORDX2" => (true, 2),
            "STORE_DWORDX3" => (true, 3),
            "STORE_DWORDX4" => (true, 4),
            _ => return Err(format!("unsupported {name}")),
        };
        let global = d.enc.name == "ENC_FLAT_GLBL";
        if !global && d.enc.name != "ENC_FLAT" {
            return Err(format!("{name}: {}", d.enc.name));
        }
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
        let raw = fv("OFFSET") as u32;
        let off = if global { ((raw << 20) as i32 >> 20) as u32 } else { raw };
        let saddr = fv("SADDR");
        let mut a = self.read(b, Var::V(fv("ADDR") as u16));
        if global && saddr != 0x7D && saddr != 0x7F {
            let s = self.read_num(b, saddr as u8)?;
            a = self.ir.bin(Bin::Add, s, a);
        }
        let off = self.ir.int(off);
        let a = self.ir.bin(Bin::Add, a, off);
        for i in 0..count {
            let o = self.ir.int(4 * i as u32);
            let ai = self.ir.bin(Bin::Add, a, o);
            let m = self.ir.int(!3);
            let ai = self.ir.bin(Bin::And, ai, m);
            if store {
                let v = self.read(b, Var::V(fv("DATA") as u16 + i as u16));
                self.ir.mem_store(ai, v);
            } else {
                let v = self.ir.mem_load(ai);
                self.write_v(b, d, fv("VDST") as u16 + i as u16, v)?;
            }
        }
        Ok(())
    }

    /// The condition under which a lane takes the branch ending block b.
    fn cond(&mut self, b: usize, d: &Decoded) -> R<I::V> {
        Ok(match d.inst.name {
            "S_CBRANCH_SCC1" => self.read_scc(b),
            "S_CBRANCH_SCC0" => {
                let s = self.read_scc(b);
                self.not(s)
            }
            n @ ("S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ") => {
                let v = self.read_bit(b, VCC)?;
                let c = if self.wave_branches { self.mask_nonzero(v) } else { v };
                if n == "S_CBRANCH_VCCZ" {
                    self.not(c)
                } else {
                    c
                }
            }
            "S_CBRANCH_EXECNZ" => {
                let e = self.read_bit(b, EXEC)?;
                self.mask_nonzero(e)
            }
            "S_CBRANCH_EXECZ" => {
                let e = self.read_bit(b, EXEC)?;
                let c = self.mask_nonzero(e);
                self.not(c)
            }
            n => return Err(format!("unsupported branch {n}")),
        })
    }

    /// Whether a lane reading mask bit v in block c may see its own stale bit: v (or a phi
    /// operand reaching it) was written in another block that not every lane passed through.
    /// The hardware's whole-wave write left the absent lanes' bits 0 there; this lane kept its
    /// own older one. EXEC is not concerned: an absent lane's bit is 0 in both.
    fn stale(&self, c: usize, v: I::V) -> bool {
        let mut seen = HashSet::new();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            let v = self.ssa.resolve_value(v);
            if !seen.insert(v) {
                continue;
            }
            match self.ssa.phi_operands(v) {
                Some(ops) => work.extend_from_slice(ops),
                None => {
                    if let Some(&b) = self.def_block.get(&v) {
                        if self.def_owner[b] != self.use_owner[c] && self.partial[b] {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Whether v is (or is a phi reaching) `marker`.
    fn reaches(&self, v: I::V, marker: I::V) -> bool {
        let mut seen = HashSet::new();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            let v = self.ssa.resolve_value(v);
            if v == marker {
                return true;
            }
            if seen.insert(v) {
                if let Some(ops) = self.ssa.phi_operands(v) {
                    work.extend_from_slice(ops);
                }
            }
        }
        false
    }
}

/// Per analysis block: whether some lane that reached a lane branch may not pass through it
/// before the lanes meet again – it is reachable from the branch but does not post-dominate
/// both of its successors (lanes that take the other one never get there; a loop's own latch
/// counts: the lanes that leave do not come back to it).
fn partial_blocks(sum: &Summary) -> Vec<bool> {
    let n = sum.blocks.len();
    if sum.lane_branches == 0 {
        return vec![false; n];
    }
    let words = (n + 1).div_ceil(64);
    // post-dominators over the blocks the entry reaches, with a virtual exit n
    let full = vec![!0u64; words];
    let mut pdom: Vec<Vec<u64>> = vec![full; n + 1];
    pdom[n] = vec![0u64; words];
    pdom[n][n / 64] |= 1 << (n % 64);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in sum.order.iter().rev() {
            let succ: Vec<usize> = if sum.blocks[b].succ.is_empty() { vec![n] } else { sum.blocks[b].succ.iter().map(|s| s.0).collect() };
            let mut new = vec![!0u64; words];
            for s in succ {
                for (w, x) in new.iter_mut().zip(&pdom[s]) {
                    *w &= x;
                }
            }
            new[b / 64] |= 1 << (b % 64);
            if new != pdom[b] {
                pdom[b] = new;
                changed = true;
            }
        }
    }
    let mut partial = vec![false; n];
    for &x in &sum.order {
        if !sum.blocks[x].lane_branch {
            continue;
        }
        let mut seen = vec![false; n];
        let mut work: Vec<usize> = sum.blocks[x].succ.iter().map(|s| s.0).collect();
        while let Some(b) = work.pop() {
            if std::mem::replace(&mut seen[b], true) {
                continue;
            }
            if sum.blocks[x].succ.iter().any(|&(s, _)| pdom[s][b / 64] >> (b % 64) & 1 == 0) {
                partial[b] = true;
            }
            work.extend(sum.blocks[b].succ.iter().map(|s| s.0));
        }
    }
    partial
}

/// Lowers a kernel-mode program (far jumps resolved) into `ir`, which gets it complete, with
/// the mask registers whose value at the end may differ from the console's: written in a block
/// some lanes skipped, so those lanes still hold their own older bits (`stale`). Nothing reads
/// them in the program (that would be refused); only the kernel ABI's store of every register.
pub fn lower_kernel<I: Ir>(insts: &[Decoded], wave64: bool, ir: I) -> R<(I, Vec<u8>)> {
    lower(insts, Stage::Kernel, wave64, ir)
}

/// Lowers a compute program (far jumps resolved) into `ir`.
pub fn lower_compute<I: Ir>(insts: &[Decoded], cs: &super::CsInterface, wave64: bool, ir: I) -> R<I> {
    Ok(lower(insts, Stage::Compute(cs), wave64, ir)?.0)
}

/// Lowers a vertex or pixel shader (`pixel`; far jumps resolved) into `ir`. `lanes32`: a wave64
/// program runs in 32-lane hardware waves (NVIDIA), as the HLSL path runs it with `upper_lanes`;
/// the per-lane form takes such a program only where it reads no lane 32–63 that holds anything.
pub fn lower_graphics<I: Ir>(insts: &[Decoded], pixel: bool, wave64: bool, lanes32: bool, ir: I) -> R<I> {
    if wave64 && lanes32 && super::upper_lanes::analyze(insts).reads.values().any(|&r| r != super::upper_lanes::Read::Native) {
        return Err("wave64 VS / PS in 32-lane waves: a read of lanes 32–63".into());
    }
    Ok(lower(insts, if pixel { Stage::Pixel } else { Stage::Vertex }, wave64, ir)?.0)
}

fn lower<I: Ir>(insts: &[Decoded], stage: Stage, wave64: bool, ir: I) -> R<(I, Vec<u8>)> {
    lower_with(insts, stage, wave64, ir, false).map(|(ir, stale, _)| (ir, stale))
}

/// Backward liveness of SCC over `sum.blocks`: `live_after[i]` is true when some path
/// from instruction `i` can read SCC before an unconditional scalar instruction overwrites it.
fn compute_scc_liveness(insts: &[Decoded], sum: &Summary) -> Vec<bool> {
    let reads_scc = |d: &Decoded| -> bool {
        matches!(d.inst.name, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1")
            || d.ienc.operands.iter().any(|o| o.input && o.ty == "OPR_SSRC_SPECIAL_SCC")
            || ["SSRC0", "SSRC1", "SRC0", "SRC1", "SRC2", "SOFFSET"]
                .iter()
                .any(|f| rdna2::field_value(d.enc, &d.words, f) == Some(253))
    };
    let writes_scc = |d: &Decoded| -> bool {
        d.inst.name.starts_with("S_")
            && d.ienc.operands.iter().any(|o| o.output && o.ty == "OPR_SSRC_SPECIAL_SCC")
    };
    let nb = sum.blocks.len();
    let mut live_in = vec![false; nb];
    let mut live_after = vec![false; insts.len()];
    loop {
        let mut changed = false;
        for &b in sum.order.iter().rev() {
            let block = &sum.blocks[b];
            let mut live = block.succ.iter().any(|&(s, _)| live_in[s]);
            for i in block.insts.clone().rev() {
                live_after[i] = live;
                let d = &insts[i];
                if writes_scc(d) {
                    live = false;
                }
                if reads_scc(d) {
                    live = true;
                }
            }
            if live_in[b] != live {
                live_in[b] = live;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    live_after
}

/// Survey: every reason the per-lane form refuses a compute program, not only the first. An
/// instruction it refuses is left out and the lowering goes on, so a later reason may follow
/// from an earlier one; a program the lane-mask analysis refuses gives that one reason.
pub fn survey_compute<I: Ir>(insts: &[Decoded], cs: &super::CsInterface, wave64: bool, ir: I) -> Vec<String> {
    match lower_with(insts, Stage::Compute(cs), wave64, ir, true) {
        Ok((_, _, noted)) => noted,
        Err(e) => vec![e],
    }
}

fn lower_with<I: Ir>(insts: &[Decoded], stage: Stage, wave64: bool, mut ir: I, survey: bool) -> R<(I, Vec<u8>, Vec<String>)> {
    let kernel = matches!(stage, Stage::Kernel);
    let graphics = matches!(stage, Stage::Vertex | Stage::Pixel);
    // a PS: lanes may run outside the launch EXEC (its helper lanes); a VS has none, each D3D
    // vertex invocation being one lane
    let pixel = matches!(stage, Stage::Pixel);
    let launch = if pixel { lane_masks::LaunchExec::Unknown } else { lane_masks::LaunchExec::All };
    // Ballots reconstruct every bit only when the entire console wave is present.
    let wave_branches = matches!(stage, Stage::Compute(cs) if cs.threads.iter().product::<u32>() % (if wave64 { 64 } else { 32 }) == 0)
        && ir.can_wave_branch();
    let sum = lane_masks::analyze_branches(insts, wave64, launch, wave_branches).map_err(|e| e.to_string())?;
    let scc_live_after = if wave_branches {
        compute_scc_liveness(insts, &sum)
    } else {
        vec![true; insts.len()]
    };
    let n = sum.blocks.len();
    if n == 0 {
        return Err("empty program".into());
    }
    // IR blocks: 0 prologue, 1..=n the program's, n + 1 exit, n + 2 lane 0's scalar stores,
    // n + 3 every lane's vector stores, n + 4 the way to the exit of lanes outside EXEC at launch,
    // then for every coalesced guarded run (only the lanes with their EXEC bit enter): the run's
    // block and the one the rest of its block continues in. `tail[b]`: the IR block the
    // program's block b ends in.
    let (exit, sstore, vstore, skip) = (n + 1, n + 2, n + 3, n + 4);
    let guarded: HashSet<u64> = sum.guarded.iter().copied().collect();
    let mut splits: Vec<Vec<(std::ops::Range<usize>, usize, usize)>> = vec![Vec::new(); n]; // (run, body, cont)
    let mut count = n + 5;
    for (b, block) in sum.blocks.iter().enumerate() {
        for run in coalesced_runs(insts, &block.insts, &guarded) {
            splits[b].push((run, count, count + 1));
            count += 2;
        }
    }
    let tail: Vec<usize> = (0..n).map(|b| splits[b].last().map_or(blk(b), |s| s.2)).collect();
    let mut partial = vec![false; count];
    let mut def_owner: Vec<usize> = (0..count).collect();
    let mut use_owner: Vec<usize> = (0..count).collect();
    for (b, p) in partial_blocks(&sum).into_iter().enumerate() {
        partial[blk(b)] = p;
        for &(_, s, c) in &splits[b] {
            partial[s] = true;
            partial[c] = p;
            def_owner[s] = s;
            use_owner[s] = blk(b);
            def_owner[c] = blk(b);
            use_owner[c] = blk(b);
        }
    }
    let ends: Vec<usize> = sum.order.iter().copied().filter(|&b| sum.blocks[b].succ.is_empty()).collect();
    let mut preds = vec![Vec::new(); count];
    preds[blk(0)] = vec![0];
    for &b in &sum.order {
        let mut seen = Vec::new();
        for &(s, _) in &sum.blocks[b].succ {
            if !seen.contains(&s) {
                seen.push(s);
                preds[blk(s)].push(tail[b]);
            }
        }
        let mut prev = blk(b);
        for &(_, s, c) in &splits[b] {
            preds[s] = vec![prev];
            preds[c] = vec![prev, s];
            prev = c;
        }
    }
    // compute: every lane that runs is in EXEC, and nothing is stored at the end – the skip and
    // store blocks are left unreachable (a bare `ret`)
    if kernel {
        preds[skip] = vec![0];
        preds[sstore] = vec![exit];
        preds[vstore] = vec![exit, sstore];
    }
    preds[exit] = kernel.then_some(skip).into_iter().chain(ends.iter().map(|&b| tail[b])).collect();
    for i in 0..count {
        let id = ir.new_block();
        debug_assert_eq!(id, i);
    }
    let num_marker = ir.undef(Var::S(0));
    let bit_marker = ir.undef(Var::M(0));
    let mut l = Lower {
        ssa: Ssa::new(preds.clone()),
        ir,

        sum: &sum,
        guarded,
        wave_branches,
        scc_live_after,
        cur_inst: 0,
        wave64,
        num_marker,
        bit_marker,
        num_uses: Vec::new(),
        bit_uses: Vec::new(),
        lane: None,
        partial,
        track_frexp_denorm: super::emit::tracks_frexp_f32_denorm(insts),
        def_owner,
        use_owner,
        unguarded: false,
        def_block: std::collections::HashMap::new(),
        lane_uses: Vec::new(),
        dpp_src: None,
        dpp_en: None,
        noted: survey.then(Vec::new),
        graphics,
    };
    // the registers the program names, and the ones the runner reads back
    let mut sregs: Vec<u8> = vec![VCC, VCC + 1, 124, EXEC, EXEC + 1];
    let mut vregs: Vec<u16> = Vec::new();
    for d in insts {
        let (r, w) = rdna2::sgpr_uses(d);
        sregs.extend(r.iter().chain(&w).map(|&x| x as u8));
        let (r, w) = rdna2::vgpr_uses(d);
        vregs.extend(r.iter().chain(&w).map(|&x| x as u16));
    }
    sregs.retain(|&r| (r as usize) < KERNEL_SGPRS && r != 125);
    sregs.sort_unstable();
    sregs.dedup();
    vregs.sort_unstable();
    vregs.dedup();

    // prologue: the SGPRs from `io` (kernel: the whole register file; compute: the user data)
    l.ir.at(0);
    l.ssa.seal(&mut l.ir, 0);
    let lane = l.lane();
    for &r in &sregs {
        let a = l.ir.int(r as u32);
        let v = l.ir.io_load(a);
        l.write_num(0, r, v);
    }
    let mut row = None;
    match stage {
        Stage::Kernel => {
            let a = l.ir.int(KERNEL_SCC_SLOT as u32);
            let scc = l.ir.io_load(a);
            let z = l.ir.int(0);
            let scc = l.ir.icmp(Cmp::Ne, scc, z);
            l.ssa.write(0, Var::Scc, scc);
            let (base, stride) = (l.ir.int(KERNEL_SGPRS as u32), l.ir.int(KERNEL_VGPRS as u32));
            let r = l.ir.bin(Bin::Mul, lane, stride);
            let r = l.ir.bin(Bin::Add, base, r);
            for &v in &vregs {
                let o = l.ir.int(v as u32);
                let a = l.ir.bin(Bin::Add, r, o);
                let x = l.ir.io_load(a);
                l.ssa.write(0, Var::V(v), x);
            }
            row = Some(r);
            // lanes outside EXEC go straight to the end
            let active = l.read_bit(0, EXEC)?;
            l.write_bit(0, EXEC, active);
            l.ir.br_if(active, blk(0), skip);
            // … with their bits of every mask 0, as a whole-wave write leaves them
            l.ir.at(skip);
            l.ssa.seal(&mut l.ir, skip);
            let f = l.ir.boolean(false);
            for &r in &sregs {
                l.ssa.write(skip, Var::M(r), f);
            }
            l.ir.br(exit);
        }
        Stage::Compute(cs) => {
            let (flat, tid) = l.ir.compute_ids(cs.threads);
            // system SGPRs after the user data: the enabled group ids, then TG_SIZE
            let mut next = cs.user_sgprs as u8;
            for k in 0..3 {
                if cs.tgid[k as usize] {
                    let g = l.ir.group_id(k);
                    l.write_num(0, next, g);
                    next += 1;
                }
            }
            if cs.tg_size {
                // waves in the group (a last partial one counts) | this wave's index << 6
                let lanes = if wave64 { 64 } else { 32 };
                let waves = l.ir.int(cs.threads.iter().product::<u32>().div_ceil(lanes));
                let sh = l.ir.int(lanes.trailing_zeros());
                let w = l.ir.bin(Bin::LShr, flat, sh);
                let six = l.ir.int(6);
                let w = l.ir.bin(Bin::Shl, w, six);
                let v = l.ir.bin(Bin::Or, waves, w);
                l.write_num(0, next, v);
            }
            let f = l.ir.boolean(false);
            l.ssa.write(0, Var::Scc, f);
            for &v in &vregs {
                let x = match v {
                    0 => tid[0],
                    1 if cs.tid_components >= 2 => tid[1],
                    2 if cs.tid_components >= 3 => tid[2],
                    _ => l.ir.int(0),
                };
                l.ssa.write(0, Var::V(v), x);
            }
            // every lane that runs is in EXEC
            let t = l.ir.boolean(true);
            l.write_bit(0, EXEC, t);
            l.ir.br(blk(0));
            l.ir.at(skip);
            l.ssa.seal(&mut l.ir, skip);
            l.ir.ret();
        }
        Stage::Vertex | Stage::Pixel => {
            let Some(start) = l.ir.graphics_start(&vregs) else {
                return Err("VS / PS: this IR has no graphics entry".into());
            };
            for &(r, v) in &start.sgprs {
                l.write_num(0, r, v);
            }
            let f = l.ir.boolean(false);
            l.ssa.write(0, Var::Scc, f);
            for &v in &vregs {
                let x = match start.vgprs.iter().find(|x| x.0 == v) {
                    Some(&(_, x)) => x,
                    None => l.ir.int(0),
                };
                l.ssa.write(0, Var::V(v), x);
            }
            // every lane runs the program: a PS's helper lanes with their EXEC bit clear (the
            // kernel's way out for lanes outside EXEC would leave their quads short)
            l.write_bit(0, EXEC, start.exec);
            l.ir.br(blk(0));
            l.ir.at(skip);
            l.ssa.seal(&mut l.ir, skip);
            l.ir.ret();
        }
    }

    // the program, in reverse postorder; a block is sealed once all its predecessors are in
    let mut done = vec![false; count];
    done[0] = true;
    done[skip] = true;
    let seal_ready = |l: &mut Lower<I>, done: &[bool], b: usize| {
        if preds[b].iter().all(|&p| done[p]) {
            l.ssa.seal(&mut l.ir, b);
            true
        } else {
            false
        }
    };
    let mut sealed = vec![false; count];
    sealed[0] = true;
    for &b in &sum.order.clone() {
        let ib = blk(b);
        l.ir.at(ib);
        if !sealed[ib] {
            sealed[ib] = seal_ready(&mut l, &done, ib);
        }
        if b == 0 && !pixel {
            // the analysis takes the lanes that start the program as its EXEC (a PS keeps the
            // launch EXEC from its start: there the analysis does not know it)
            let t = l.ir.boolean(true);
            l.write_bit(ib, EXEC, t);
        }
        let block = &sum.blocks[b];
        let last = block.insts.end - 1;
        let body_end = block_body_end(insts, &block.insts);
        let mut cur = ib;
        let mut split = splits[b].iter().peekable();
        let mut i = block.insts.start;
        while i < body_end {
            if let Some(&(ref run, s, c)) = split.next_if(|x| x.0.start == i) {
                // a coalesced guarded run: only the lanes with their EXEC bit enter `s`,
                // lowered unguarded; updated registers merge through SSA phis at `c`
                let e = l.read_bit(cur, EXEC)?;
                l.ir.br_if(e, s, c);
                done[cur] = true;
                l.ir.at(s);
                l.ssa.seal(&mut l.ir, s);
                l.unguarded = true;
                for j in run.clone() {
                    l.cur_inst = j;
                    let r = l.lower_inst(s, &insts[j]);
                    l.note_inst(r, s, &insts[j])?;
                }
                l.unguarded = false;
                l.ir.br(c);
                done[s] = true;
                l.ir.at(c);
                l.ssa.seal(&mut l.ir, c);
                cur = c;
                i = run.end;
            } else {
                let name = insts[i].inst.name;
                if matches!(stage, Stage::Pixel) && whole_quad(name) && l.partial[cur] {
                    l.note_inst(Err(format!("{name}: needs whole quads, in a block not every lane reaches")), cur, &insts[i])?;
                    i += 1;
                    continue;
                }
                l.cur_inst = i;
                let r = l.lower_inst(cur, &insts[i]);
                l.note_inst(r, cur, &insts[i])?;
                i += 1;
            }
        }
        match block.succ.as_slice() {
            [] => l.ir.br(exit),
            [(t, Edge::Always)] => l.ir.br(blk(*t)),
            [(t, _), (f, _)] if t == f => l.ir.br(blk(*t)),
            [(t, Edge::Taken), (f, Edge::NotTaken)] => match l.cond(cur, &insts[last]) {
                Ok(c) => l.ir.br_if(c, blk(*t), blk(*f)),
                Err(e) => {
                    l.note(Err(e))?;
                    let c = l.ir.boolean(true);
                    l.ir.br_if(c, blk(*t), blk(*f));
                }
            },
            s => {
                l.note(Err(format!("block {b}: successors {s:?}")))?;
                l.ir.br(blk(s[0].0));
            }
        }
        done[cur] = true;
        for &(s, _) in &block.succ {
            let is = blk(s);
            if !sealed[is] && l.sum.order.contains(&s) {
                // the successor's own code comes later; sealing it early is fine once its
                // predecessors are all in
                if preds[is].iter().all(|&p| done[p]) {
                    l.ssa.seal(&mut l.ir, is);
                    sealed[is] = true;
                }
            }
        }
    }
    for b in 1..=n {
        if !sealed[b] && preds[b].iter().all(|&p| done[p]) {
            l.ssa.seal(&mut l.ir, b);
            sealed[b] = true;
        }
    }

    l.ir.at(exit);
    l.ssa.seal(&mut l.ir, exit);
    let mut stale: Vec<u8> = Vec::new();
    let Some(row) = row else {
        // compute: nothing to store; a VS / PS: its end
        if graphics {
            l.ir.graphics_end();
        }
        l.ir.ret();
        for b in [sstore, vstore] {
            l.ir.at(b);
            l.ssa.seal(&mut l.ir, b);
            l.ir.ret();
        }
        return l.finish(stale);
    };
    // kernel exit: every lane is here; masks the runner reads as numbers are the wave's ballots
    let mut sval: Vec<(u8, I::V)> = Vec::new();
    let mut i = 0;
    while i < sregs.len() {
        let r = sregs[i];
        i += 1;
        let s = l.read(exit, Var::S(r));
        if l.ssa.resolve_value(s) != l.num_marker && !l.reaches(s, l.num_marker) {
            l.num_uses.push(s);
            if l.stale(exit, s) {
                stale.push(r);
            }
            sval.push((r, s));
            continue;
        }
        if !l.holds_mask(exit, r) {
            if l.wave64 && r > 0 && sval.iter().any(|&(x, _)| x == r) {
                continue; // the upper half of a mask stored with its pair
            }
            return Err(format!("s{r}: neither a number nor a mask at the end"));
        }
        // read directly: a stale bit here is reported, not refused (only the ABI's store sees it)
        let m = l.read(exit, Var::M(r));
        if r != EXEC && l.stale(exit, m) {
            stale.push(r);
            if l.wave64 {
                stale.push(r + 1); // the upper half comes from the same mask
            }
        }
        let [lo, hi] = l.ir.ballot(m);
        sval.push((r, lo));
        if l.wave64 {
            sval.retain(|&(x, _)| x != r + 1);
            sval.push((r + 1, hi));
            if sregs.get(i) == Some(&(r + 1)) {
                i += 1;
            }
        }
    }
    let scc = l.read(exit, Var::Scc);
    if l.stale(exit, scc) {
        stale.push(KERNEL_SCC_SLOT as u8);
    }
    let vval: Vec<(u16, I::V)> = vregs.iter().map(|&v| (v, l.read(exit, Var::V(v)))).collect();
    let z = l.ir.int(0);
    let first = l.ir.icmp(Cmp::Eq, lane, z);
    l.ir.br_if(first, sstore, vstore);
    l.ir.at(sstore);
    l.ssa.seal(&mut l.ir, sstore);
    for &(r, v) in &sval {
        let a = l.ir.int(r as u32);
        l.ir.res_store(a, v);
    }
    let a = l.ir.int(KERNEL_SCC_SLOT as u32);
    let s = l.ir.zext(scc);
    l.ir.res_store(a, s);
    l.ir.br(vstore);
    l.ir.at(vstore);
    l.ssa.seal(&mut l.ir, vstore);
    for &(v, x) in &vval {
        let o = l.ir.int(v as u32);
        let a = l.ir.bin(Bin::Add, row, o);
        l.ir.res_store(a, x);
    }
    l.ir.ret();
    l.finish(stale)
}

impl<I: Ir> Lower<'_, I> {
    /// Every phi is complete: no number may come from a mask, nor a mask bit from a number, nor
    /// a lane read a mask bit it kept from before a write it skipped.
    fn finish(mut self, mut stale: Vec<u8>) -> R<(I, Vec<u8>, Vec<String>)> {
        if self.num_uses.iter().any(|&v| self.reaches(v, self.num_marker)) {
            self.note(Err("a mask read as a number (through a join)".into()))?;
        }
        if self.bit_uses.iter().any(|&v| self.reaches(v, self.bit_marker)) {
            self.note(Err("a number read as a mask (through a join)".into()))?;
        }
        if self.lane_uses.iter().any(|&(c, v)| self.stale(c, v)) {
            self.note(Err("a value written while some lanes were elsewhere, read after they met again".into()))?;
        }
        stale.sort_unstable();
        stale.dedup();
        Ok((self.ir, stale, self.noted.unwrap_or_default()))
    }

    /// A refused instruction in block b: as `note`, and in survey mode the SGPRs (and, for a
    /// scalar instruction, SCC) it writes are left as 0 in both views (number and mask bit),
    /// so that a later read does not report a refusal that only follows from this one.
    /// An instruction this lowering has no semantics of, through `Ir::foreign` (#1426's
    /// per-instruction lowering) with the lane-mask analysis's guard: `true` where EXEC is proven
    /// (or inside a coalesced guarded run's block, `in_guarded_run`), else the lane's EXEC bit. Its SGPR
    /// operands are read as numbers (a mask read whole is refused, as everywhere), the mask a
    /// VALU reads lane by lane (`v_cndmask_b32`'s selector, a carry-in) as a number holding only
    /// this lane's bit, and a mask it writes (a compare, a carry-out) comes back as this lane's
    /// bit of the ballot it makes. Of the instructions writing EXEC, only `v_cmpx_*` is taken.
    fn bridge(&mut self, b: usize, d: &Decoded, in_guarded_run: bool) -> R<()> {
        let name = d.inst.name;
        let (whole, written) = super::x64_masks::uses(d);
        let (_, explicit_w) = rdna2::sgpr_uses(d);
        let writes = explicit_w.iter().fold(written, |m, &r| m | 1u128 << r);
        // `v_cmpx_*`: EXEC goes in as this lane's bit and comes back as its bit of the result
        let cmpx = name.starts_with("V_CMPX");
        if writes & (3 << 126) != 0 && !cmpx {
            return Err("writes EXEC".into());
        }
        if name == "S_BARRIER" && self.partial[b] {
            return Err("a barrier where some lanes are elsewhere".into());
        }
        if self.track_frexp_denorm && self.partial[b] && super::emit::writes_frexp_f32_denorm(d) {
            return Err("a FREXP denormal mode write where some lanes are elsewhere".into());
        }
        // instructions whose #1426 semantics are known to differ from the HLSL path and the
        // console (each reported on #1426): left to the fallback until fixed there
        const DIFFERS_ON_1426: &[&str] = &[
            // f32 → f16 of a value below the smallest f16 denormal: 0 instead of rounding to it
            "V_CVT_F16_F32",
            // f16 products round toward zero (`LegacyF32ToF16`) where the console rounds to
            // nearest even (vladbondarenko97's f16 programs; itsSio is fixing it on `main`)
            "V_MUL_F16",
            "V_PK_MUL_F16",
        ];
        if DIFFERS_ON_1426.contains(&name) {
            return Err("its #1426 semantics differ from the console's".into());
        }
        let operand = |input: bool, ty: &str| d.ienc.operands.iter().any(|o| o.input == input && o.output != input && o.ty == ty);
        let mut inputs: Vec<(Var, I::V)> = Vec::new();
        // SGPRs read whole: numbers (EXEC included, which holds a mask here: refused)
        for r in (0..128u8).filter(|&r| whole >> r & 1 != 0 && r != 125 && !(cmpx && r >= EXEC)) {
            let v = self.read_num(b, r)?;
            inputs.push((Var::S(r), v));
        }
        // a mask read lane by lane: a number with only this lane's bit
        let vop3 = d.enc.name.starts_with("ENC_VOP3") || d.enc.name.starts_with("VOP3");
        let lane_mask = if cmpx {
            Some(EXEC as u64)
        } else if matches!(name, "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32") {
            Some(if vop3 { rdna2::field_value(d.enc, &d.words, "SRC2").unwrap_or(0) } else { VCC as u64 })
        } else {
            None
        };
        if let Some(m) = lane_mask {
            if m <= 127 && m != 125 {
                let bit = self.src_bit(b, d, m)?;
                let lane = self.lane();
                let one = self.ir.zext(bit);
                let c31 = self.ir.int(31);
                let sh = self.ir.bin(Bin::And, lane, c31);
                let own = self.ir.bin(Bin::Shl, one, sh);
                if self.wave64 {
                    let (c32, z) = (self.ir.int(32), self.ir.int(0));
                    let upper = self.ir.icmp(Cmp::Uge, lane, c32);
                    let (lo, hi) = (self.ir.select(upper, z, own), self.ir.select(upper, own, z));
                    inputs.push((Var::S(m as u8), lo));
                    inputs.push((Var::S(m as u8 + 1), hi));
                } else {
                    // #1426 reads the mask as an SGPR pair in wave32 too: the upper one is 0
                    let z = self.ir.int(0);
                    inputs.push((Var::S(m as u8), own));
                    inputs.push((Var::S(m as u8 + 1), z));
                }
            }
        }
        if operand(true, "OPR_SSRC_SPECIAL_SCC") {
            let s = self.read_scc(b);
            let s = self.ir.zext(s);
            inputs.push((Var::Scc, s));
        }
        // VGPRs read, and the ones written (a write keeps the old value where the guard is off)
        let (vr, vw) = rdna2::vgpr_uses(d);
        let mut vregs: Vec<u16> = vr.iter().chain(&vw).map(|&v| v as u16).collect();
        vregs.sort_unstable();
        vregs.dedup();
        for &v in &vregs {
            let x = self.read(b, Var::V(v));
            inputs.push((Var::V(v), x));
        }
        // outputs: VGPRs, SGPRs (their current values seeded too, so an output the instruction
        // leaves alone reads back unchanged), SCC
        let sw: Vec<u8> = (0..128u8).filter(|&r| writes >> r & 1 != 0 && r != 125).collect();
        for &r in &sw {
            if !inputs.iter().any(|&(v, _)| v == Var::S(r)) {
                let x = self.read(b, Var::S(r));
                inputs.push((Var::S(r), x));
            }
        }
        let writes_scc = operand(false, "OPR_SSRC_SPECIAL_SCC");
        let mut outputs: Vec<Var> = vw.iter().map(|&v| Var::V(v as u16)).collect();
        outputs.extend(sw.iter().map(|&r| Var::S(r)));
        if writes_scc {
            outputs.push(Var::Scc);
        }
        let old_exec = if cmpx { Some(self.read_bit(b, EXEC)?) } else { None };
        let guard = if cmpx || in_guarded_run || !self.guarded.contains(&d.pc) { self.ir.boolean(true) } else { self.read_bit(b, EXEC)? };
        let vals = self.ir.foreign(d, guard, &inputs, &outputs)?;
        for (&var, &v) in outputs.iter().zip(&vals) {
            match var {
                Var::V(_) => self.ssa.write(b, var, v),
                Var::Scc => {
                    let z = self.ir.int(0);
                    let s = self.ir.icmp(Cmp::Ne, v, z);
                    self.write_scc(b, s);
                }
                Var::S(_) | Var::M(_) => {}
            }
        }
        if name.starts_with("V_") && !sw.is_empty() {
            // a mask (a compare, a carry-out): this lane's bit of the ballot the instruction made
            let base = sw[0];
            let lo = vals[outputs.iter().position(|&v| v == Var::S(base)).unwrap()];
            let word = match (self.wave64, outputs.iter().position(|&v| v == Var::S(base + 1))) {
                (true, Some(k)) => {
                    let lane = self.lane();
                    let c32 = self.ir.int(32);
                    let upper = self.ir.icmp(Cmp::Uge, lane, c32);
                    self.ir.select(upper, vals[k], lo)
                }
                _ => lo,
            };
            let lane = self.lane();
            let c31 = self.ir.int(31);
            let sh = self.ir.bin(Bin::And, lane, c31);
            let x = self.ir.bin(Bin::LShr, word, sh);
            let one = self.ir.int(1);
            let x = self.ir.bin(Bin::And, x, one);
            let z = self.ir.int(0);
            let bit = self.ir.icmp(Cmp::Ne, x, z);
            // `v_cmpx`: the new EXEC is the compare in the lanes that were in EXEC (#1426's
            // lowering ran with the guard `true`, so its result is the compare alone)
            let bit = match old_exec {
                Some(e) => self.ir.bin(Bin::And, bit, e),
                None => bit,
            };
            self.write_bit(b, base, bit);
        } else {
            for (&var, &v) in outputs.iter().zip(&vals) {
                if let Var::S(r) = var {
                    self.write_num(b, r, v);
                }
            }
        }
        Ok(())
    }

    /// Lowers one non-terminator instruction in block `b`, falling back to `bridge`.
    fn lower_inst(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        let r = if name.starts_with("S_LOAD_") || name.starts_with("S_BUFFER_LOAD_") {
            self.smem(b, d)
        } else if name.starts_with("S_") {
            self.salu(b, d)
        } else if name.starts_with("V_") {
            self.valu(b, d)
        } else if name.starts_with("GLOBAL_") || name.starts_with("FLAT_") {
            self.vmem(b, d)
        } else if name == "DS_SWIZZLE_B32" {
            self.swizzle(b, d)
        } else {
            Err(format!("unsupported {name}"))
        };
        self.or_bridge(r, b, d, self.unguarded)
    }

    /// `r`, or an instruction it calls unsupported lowered through `bridge`.
    fn or_bridge(&mut self, r: R<()>, b: usize, d: &Decoded, in_guarded_run: bool) -> R<()> {
        match r {
            Err(e) if e.starts_with("unsupported ") => match self.bridge(b, d, in_guarded_run) {
                Ok(()) => Ok(()),
                Err(f) if f == "no foreign lowering" => Err(e),
                Err(f) => Err(format!("{e} ({f})")),
            },
            r => r,
        }
    }

    fn note_inst(&mut self, r: R<()>, b: usize, d: &Decoded) -> R<()> {
        if r.is_err() && self.noted.is_some() {
            let (z, f) = (self.ir.int(0), self.ir.boolean(false));
            for s in rdna2::sgpr_uses(d).1 {
                if s < KERNEL_SGPRS as u64 && s != 125 {
                    self.ssa.write(b, Var::S(s as u8), z);
                    self.ssa.write(b, Var::M(s as u8), f);
                }
            }
            if d.inst.name.starts_with("S_") {
                self.ssa.write(b, Var::Scc, f);
            }
        }
        self.note(r)
    }

    /// A refusal: the error, or in survey mode noted and passed over.
    fn note(&mut self, r: R<()>) -> R<()> {
        match (r, &mut self.noted) {
            (Err(e), Some(n)) => {
                n.push(e);
                Ok(())
            }
            (r, _) => r,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The IR as a small program, run lane by lane on the CPU: what the lowering means,
    /// without a DXIL writer or a GPU.
    #[derive(Clone, Debug)]
    enum Ins {
        Int(u32),
        Bool(bool),
        Bin(Bin, u32, u32),
        Icmp(Cmp, u32, u32),
        Select(u32, u32, u32),
        Zext(u32),
        Lane,
        GroupId(u32),
        /// the flat thread index in the group: the wave's first thread + lane
        Flat,
        /// component k of a flat index `v` in a group whose x, y sizes are given
        ThreadId(u32, u32, [u32; 2]),
        Ballot(u32, usize),
        ReadLane(u32, u32),
        /// as `Ballot` and `ReadLane`, but the lowering may use only its own 32-lane half's part
        BallotHalf(u32, usize),
        ReadLaneHalf(u32, u32),
        CountBits(u32),
        IoLoad(u32),
        ResStore(u32, u32),
        MemLoad(u32),
        MemStore(u32, u32),
        Phi(Vec<(u32, usize)>),
        /// a VS / PS start: VGPR v from the lane's row of `io` (as in kernel mode), and the
        /// lane's bit of the launch EXEC from `io[126..=127]`
        GfxVgpr(u16),
        GfxExec,
    }

    #[derive(Clone, Debug)]
    enum Term {
        None,
        Br(usize),
        BrIf(u32, usize, usize),
        Ret,
    }

    #[derive(Default)]
    struct Interp {
        /// per block: (value id, instruction) in order, and the terminator
        blocks: Vec<(Vec<(u32, Ins)>, Term)>,
        cur: usize,
        next: u32,
        forward: HashMap<u32, u32>,
        undef: HashMap<bool, u32>,
        phi_block: HashMap<u32, usize>,
    }

    impl Interp {
        fn push(&mut self, ins: Ins) -> u32 {
            self.next += 1;
            let v = self.next;
            self.blocks[self.cur].0.push((v, ins));
            v
        }
        fn res(&self, mut v: u32) -> u32 {
            while let Some(&n) = self.forward.get(&v) {
                v = n;
            }
            v
        }
        fn selects(&self) -> usize {
            self.blocks.iter().map(|(ins, _)| ins.iter().filter(|(_, op)| matches!(op, Ins::Select(..))).count()).sum()
        }
        fn br_ifs(&self) -> usize {
            self.blocks.iter().filter(|(_, t)| matches!(t, Term::BrIf(..))).count()
        }
    }

    impl Sink for Interp {
        type V = u32;
        fn phi(&mut self, block: usize, _: Var) -> u32 {
            self.next += 1;
            let v = self.next;
            self.blocks[block].0.insert(0, (v, Ins::Phi(Vec::new())));
            self.phi_block.insert(v, block);
            v
        }
        fn incoming(&mut self, phi: u32, value: u32, pred: usize) {
            let b = self.phi_block[&phi];
            for (id, ins) in &mut self.blocks[b].0 {
                if *id == phi {
                    if let Ins::Phi(ops) = ins {
                        ops.push((value, pred));
                    }
                }
            }
        }
        fn replace(&mut self, old: u32, new: u32) {
            self.forward.insert(old, new);
            let b = self.phi_block[&old];
            self.blocks[b].0.retain(|(id, _)| *id != old);
        }
        fn undef(&mut self, var: Var) -> u32 {
            let bit = matches!(var, Var::M(_) | Var::Scc);
            if let Some(&v) = self.undef.get(&bit) {
                return v;
            }
            self.next += 1;
            let v = self.next;
            self.undef.insert(bit, v);
            v
        }
    }

    impl Ir for Interp {
        fn new_block(&mut self) -> usize {
            self.blocks.push((Vec::new(), Term::None));
            self.blocks.len() - 1
        }
        fn at(&mut self, block: usize) {
            self.cur = block;
        }
        fn int(&mut self, v: u32) -> u32 {
            self.push(Ins::Int(v))
        }
        fn boolean(&mut self, v: bool) -> u32 {
            self.push(Ins::Bool(v))
        }
        fn bin(&mut self, op: Bin, a: u32, b: u32) -> u32 {
            self.push(Ins::Bin(op, a, b))
        }
        fn icmp(&mut self, c: Cmp, a: u32, b: u32) -> u32 {
            self.push(Ins::Icmp(c, a, b))
        }
        fn select(&mut self, c: u32, a: u32, b: u32) -> u32 {
            self.push(Ins::Select(c, a, b))
        }
        fn zext(&mut self, b: u32) -> u32 {
            self.push(Ins::Zext(b))
        }
        fn lane(&mut self) -> u32 {
            self.push(Ins::Lane)
        }
        fn group_id(&mut self, k: u32) -> u32 {
            self.push(Ins::GroupId(k))
        }
        fn compute_ids(&mut self, threads: [u32; 3]) -> (u32, [u32; 3]) {
            let flat = self.push(Ins::Flat);
            let xy = [threads[0], threads[1]];
            (flat, [0, 1, 2].map(|k| self.push(Ins::ThreadId(k, flat, xy))))
        }
        fn ballot(&mut self, c: u32) -> [u32; 2] {
            [self.push(Ins::Ballot(c, 0)), self.push(Ins::Ballot(c, 1))]
        }
        fn ballot_half(&mut self, c: u32) -> [u32; 2] {
            [self.push(Ins::BallotHalf(c, 0)), self.push(Ins::BallotHalf(c, 1))]
        }
        fn read_lane(&mut self, v: u32, lane: u32) -> u32 {
            self.push(Ins::ReadLane(v, lane))
        }
        fn read_lane_half(&mut self, v: u32, lane: u32) -> u32 {
            self.push(Ins::ReadLaneHalf(v, lane))
        }
        fn count_bits(&mut self, v: u32) -> u32 {
            self.push(Ins::CountBits(v))
        }
        fn io_load(&mut self, dword: u32) -> u32 {
            self.push(Ins::IoLoad(dword))
        }
        fn res_store(&mut self, dword: u32, value: u32) {
            self.push(Ins::ResStore(dword, value));
        }
        fn mem_load(&mut self, addr: u32) -> u32 {
            self.push(Ins::MemLoad(addr))
        }
        fn mem_store(&mut self, addr: u32, value: u32) {
            self.push(Ins::MemStore(addr, value));
        }
        fn br(&mut self, target: usize) {
            self.blocks[self.cur].1 = Term::Br(target);
        }
        fn br_if(&mut self, c: u32, then: usize, otherwise: usize) {
            self.blocks[self.cur].1 = Term::BrIf(c, then, otherwise);
        }
        fn ret(&mut self) {
            self.blocks[self.cur].1 = Term::Ret;
        }
        fn graphics_start(&mut self, vregs: &[u16]) -> Option<GraphicsStart<u32>> {
            let vgprs = vregs.iter().map(|&v| (v, self.push(Ins::GfxVgpr(v)))).collect();
            let exec = self.push(Ins::GfxExec);
            Some(GraphicsStart { sgprs: Vec::new(), vgprs, exec })
        }
    }

    /// One lane's position: block, instruction index, the block it came from, its values.
    struct LaneState {
        block: usize,
        at: usize,
        from: usize,
        env: HashMap<u32, u32>,
        done: bool,
    }

    impl Interp {
        /// A value in a lane: what it computed, or 0 for `undef` (a marker of the lowering).
        fn value(&self, env: &HashMap<u32, u32>, v: u32) -> u32 {
            let v = self.res(v);
            if self.undef.values().any(|&u| u == v) {
                return 0;
            }
            *env.get(&v).unwrap_or_else(|| panic!("value {v} not defined"))
        }
    }

    impl Interp {
        /// Runs the program over `lanes` lanes; `io` in, `res` out. Ballots wait for every lane.
        fn run(&self, lanes: usize, io: &[u32], res: &mut [u32], mem: &mut [u32]) {
            self.run_with(lanes, 0, [0; 3], io, res, mem)
        }

        /// One wave of a compute group: its first thread `base`, the group `group`.
        fn run_with(&self, lanes: usize, base: u32, group: [u32; 3], io: &[u32], res: &mut [u32], mem: &mut [u32]) {
            let mut ls: Vec<LaneState> = (0..lanes).map(|_| LaneState { block: 0, at: 0, from: 0, env: HashMap::new(), done: false }).collect();
            let mut steps = 0usize;
            loop {
                // run every lane up to a ballot or its end
                for (lane, s) in ls.iter_mut().enumerate() {
                    while !s.done {
                        steps += 1;
                        assert!(steps < 10_000_000, "runaway program");
                        let (ins, term) = &self.blocks[s.block];
                        if s.at == 0 {
                            // phis read their operands from the edge taken, all at once
                            let vals: Vec<(u32, u32)> = ins
                                .iter()
                                .filter_map(|(id, i)| match i {
                                    Ins::Phi(ops) => {
                                        let (v, _) = ops.iter().find(|(_, p)| *p == s.from).unwrap_or_else(|| panic!("phi {id} in block {} has no operand from {}", s.block, s.from));
                                        Some((*id, self.value(&s.env, *v)))
                                    }
                                    _ => None,
                                })
                                .collect();
                            s.env.extend(vals);
                        }
                        if s.at < ins.len() {
                            let (id, i) = &ins[s.at];
                            let g = |v: &u32| self.value(&s.env, *v);
                            let r = match i {
                                Ins::Phi(_) => None,
                                Ins::Int(x) => Some(*x),
                                Ins::Bool(x) => Some(*x as u32),
                                Ins::Bin(op, a, b) => {
                                    let (a, b) = (g(a), g(b));
                                    Some(match op {
                                        Bin::Add => a.wrapping_add(b),
                                        Bin::Sub => a.wrapping_sub(b),
                                        Bin::Mul => a.wrapping_mul(b),
                                        Bin::And => a & b,
                                        Bin::Or => a | b,
                                        Bin::Xor => a ^ b,
                                        Bin::Shl => a << (b & 31),
                                        Bin::LShr => a >> (b & 31),
                                        Bin::AShr => ((a as i32) >> (b & 31)) as u32,
                                    })
                                }
                                Ins::Icmp(c, a, b) => {
                                    let (a, b) = (g(a), g(b));
                                    let (sa, sb) = (a as i32, b as i32);
                                    Some(match c {
                                        Cmp::Eq => a == b,
                                        Cmp::Ne => a != b,
                                        Cmp::Ult => a < b,
                                        Cmp::Ule => a <= b,
                                        Cmp::Ugt => a > b,
                                        Cmp::Uge => a >= b,
                                        Cmp::Slt => sa < sb,
                                        Cmp::Sle => sa <= sb,
                                        Cmp::Sgt => sa > sb,
                                        Cmp::Sge => sa >= sb,
                                    } as u32)
                                }
                                Ins::Select(c, a, b) => Some(if g(c) != 0 { g(a) } else { g(b) }),
                                Ins::Zext(b) => Some(g(b)),
                                Ins::Lane => Some(lane as u32),
                                Ins::GroupId(k) => Some(group[*k as usize]),
                                Ins::Flat => Some(base + lane as u32),
                                Ins::ThreadId(k, v, [x, y]) => Some(match k {
                                    0 => g(v) % x,
                                    1 => g(v) / x % y,
                                    _ => g(v) / (x * y),
                                }),
                                Ins::Ballot(..) | Ins::BallotHalf(..) | Ins::ReadLane(..) | Ins::ReadLaneHalf(..) => break,
                                Ins::CountBits(v) => Some(g(v).count_ones()),
                                Ins::IoLoad(a) => Some(io[g(a) as usize]),
                                Ins::MemLoad(a) => Some(mem.get((g(a) / 4) as usize).copied().unwrap_or(0)),
                                Ins::MemStore(a, v) => {
                                    let (a, v) = (g(a), g(v));
                                    if let Some(w) = mem.get_mut((a / 4) as usize) {
                                        *w = v;
                                    }
                                    None
                                }
                                Ins::ResStore(a, v) => {
                                    res[g(a) as usize] = g(v);
                                    None
                                }
                                Ins::GfxVgpr(v) => Some(io[KERNEL_SGPRS + lane * KERNEL_VGPRS + *v as usize]),
                                Ins::GfxExec => Some(io[EXEC as usize + lane / 32] >> (lane % 32) & 1),
                            };
                            if let Some(r) = r {
                                s.env.insert(*id, r);
                            }
                            s.at += 1;
                            continue;
                        }
                        match term {
                            Term::Br(t) => {
                                s.from = s.block;
                                s.block = *t;
                            }
                            Term::BrIf(c, t, f) => {
                                s.from = s.block;
                                s.block = if s.env[&self.res(*c)] != 0 { *t } else { *f };
                            }
                            Term::Ret => s.done = true,
                            Term::None => panic!("block {} has no terminator", s.block),
                        }
                        s.at = 0;
                    }
                }
                if ls.iter().all(|s| s.done) {
                    return;
                }
                // the lanes waiting at the first wave operation (in block order: a loop's body
                // before what follows it) run it together, as the lanes active there
                let (b, at) = ls.iter().filter(|s| !s.done).map(|s| (s.block, s.at)).min().unwrap();
                let group: Vec<usize> = (0..ls.len()).filter(|&l| !ls[l].done && (ls[l].block, ls[l].at) == (b, at)).collect();
                let (id, ins) = &self.blocks[b].0[at];
                let out: Vec<u32> = match ins {
                    Ins::Ballot(c, half) | Ins::BallotHalf(c, half) => {
                        let mut bits = 0u64;
                        for &l in &group {
                            if self.value(&ls[l].env, *c) != 0 {
                                bits |= 1 << l;
                            }
                        }
                        let word = (bits >> (32 * half)) as u32;
                        // the half-local form: the other half's word is garbage, so a lowering
                        // that reads it gets a wrong result here
                        let local = matches!(ins, Ins::BallotHalf(..));
                        group.iter().map(|&l| if local && l / 32 != *half { 0xA5A5_0000 | l as u32 } else { word }).collect()
                    }
                    Ins::ReadLane(v, from) | Ins::ReadLaneHalf(v, from) => group
                        .iter()
                        .map(|&l| {
                            let src = self.value(&ls[l].env, *from) as usize;
                            assert!(group.contains(&src), "lane {l} reads lane {src}, which is elsewhere");
                            if matches!(ins, Ins::ReadLaneHalf(..)) {
                                assert_eq!(src / 32, l / 32, "lane {l} reads lane {src} of the other half");
                            }
                            self.value(&ls[src].env, *v)
                        })
                        .collect(),
                    _ => unreachable!(),
                };
                for (&l, x) in group.iter().zip(out) {
                    ls[l].env.insert(*id, x);
                    ls[l].at += 1;
                }
            }
        }
    }

    pub(crate) type I = (&'static str, &'static str, &'static [(&'static str, u64)]);
    /// A GPU that runs `[WaveSize(32)]` programs, as the per-lane tests need (WARP, the CI
    /// adapter, has neither 32 nor 64: skipped, as `recomp::tests::gpu_ready` does).
    pub(crate) fn gpu_waves() -> bool {
        use super::super::runner;
        runner::adapter().is_ok() && runner::wave32_supported() == Ok(true)
    }

    pub(crate) const V: u64 = 256;
    pub(crate) const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]);

    // wave32 programs shared by the CPU tests and the comparison with the HLSL path
    pub(crate) const STRAIGHT: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 0)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)]),
        ENDPGM,
    ];
    /// v_cmp_gt_u32 vcc, 16, v0; s_and_saveexec s0, vcc; s_cbranch_execz L; v_mov v1, 7;
    /// L: s_or exec, exec, s0; v_add v2, v1, 1
    pub(crate) const SKIPPED_IF: &[I] = &[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 1)]),
        ENDPGM,
    ];
    /// s_mov s0, exec; L: v_add v1, v1, 1; v_cmp_ge_u32 vcc, v1, v0; exec &= ~vcc;
    /// s_cbranch_execnz L; exec |= s0  →  v1 = max(1, lane)
    pub(crate) const LEAVING_LOOP: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 0)]),
        ("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
        ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ENDPGM,
    ];
    /// s1 = 0; L: s1 += 3; v1 += 1; s_cmp_lt_u32 s1, 12; s_cbranch_scc1 L
    pub(crate) const UNIFORM_LOOP: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 128)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 131)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 140)]),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
        ENDPGM,
    ];

    /// v1 = 4·lane; s[4:5] = mem[2..4]; v2 = mem[lane] + s4; then lanes below 16 store v2 at
    /// mem[64 + lane] (a guarded store: no skip branch), and every lane stores v0 at mem[128 + lane]
    pub(crate) const MEMORY: &[I] = &[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 8), ("SOFFSET", 125)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 1), ("SADDR", 125), ("OFFSET", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 4), ("VSRC1", 2)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 2), ("SADDR", 125), ("OFFSET", 0x100)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 6)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 0), ("SADDR", 125), ("OFFSET", 0x200)]),
        ENDPGM,
    ];

    /// A compute program: 64 threads, user SGPRs s0–s1, the group id x in s2;
    /// mem[64·group + thread] = 3·thread + s0 + group
    pub(crate) const COMPUTE: &[I] = &[
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 192)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 3), ("VSRC1", 0)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 1), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 2), ("VSRC1", 1)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 2), ("DATA", 1), ("SADDR", 125), ("OFFSET", 0)]),
        ENDPGM,
    ];

    // f32 programs: the per-lane form refuses f32 so far, and `lower_dxil`'s
    // `f32_programs_on_the_gpu` runs them on the HLSL path, checks its registers with the
    // `check_f32_*` below (32 lanes) and compares the per-lane form with it where it lowers them

    /// f32 VOP2 / VOP3 arithmetic on v1 = 2^lane and v2 = 2^lane · (1 + lane / 64), with the
    /// VOP3 modifiers, and v19 = NaN (0x7FFFFFFF) into min / max / clamp. Every product is exact,
    /// so `FMad` gives the same fused or not.
    pub(crate) const F32_ARITHMETIC: &[I] = &[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 192), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 191), ("VSRC1", 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 151), ("VSRC1", 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 145), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        // 0.5 + x, x - p, x - (-4), x · x, min / max(2, x)
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 240), ("VSRC1", 2)]),
        ("V_SUB_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 2), ("VSRC1", 1)]),
        ("V_SUBREV_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 247), ("VSRC1", 2)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 2), ("VSRC1", 2)]),
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 244), ("VSRC1", 2)]),
        ("V_MAX_F32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 244), ("VSRC1", 2)]),
        // fma(x, -2, p), mad(x, 0.5, v3); mac v11 = 4 · x + p, fmac v12 = -0.5 · x + v3
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 2), ("SRC1", 245), ("SRC2", V + 1)]),
        ("V_MAD_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 240), ("SRC2", V + 3)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 1)]),
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 246), ("VSRC1", 2)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 3)]),
        ("V_FMAC_F32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 241), ("VSRC1", 2)]),
        // -|x| · |-4| (abs before neg); omod 0.5, 2, 4 (with neg on src2); clamp at 1 and at 0
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 2), ("SRC1", 247), ("ABS", 3), ("NEG", 1)]),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", V + 1), ("OMOD", 3)]),
        ("V_SUB_F32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 1), ("SRC1", V + 2), ("OMOD", 1)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 1), ("SRC2", V + 2), ("NEG", 4), ("OMOD", 2)]),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 4), ("SRC1", 240), ("CLAMP", 1)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 2), ("SRC1", 243), ("SRC2", 244), ("CLAMP", 1)]),
        // min(NaN, x), max(x, -NaN), clamp(NaN · 1); VOP3 fmac v23 = |-0.5| · x + p
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 19), ("VSRC1", 2)]),
        ("V_MAX_F32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 2), ("SRC1", V + 19), ("NEG", 2)]),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 19), ("SRC1", 242), ("CLAMP", 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 1)]),
        ("V_FMAC_F32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 241), ("SRC1", V + 2), ("ABS", 1)]),
        ENDPGM,
    ];

    /// f32 compares of a = v3 (NaN in every fourth lane, 2^lane in the others) with 4.0: all 16
    /// predicates into s[2k], two with VOP3 abs / neg, then VOPC into VCC and a cndmask of it,
    /// and v_cmpx narrowing EXEC for a v_mov.
    pub(crate) const F32_COMPARES: &[I] = &[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 192), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 191), ("VSRC1", 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 151), ("VSRC1", 1)]),
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 131), ("VSRC1", 0)]),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 4)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 19)]),
        ("V_CMP_F_F32", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_LT_F32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_EQ_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_LE_F32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_GT_F32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_LG_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_GE_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_O_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_U_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NGE_F32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NLG_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NGT_F32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NLE_F32", "ENC_VOP3", &[("VDST", 24), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NEQ_F32", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NLT_F32", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_TRU_F32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 3), ("SRC1", 246)]),
        // -|a| == -4; !(|-2| > -a)
        ("V_CMP_EQ_F32", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 3), ("SRC1", 247), ("ABS", 1), ("NEG", 1)]),
        ("V_CMP_NGT_F32", "ENC_VOP3", &[("VDST", 34), ("SRC0", 245), ("SRC1", V + 3), ("ABS", 1), ("NEG", 2)]),
        // vcc = a < 4, v6 = vcc ? p : 0; exec = a >= 4, v7 = 1.0 there
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 246)]),
        ("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 5)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 1)]),
        ("V_CMPX_GE_F32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 5)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)]),
        ENDPGM,
    ];

    /// f32 ↔ i32 / u32 / f16 conversions: v2 = lane − 16, v4 = (lane << 26 | lane) rounded,
    /// v6 = (lane − 16) / 4 into the f32 → integer ones, v11 = v4² (out of range) and v19 = NaN
    /// into the saturating ones; binary16 in and out (denormals, ties to even, overflow, NaN),
    /// with OP_SEL halves, omod and clamp.
    pub(crate) const F32_CONVERSIONS: &[I] = &[
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 154), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 3)]),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)]),
        ("V_CVT_F32_I32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 1), ("OMOD", 3), ("CLAMP", 1)]),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 2), ("SRC1", 240), ("OMOD", 3)]),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 6)]),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 6)]),
        ("V_CVT_RPI_I32_F32", "ENC_VOP1", &[("VDST", 9), ("SRC0", V + 6)]),
        ("V_CVT_FLR_I32_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 6)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 4), ("VSRC1", 4)]),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 11)]),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 11)]),
        ("V_CVT_I32_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 11), ("NEG", 1)]),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 19)]),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 19)]),
        // v17 = f16 (lane & 31) << 9 | (lane & 31), the lane in the high half; v20 = it in the
        // high half, the lane in the low one
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 23), ("SRC0", 159), ("VSRC1", 0)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 17), ("SRC0", 137), ("VSRC1", 23)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 17), ("SRC0", V + 23), ("VSRC1", 17)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 144), ("VSRC1", 17)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V), ("VSRC1", 20)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 18), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 17), ("SRC0", V + 18), ("VSRC1", 17)]),
        ("V_CVT_F32_F16", "ENC_VOP1", &[("VDST", 21), ("SRC0", V + 17)]),
        ("V_CVT_F32_F16", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 20), ("OP_SEL", 1)]),
        // v24 = 1 + (lane & 31) · 2^-14: f16 ties at lanes 8 and 24; into v25's low half
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 24), ("SRC0", 137), ("VSRC1", 23)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 24), ("SRC0", 255), ("VSRC1", 24), ("LITERAL", 0x3F80_0000)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", V + 20)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 25), ("SRC0", V + 24)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", V)]),
        ("V_CVT_F16_F32", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 6), ("OP_SEL", 8)]),
        // f16 denormals (· 2^-20), below the smallest one (· 2^-25), NaN, overflow, clamp
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 28), ("SRC0", 255), ("VSRC1", 24), ("LITERAL", 0x3580_0000)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 27), ("SRC0", V + 28)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 28), ("SRC0", 255), ("VSRC1", 24), ("LITERAL", 0x3300_0000)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 29), ("SRC0", V + 28)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 19)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 31), ("SRC0", V + 11)]),
        ("V_CVT_F16_F32", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 6), ("CLAMP", 1)]),
        ENDPGM,
    ];

    /// floor / ceil / trunc / rndne / fract of v3 = (lane − 16) / 8 (ties, signed zeros), fract
    /// of −2^-30 (just below 1.0), and two with VOP3 modifiers.
    pub(crate) const F32_ROUNDING: &[I] = &[
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 2), ("LITERAL", 0x3E00_0000)]),
        ("V_FLOOR_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)]),
        ("V_CEIL_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 3)]),
        ("V_TRUNC_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 3)]),
        ("V_RNDNE_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 3)]),
        ("V_FRACT_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 3)]),
        ("V_FRACT_F32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 255), ("LITERAL", 0xB080_0000)]),
        ("V_FLOOR_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("NEG", 1), ("OMOD", 1)]),
        ("V_FRACT_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 3), ("ABS", 1), ("CLAMP", 1)]),
        ENDPGM,
    ];

    /// rcp / rsq / sqrt / log of v1 = 1 + (lane & 31) / 32, exp / sin / cos of v2 = v1 − 1.5,
    /// and three with VOP3 modifiers (log2 |0| = −∞ at lane 16). The GPU's are approximate.
    pub(crate) const F32_TRANSCENDENTALS: &[I] = &[
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 23), ("SRC0", 159), ("VSRC1", 0)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 146), ("VSRC1", 23)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 255), ("VSRC1", 1), ("LITERAL", 0x3F80_0000)]),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 1), ("SRC1", 243)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 241), ("VSRC1", 2)]),
        ("V_RCP_F32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 1)]),
        ("V_RCP_IFLAG_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 1)]),
        ("V_RSQ_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 1)]),
        ("V_SQRT_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)]),
        ("V_EXP_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 2)]),
        ("V_LOG_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 1)]),
        ("V_SIN_F32", "ENC_VOP1", &[("VDST", 9), ("SRC0", V + 2)]),
        ("V_COS_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 2)]),
        ("V_EXP_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 2), ("NEG", 1), ("OMOD", 1)]),
        ("V_SIN_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("CLAMP", 1)]),
        ("V_LOG_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 2), ("ABS", 1)]),
        ENDPGM,
    ];

    /// v_fmamk / fmaak / madmk / madak (products exact) and min3 / max3 / med3 of a = the lane
    /// (NaN in every fourth lane), b = lane − 16 and 4.0, one with VOP3 abs / neg.
    pub(crate) const F32_THREE_SOURCE: &[I] = &[
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)]),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V)]),
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 131), ("VSRC1", 0)]),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 4)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 3), ("VSRC1", 19)]),
        ("V_FMAMK_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 3), ("VSRC1", 2), ("LITERAL", 0x3FC0_0000)]),
        ("V_FMAAK_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 3), ("VSRC1", 2), ("LITERAL", 0x3E80_0000)]),
        ("V_MADMK_F32", "ENC_VOP2", &[("VDST", 8), ("SRC0", V + 3), ("VSRC1", 2), ("LITERAL", 0x3FC0_0000)]),
        ("V_MADAK_F32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 3), ("VSRC1", 2), ("LITERAL", 0x3E80_0000)]),
        ("V_MIN3_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 5), ("SRC1", V + 2), ("SRC2", 246)]),
        ("V_MAX3_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 5), ("SRC1", V + 2), ("SRC2", 246)]),
        ("V_MED3_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 5), ("SRC1", V + 2), ("SRC2", 246)]),
        ("V_MED3_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 2), ("SRC1", V + 5), ("SRC2", 246), ("ABS", 1), ("NEG", 2)]),
        ENDPGM,
    ];

    // f16 and packed programs, as the f32 ones: `lower_dxil`'s `f16_programs_on_the_gpu` checks
    // the HLSL path's registers with the `check_f16_*` below and compares the per-lane form with it

    /// f16 VOP1 / VOP2 / VOP3 on x = 1 + lane · 2^-10 (v1, and v6's low half; v6's high half is
    /// −2 − lane · 2^-9): ties to even, denormal results and their ties, overflow, a fused fma
    /// just below a tie (lane 1), NaN into min / max / med3, OP_SEL halves, omod and clamp, and
    /// f16 ↔ i16 / u16.
    pub(crate) const F16_ARITHMETIC: &[I] = &[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0x3C00)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0xC000)]),
        ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 6), ("SRC1", 144), ("SRC2", V + 1)]),
        // v2 = 2^-11, v3 = 2^-15 (a denormal), v4 = 65504, v5 = NaN, v7 = 2^-11 · (1 − 2^-10)
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 255), ("LITERAL", 0x1000)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 255), ("LITERAL", 0x0200)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 255), ("LITERAL", 0x7BFF)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 255), ("LITERAL", 0x7E00)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 255), ("LITERAL", 0x0FFE)]),
        // x + 2^-11, 2^-11 − x (ties), x · 2^-15 (denormal ties), x · 65504 (∞ from lane 1),
        // fma(x, v7, x)
        ("V_ADD_F16", "ENC_VOP2", &[("VDST", 8), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_SUB_F16", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 2), ("VSRC1", 1)]),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 1), ("VSRC1", 3)]),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 1), ("VSRC1", 4)]),
        ("V_FMA_F16", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 7), ("SRC2", V + 1)]),
        // min(NaN, x), max(x, −NaN); v15's high half = v6's + x, its low half kept
        ("V_MIN_F16", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 5), ("VSRC1", 1)]),
        ("V_MAX_F16", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 1), ("SRC1", V + 5), ("NEG", 2)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V)]),
        ("V_ADD_F16", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 6), ("SRC1", V + 1), ("OP_SEL", 9)]),
        // omod: (x · 0.5) · 2, (x · x − 1) · 4; clamp of 65504 · 2
        ("V_MUL_F16", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", 240), ("OMOD", 1)]),
        ("V_FMA_F16", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 1), ("SRC1", V + 1), ("SRC2", 243), ("OMOD", 2)]),
        ("V_MUL_F16", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 4), ("SRC1", 244), ("CLAMP", 1)]),
        // ldexp(x, −lane): denormals and their ties from lane 15, 0 from lane 26
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 128), ("VSRC1", 0)]),
        ("V_LDEXP_F16", "ENC_VOP2", &[("VDST", 19), ("SRC0", V + 1), ("VSRC1", 20)]),
        // fract of v6's high half; q = (lane − 16) / 2 into rndne (ties, −0 at lane 15), i16, u16;
        // u16 2049 · lane into f16 (a tie at lane 16)
        ("V_FRACT_F16", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 6), ("OP_SEL", 1)]),
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 25), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F16_I16", "ENC_VOP1", &[("VDST", 24), ("SRC0", V + 25)]),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 26), ("SRC0", 240), ("VSRC1", 24)]),
        ("V_RNDNE_F16", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 26)]),
        ("V_CVT_I16_F16", "ENC_VOP1", &[("VDST", 27), ("SRC0", V + 26)]),
        ("V_CVT_U16_F16", "ENC_VOP1", &[("VDST", 28), ("SRC0", V + 26)]),
        ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V), ("SRC1", 139), ("SRC2", V)]),
        ("V_CVT_F16_U16", "ENC_VOP1", &[("VDST", 29), ("SRC0", V + 30)]),
        // med3(NaN, x, 2); fmac v32 = x · 2^-11 + x
        ("V_MED3_F16", "ENC_VOP3", &[("VDST", 31), ("SRC0", V + 5), ("SRC1", V + 1), ("SRC2", 244)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 32), ("SRC0", V + 1)]),
        ("V_FMAC_F16", "ENC_VOP2", &[("VDST", 32), ("SRC0", V + 1), ("VSRC1", 2)]),
        ENDPGM,
    ];

    /// VOP3P packed math on v1 = x | y << 16 (x, y as in `F16_ARITHMETIC`), v2 = 2^-11 | 2^-15 << 16,
    /// v3 = NaN | 65504 << 16 and the u16 / i16 pairs v4 = 2049 · lane | (0xFFFF − lane) << 16,
    /// v5 = (0x8000 + lane << 10) | (0x7FF0 + lane) << 16: OP_SEL / OP_SEL_HI picking swapped and
    /// broadcast halves, NEG / NEG_HI, clamp (f16, and saturating u16 / i16), and
    /// v_cvt_pkrtz_f16_f32 (toward zero, 65504 for the out-of-range ones).
    pub(crate) const F16_PACKED: &[I] = &[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0x3C00)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0xC000)]),
        ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V + 9), ("SRC1", 144), ("SRC2", V + 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 255), ("LITERAL", 0x0200_1000)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 255), ("LITERAL", 0x7BFF_7E00)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 255), ("LITERAL", 0x0FFE_0FFE)]),
        ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V), ("SRC1", 139), ("SRC2", V)]),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0xFFFF)]),
        ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 11), ("SRC1", 144), ("SRC2", V + 10)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 138), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 255), ("VSRC1", 12), ("LITERAL", 0x8000)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0x7FF0)]),
        ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 13), ("SRC1", 144), ("SRC2", V + 12)]),
        // v1 + v2; v1.hi · v2.lo | v1.lo · v2.hi; fma(v1, v7, v1) (v7 as in `F16_ARITHMETIC`)
        ("V_PK_ADD_F16", "ENC_VOP3P", &[("VDST", 14), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)]),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 15), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL", 1), ("OP_SEL_HI", 2)]),
        ("V_PK_FMA_F16", "ENC_VOP3P", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 7), ("SRC2", V + 1), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)]),
        // min(v3, v1), max(v1, −v3), v1 + (−lo | hi of v1's high half), clamp(v3.hi · v1)
        ("V_PK_MIN_F16", "ENC_VOP3P", &[("VDST", 17), ("SRC0", V + 3), ("SRC1", V + 1), ("OP_SEL_HI", 3)]),
        ("V_PK_MAX_F16", "ENC_VOP3P", &[("VDST", 18), ("SRC0", V + 1), ("SRC1", V + 3), ("OP_SEL_HI", 3), ("NEG", 2), ("NEG_HI", 2)]),
        ("V_PK_ADD_F16", "ENC_VOP3P", &[("VDST", 19), ("SRC0", V + 1), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("NEG_HI", 1)]),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 20), ("SRC0", V + 3), ("SRC1", V + 1), ("OP_SEL", 1), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        // u16 / i16: wrapping, saturating (clamp), v4.lo · v5.hi | v4.hi · v5.hi, v4 · v4 + v5
        ("V_PK_ADD_U16", "ENC_VOP3P", &[("VDST", 21), ("SRC0", V + 4), ("SRC1", V + 5), ("OP_SEL_HI", 3)]),
        ("V_PK_ADD_U16", "ENC_VOP3P", &[("VDST", 22), ("SRC0", V + 4), ("SRC1", V + 5), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        ("V_PK_SUB_U16", "ENC_VOP3P", &[("VDST", 23), ("SRC0", V + 4), ("SRC1", V + 5), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        ("V_PK_ADD_I16", "ENC_VOP3P", &[("VDST", 24), ("SRC0", V + 5), ("SRC1", V + 5), ("OP_SEL_HI", 3), ("CLAMP", 1)]),
        ("V_PK_SUB_I16", "ENC_VOP3P", &[("VDST", 25), ("SRC0", V + 4), ("SRC1", V + 5), ("OP_SEL_HI", 3)]),
        ("V_PK_MUL_LO_U16", "ENC_VOP3P", &[("VDST", 26), ("SRC0", V + 4), ("SRC1", V + 5), ("OP_SEL", 2), ("OP_SEL_HI", 3)]),
        ("V_PK_MAD_U16", "ENC_VOP3P", &[("VDST", 27), ("SRC0", V + 4), ("SRC1", V + 4), ("SRC2", V + 5), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)]),
        // pkrtz of 1 + lane · 2^-11 and of 65520 + 16 · lane (f32)
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 140), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 29), ("SRC0", 255), ("VSRC1", 31), ("LITERAL", 0x3F80_0000)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 255), ("VSRC1", 31), ("LITERAL", 0x477F_F000)]),
        ("V_CVT_PKRTZ_F16_F32", "ENC_VOP2", &[("VDST", 28), ("SRC0", V + 29), ("VSRC1", 30)]),
        ENDPGM,
    ];

    /// binary16 of an f64 (exact: an f16 sum or product), rounded to nearest even: denormals,
    /// and ∞ past 65504
    pub(crate) fn f16_bits(x: f64) -> u32 {
        let s = (x.is_sign_negative() as u32) << 15;
        let a = x.abs();
        if a.is_infinite() {
            return s | 0x7C00;
        }
        let e = ((a.to_bits() >> 52) as i32 - 1023).max(-14);
        let q = (a * 2f64.powi(10 - e)).round_ties_even() as u32;
        s | ((e + 14) as u32 * 1024 + q).min(0x7C00)
    }

    pub(crate) fn check_f16_arithmetic(_: &[u32], v: &[Vec<u32>]) {
        let h = |b: u32| f16_to_f32(b & 0xFFFF) as f64;
        let p = |e: i32| 2f64.powi(e);
        for (l, r) in v.iter().enumerate() {
            let (x, y) = (0x3C00 + l as u32, 0xC000 + l as u32);
            let (fx, fy, q) = (h(x), h(y), (l as f64 - 16.0) / 2.0);
            assert_eq!((r[1], r[6]), (x, y << 16 | x), "lane {l}");
            // v8, v9, v12, v17, v19, v29 and v32 round to nearest even on the console (MODE.FP_ROUND
            // = 0) and the HLSL path gives them rounded toward zero on NVIDIA (`f32tof16`, whose
            // rounding the driver picks), so they are left to the per-lane comparison
            let want = [
                (10, f16_bits(fx * p(-15))),
                (11, f16_bits(fx * 65504.0)),
                (13, x),
                (14, x),
                (15, f16_bits(fy + fx) << 16 | l as u32),
                (16, x),
                (18, 0x3C00),
                (22, f16_bits(fy - fy.floor())),
                (23, f16_bits(q.round_ties_even())),
                (24, f16_bits(l as f64 - 16.0)),
                (26, f16_bits(q)),
                (27, q as i32 as u32 & 0xFFFF),
                (28, q as u32),
                (31, x),
            ];
            for (i, w) in want {
                assert_eq!(r[i], w, "lane {l} v{i}: {:#x} against {w:#x}", r[i]);
            }
        }
    }

    pub(crate) fn check_f16_packed(_: &[u32], v: &[Vec<u32>]) {
        let h = |b: u32| f16_to_f32(b & 0xFFFF) as f64;
        let p = |e: i32| 2f64.powi(e);
        let pk = |lo: u32, hi: u32| lo & 0xFFFF | hi << 16;
        let i16s = |a: u32| a as u16 as i16 as i32;
        let sat = |a: i32| a.clamp(-32768, 32767) as u32;
        for (l, r) in v.iter().enumerate() {
            let n = l as u32;
            let (x, y) = (0x3C00 + n, 0xC000 + n);
            let (fx, fy) = (h(x), h(y));
            let (a0, a1, c0, c1) = (2049 * n, 0xFFFF - n, 0x8000 + (n << 10), 0x7FF0 + n);
            assert_eq!((r[1], r[4], r[5]), (pk(x, y), pk(a0, a1), pk(c0, c1)), "lane {l}");
            // v14 and v16 as v8 and v12 of `check_f16_arithmetic`
            let want = [
                (15, pk(f16_bits(fy * p(-11)), f16_bits(fx * p(-15)))),
                (17, pk(x, y)),
                (18, pk(x, y)),
                (19, pk(f16_bits(2.0 * fx), 0)),
                (20, pk(0x3C00, 0)),
                (21, pk(a0 + c0, a1 + c1)),
                (22, pk((a0 + c0).min(0xFFFF), (a1 + c1).min(0xFFFF))),
                (23, pk(a0.saturating_sub(c0), a1.saturating_sub(c1))),
                (24, pk(sat(2 * i16s(c0)), sat(2 * i16s(c1)))),
                (25, pk(a0.wrapping_sub(c0), a1.wrapping_sub(c1))),
                (26, pk(a0 * c1, a1 * c1)),
                (27, pk(a0 * a0 + c0, a1 * a1 + c1)),
                (28, pk(0x3C00 + n / 2, 0x7BFF)),
            ];
            for (i, w) in want {
                assert_eq!(r[i], w, "lane {l} v{i}: {:#x} against {w:#x}", r[i]);
            }
        }
    }

    pub(crate) fn check_f32_arithmetic(_: &[u32], v: &[Vec<u32>]) {
        let sat = |x: f32| x.clamp(0.0, 1.0);
        for (l, r) in v.iter().enumerate() {
            let f = |i: usize| f32::from_bits(r[i]);
            let (p, x) = (f(1), f(2));
            assert_eq!((p, x), (2f32.powi(l as i32), 2f32.powi(l as i32) * (1.0 + l as f32 / 64.0)), "lane {l}");
            let y = x + 0.5;
            let want = [y, x - p, x + 4.0, x * x, x.min(2.0), x.max(2.0), x * -2.0 + p, x * 0.5 + y, 4.0 * x + p, -0.5 * x + y, -x * 4.0, (x + p) * 0.5, (p - x) * 2.0, (p * p - x) * 4.0, sat((x - p) * 0.5), sat(2.0 - x)];
            for (i, w) in want.iter().enumerate() {
                assert_eq!(f(3 + i).to_bits(), w.to_bits(), "lane {l} v{}: {} against {w}", 3 + i, f(3 + i));
            }
            assert_eq!((r[20], r[21], r[22]), (r[2], r[2], 0), "lane {l}: NaN into min, max, clamp");
            assert_eq!(f(23), 0.5 * x + p, "lane {l}: VOP3 fmac");
        }
    }

    pub(crate) fn check_f32_compares(s: &[u32], v: &[Vec<u32>]) {
        let preds: [fn(f32, f32) -> bool; 18] = [
            |_, _| false,
            |a, b| a < b,
            |a, b| a == b,
            |a, b| a <= b,
            |a, b| a > b,
            |a, b| a < b || a > b,
            |a, b| a >= b,
            |a, b| !a.is_nan() && !b.is_nan(),
            |a, b| a.is_nan() || b.is_nan(),
            |a, b| !(a >= b),
            |a, b| !(a < b || a > b),
            |a, b| !(a > b),
            |a, b| !(a <= b),
            |a, b| !(a == b),
            |a, b| !(a < b),
            |_, _| true,
            |a, _| -a.abs() == -4.0,
            |a, _| !(2.0 > -a),
        ];
        let a: Vec<f32> = (0..32).map(|l| if l % 4 == 0 { f32::from_bits(0x7FFF_FFFF) } else { 2f32.powi(l) }).collect();
        let mask = |p: &dyn Fn(f32) -> bool| (0..32).filter(|&l| p(a[l])).fold(0u32, |m, l| m | 1 << l);
        for (k, p) in preds.iter().enumerate() {
            assert_eq!(s[2 * k], mask(&|x| p(x, 4.0)), "s{}", 2 * k);
        }
        assert_eq!(s[106], mask(&|x| x < 4.0), "vcc");
        assert_eq!(s[126], mask(&|x| x >= 4.0), "exec");
        for (l, r) in v.iter().enumerate() {
            assert_eq!(r[3], a[l].to_bits(), "lane {l}");
            assert_eq!(r[6], if a[l] < 4.0 { r[1] } else { 0 }, "lane {l}: cndmask");
            assert_eq!(r[7], if a[l] >= 4.0 { 1f32.to_bits() } else { 0 }, "lane {l}: under v_cmpx");
        }
    }

    pub(crate) fn check_f32_conversions(_: &[u32], v: &[Vec<u32>]) {
        for (l, r) in v.iter().enumerate() {
            let f = |i: usize| f32::from_bits(r[i]);
            let (li, q) = (l as i32 - 16, (l as i32 - 16) as f32 / 4.0);
            assert_eq!(f(2), li as f32, "lane {l}");
            assert_eq!(f(4), ((l as u32) << 26 | l as u32) as f32, "lane {l}: u32 rounded to nearest even");
            assert_eq!(f(5), (li as f32 * 0.5).clamp(0.0, 1.0), "lane {l}: omod and clamp");
            assert_eq!(f(6), q, "lane {l}");
            // the HLSL path's cvt_i32 / cvt_u32 saturate as Rust's `as` does, NaN giving 0
            assert_eq!((r[7], r[8]), (q as i32 as u32, q as u32), "lane {l}: truncated");
            assert_eq!((r[9], r[10]), ((q + 0.5).floor() as i32 as u32, q.floor() as i32 as u32), "lane {l}: rpi, flr");
            assert_eq!((r[12], r[13], r[14]), (f(11) as i32 as u32, f(11) as u32, (-f(11)) as i32 as u32), "lane {l}: out of range");
            assert_eq!((r[15], r[16]), (0, 0), "lane {l}: NaN");
            assert_eq!(f(21), f16_to_f32(r[17] & 0xFFFF), "lane {l}");
            assert_eq!(r[22], r[21], "lane {l}: the high half");
            assert_eq!(r[25] >> 16, r[20] >> 16, "lane {l}: the half kept");
            assert_eq!(r[26] & 0xFFFF, l as u32, "lane {l}: the half kept");
            assert_eq!(r[29], (l != 0) as u32, "lane {l}: below the smallest denormal");
            assert_eq!(r[30], 0x7FFF, "lane {l}: NaN");
            assert_eq!(r[31], if l == 0 { 0 } else { 0x7C00 }, "lane {l}: overflow");
        }
        let lo = |l: usize, i: usize| v[l][i] & 0xFFFF;
        assert_eq!([lo(0, 21), lo(1, 17), lo(30, 17)], [0, 0x0201, 0x3C1E]);
        assert_eq!((f32::from_bits(v[1][21]), f32::from_bits(v[30][21])), (513.0 * 2f32.powi(-24), 1.0 + 30.0 / 1024.0), "f16 in");
        // 1 + k · 2^-14 at k = 1, 8, 16, 24, 31: ties to even at 8 and 24
        assert_eq!([1, 8, 16, 24, 31].map(|l| lo(l, 25)), [0x3C00, 0x3C00, 0x3C01, 0x3C02, 0x3C02]);
        assert_eq!([0, 16, 18, 20].map(|l| v[l][26] >> 16), [0xC400, 0, 0x3800, 0x3C00], "f16 into the high half");
        assert_eq!([0, 18, 20, 31].map(|l| lo(l, 32)), [0, 0x3800, 0x3C00, 0x3C00], "clamp");
        assert_eq!(lo(0, 27), 0x0010, "2^-20: an f16 denormal");
    }

    pub(crate) fn check_f32_rounding(_: &[u32], v: &[Vec<u32>]) {
        for (l, r) in v.iter().enumerate() {
            let x = (l as i32 - 16) as f32 / 8.0;
            let want = [x.floor(), x.ceil(), x.trunc(), x.round_ties_even(), x - x.floor()];
            for (i, w) in want.iter().enumerate() {
                assert_eq!(r[4 + i], w.to_bits(), "lane {l} v{}", 4 + i);
            }
            assert_eq!(r[9], 0x3F7F_FFFF, "fract(-2^-30): below 1.0");
            assert_eq!(r[10], (2.0 * (-x).floor()).to_bits(), "lane {l} v10");
            assert_eq!(f32::from_bits(r[11]), x.abs() - x.abs().floor(), "lane {l}");
        }
    }

    pub(crate) fn check_f32_three_source(_: &[u32], v: &[Vec<u32>]) {
        let med3 = |a: f32, b: f32, c: f32| a.min(b).max(a.max(b).min(c));
        for (l, r) in v.iter().enumerate() {
            let f = |i: usize| f32::from_bits(r[i]);
            let (x, y) = (l as f32, l as f32 - 16.0);
            let a = if l % 4 == 0 { f32::from_bits(0x7FFF_FFFF) } else { x };
            assert_eq!([f(6), f(7), f(8), f(9)], [x * 1.5 + y, x * y + 0.25, x * 1.5 + y, x * y + 0.25], "lane {l}");
            assert_eq!([f(10), f(11), f(12)], [a.min(y).min(4.0), a.max(y).max(4.0), med3(a, y, 4.0)], "lane {l}");
            assert_eq!(f(13), med3(y.abs(), -a, 4.0), "lane {l}");
        }
    }

    pub(crate) fn check_f32_transcendentals(_: &[u32], v: &[Vec<u32>]) {
        let sat = |x: f32| x.clamp(0.0, 1.0);
        for (l, r) in v.iter().enumerate() {
            let x = 1.0 + (l & 31) as f32 / 32.0;
            let y = x - 1.5;
            let t = y * 6.283_185_3;
            assert_eq!((r[1], r[2]), (x.to_bits(), y.to_bits()), "lane {l}");
            let want = [1.0 / x, 1.0 / x, 1.0 / x.sqrt(), x.sqrt(), y.exp2(), x.log2(), t.sin(), t.cos(), 2.0 * (-y).exp2(), sat(t.sin()), y.abs().log2()];
            for (i, w) in want.iter().enumerate() {
                assert!(f32_close(r[3 + i], w.to_bits()), "lane {l} v{}: {} against {w}", 3 + i, f32::from_bits(r[3 + i]));
            }
        }
    }

    /// The same bits, or within `approximate_float_unary_ops_are_within_tolerance`'s tolerance.
    pub(crate) fn f32_close(a: u32, b: u32) -> bool {
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        a.to_bits() == b.to_bits() || (a - b).abs() <= 1e-3f32.max(b.abs() * 1e-5)
    }

    /// binary16 in the low 16 bits → f32 (`f16tof32`)
    pub(crate) fn f16_to_f32(h: u32) -> f32 {
        let (s, e, m) = (h >> 15 & 1, h >> 10 & 0x1F, h & 0x3FF);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 => f32::from_bits(0x7F80_0000 | m << 13),
            _ => f32::from_bits((e + 112) << 23 | m << 13),
        };
        if s == 1 { -v } else { v }
    }

    // integer programs, as the f32 ones: `lower_dxil`'s `integer_programs_on_the_gpu` checks the
    // HLSL path's registers with the `check_int_*` below and compares the per-lane form with it

    /// a = lane · 0x9E3779B9 (v1), b = lane − (a >> 13) (v2, negative but at lane 0) and
    /// c = 0x7FFFFFF0 + lane (v3, past i32::MAX from lane 16)
    const INT_INPUTS: [I; 5] = [
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 255), ("LITERAL", 0x9E37_79B9)]),
        ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", V + 9)]),
        ("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 141), ("VSRC1", 1)]),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 2)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0x7FFF_FFF0)]),
    ];

    /// Integer VALU on a, b, c (`INT_INPUTS`): high products, 24-bit and 16-bit multiplies (OP_SEL
    /// halves), bitfields with offset + width past 32 and width 0, align, SAD, min3 / max3 / med3,
    /// clamped (saturating) adds, bit counts and scans of 0 and −1, packs that saturate.
    pub(crate) const INT_ARITHMETIC: &[I] = &[
        INT_INPUTS[0], INT_INPUTS[1], INT_INPUTS[2], INT_INPUTS[3], INT_INPUTS[4],
        ("V_MUL_HI_U32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", V + 2)]),
        ("V_MUL_HI_I32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 1), ("SRC1", V + 2)]),
        ("V_MUL_U32_U24", "ENC_VOP2", &[("VDST", 12), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_MUL_HI_U32_U24", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_MUL_I32_I24", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_MUL_HI_I32_I24", "ENC_VOP2", &[("VDST", 15), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MAD_I32_I24", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        // bfe(a, lane, 8) unsigned and signed, bfe(a, 4, lane), bfi, bfm(lane, 4)
        ("V_BFE_U32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 1), ("SRC1", V), ("SRC2", 136)]),
        ("V_BFE_I32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 1), ("SRC1", V), ("SRC2", 136)]),
        ("V_BFE_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", 132), ("SRC2", V)]),
        ("V_BFI_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_BFM_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V), ("SRC1", 132)]),
        ("V_ALIGNBIT_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V)]),
        ("V_ALIGNBYTE_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V)]),
        ("V_XAD_U32", "ENC_VOP3", &[("VDST", 25), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_SAD_U8", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_SAD_HI_U8", "ENC_VOP3", &[("VDST", 27), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_SAD_U16", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_SAD_U32", "ENC_VOP3", &[("VDST", 29), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MSAD_U8", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MIN3_U32", "ENC_VOP3", &[("VDST", 31), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MAX3_I32", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MED3_U32", "ENC_VOP3", &[("VDST", 33), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MED3_I32", "ENC_VOP3", &[("VDST", 34), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MIN3_I32", "ENC_VOP3", &[("VDST", 35), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        ("V_MAX3_U32", "ENC_VOP3", &[("VDST", 36), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3)]),
        // clamp: a + b + c, a + b, b − a (u32); a + c, b − c (i32); a + c without
        ("V_ADD3_U32", "ENC_VOP3", &[("VDST", 37), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("CLAMP", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 38), ("SRC0", V + 1), ("SRC1", V + 2), ("CLAMP", 1)]),
        ("V_SUB_NC_U32", "ENC_VOP3", &[("VDST", 39), ("SRC0", V + 2), ("SRC1", V + 1), ("CLAMP", 1)]),
        ("V_ADD_NC_I32", "ENC_VOP3", &[("VDST", 40), ("SRC0", V + 1), ("SRC1", V + 3), ("CLAMP", 1)]),
        ("V_SUB_NC_I32", "ENC_VOP3", &[("VDST", 41), ("SRC0", V + 2), ("SRC1", V + 3), ("CLAMP", 1)]),
        ("V_ADD_NC_I32", "ENC_VOP3", &[("VDST", 42), ("SRC0", V + 1), ("SRC1", V + 3)]),
        // bcnt(a) + lane, ffbh / ffbl / ffbh_i32 (0 and −1 at lane 0), bfrev, not, xnor
        ("V_BCNT_U32_B32", "ENC_VOP3", &[("VDST", 43), ("SRC0", V + 1), ("SRC1", V)]),
        ("V_FFBH_U32", "ENC_VOP1", &[("VDST", 44), ("SRC0", V + 1)]),
        ("V_FFBL_B32", "ENC_VOP1", &[("VDST", 45), ("SRC0", V + 2)]),
        ("V_FFBH_I32", "ENC_VOP1", &[("VDST", 46), ("SRC0", V + 2)]),
        ("V_BFREV_B32", "ENC_VOP1", &[("VDST", 47), ("SRC0", V + 1)]),
        ("V_NOT_B32", "ENC_VOP1", &[("VDST", 48), ("SRC0", V + 2)]),
        ("V_XNOR_B32", "ENC_VOP2", &[("VDST", 49), ("SRC0", V + 1), ("VSRC1", 2)]),
        // packs of (a, b) to u16 and (b, c) to i16; 16-bit mads of a.hi · b.lo, a.lo · b.hi
        ("V_CVT_PK_U16_U32", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 1), ("SRC1", V + 2)]),
        ("V_CVT_PK_I16_I32", "ENC_VOP3", &[("VDST", 51), ("SRC0", V + 2), ("SRC1", V + 3)]),
        ("V_MAD_U32_U16", "ENC_VOP3", &[("VDST", 52), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("OP_SEL", 1)]),
        ("V_MAD_I32_I16", "ENC_VOP3", &[("VDST", 53), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("OP_SEL", 2)]),
        ("V_MIN_I32", "ENC_VOP2", &[("VDST", 54), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_MAX_U32", "ENC_VOP2", &[("VDST", 55), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 56), ("SRC0", V + 1), ("SRC1", V), ("SRC2", V + 3)]),
        ("V_ADD_LSHL_U32", "ENC_VOP3", &[("VDST", 57), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V)]),
        ENDPGM,
    ];

    /// Carries on a, b, c (`INT_INPUTS`): VOP3 carry-outs into s20..s30 and carry-ins from them,
    /// a VOP2 chain through VCC, a VOP3 cndmask on s22, v_mad_u64_u32 / v_mad_i64_i32 with their
    /// carry-outs (s32, s34) and the 64-bit shifts by 2 · lane + 3 (65 at lane 31: 1).
    pub(crate) const INT_CARRIES: &[I] = &[
        INT_INPUTS[0], INT_INPUTS[1], INT_INPUTS[2], INT_INPUTS[3], INT_INPUTS[4],
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 10), ("SDST", 20), ("SRC0", V + 1), ("SRC1", V + 2)]),
        ("V_ADD_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 11), ("SDST", 22), ("SRC0", V + 1), ("SRC1", V + 3), ("SRC2", 20)]),
        ("V_SUB_CO_U32", "VOP3_SDST_ENC", &[("VDST", 12), ("SDST", 24), ("SRC0", V + 1), ("SRC1", V + 2)]),
        ("V_SUB_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 13), ("SDST", 26), ("SRC0", V + 3), ("SRC1", V + 2), ("SRC2", 24)]),
        ("V_SUBREV_CO_U32", "VOP3_SDST_ENC", &[("VDST", 14), ("SDST", 28), ("SRC0", V + 1), ("SRC1", V + 2)]),
        ("V_SUBREV_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 15), ("SDST", 30), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", 28)]),
        // vcc = a > b; a + b + vcc, c − a − vcc, a − c − vcc, each into vcc
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 16), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_SUB_CO_CI_U32", "ENC_VOP2", &[("VDST", 17), ("SRC0", V + 3), ("VSRC1", 1)]),
        ("V_SUBREV_CO_CI_U32", "ENC_VOP2", &[("VDST", 18), ("SRC0", V + 3), ("VSRC1", 1)]),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", 22)]),
        // v[20:21] = a · b + (c | a << 32), v[24:25] the same signed; shifts of them and of v[22:23]
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 22), ("SRC0", V + 3)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 1)]),
        ("V_MAD_U64_U32", "VOP3_SDST_ENC", &[("VDST", 20), ("SDST", 32), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 22)]),
        ("V_MAD_I64_I32", "VOP3_SDST_ENC", &[("VDST", 24), ("SDST", 34), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 22)]),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", 129), ("SRC2", 131)]),
        ("V_LSHLREV_B64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 4), ("SRC1", V + 20)]),
        ("V_LSHRREV_B64", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 4), ("SRC1", V + 22)]),
        ("V_ASHRREV_I64", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 4), ("SRC1", V + 24)]),
        ENDPGM,
    ];

    /// `INT_INPUTS`' a, b, c of a lane
    fn int_inputs(l: usize) -> (u32, u32, u32) {
        let a = (l as u32).wrapping_mul(0x9E37_79B9);
        (a, (l as u32).wrapping_sub(a >> 13), 0x7FFF_FFF0 + l as u32)
    }

    pub(crate) fn check_int_arithmetic(_: &[u32], v: &[Vec<u32>]) {
        let (u24, i24) = (|x: u32| (x & 0xFF_FFFF) as u64, |x: u32| ((x << 8) as i32 >> 8) as i64);
        let i16s = |x: u32| x as u16 as i16 as i32;
        let bytes = |x: u32| x.to_le_bytes().map(|b| b as i32);
        let first = |n: u32, x: u32| if x == 0 { u32::MAX } else { n };
        for (l, r) in v.iter().enumerate() {
            let (a, b, c) = int_inputs(l);
            let n = l as u32;
            let (ai, bi, ci) = (a as i32, b as i32, c as i32);
            assert_eq!((r[1], r[2], r[3]), (a, b, c), "lane {l}");
            let mask = |w: u32| ((1u64 << w) - 1) as u32;
            let sext = |x: u32, w: u32| if w == 0 { 0 } else { ((x << (32 - w)) as i32 >> (32 - w)) as u32 };
            let (ba, bb) = (bytes(a), bytes(b));
            let sad8: i32 = (0..4).map(|k| (ba[k] - bb[k]).abs()).sum();
            let msad8: i32 = (0..4).filter(|&k| bb[k] != 0).map(|k| (ba[k] - bb[k]).abs()).sum();
            let sad16 = (i16s(a) as u16 as i32 - b as u16 as i32).abs() + ((a >> 16) as i32 - (b >> 16) as i32).abs();
            let med3 = |x: u32, y: u32, z: u32| x.min(y).max(x.max(y).min(z));
            let med3i = |x: i32, y: i32, z: i32| x.min(y).max(x.max(y).min(z));
            let ffbh_i = |x: u32| if x == 0 || x == u32::MAX { u32::MAX } else { (x ^ (x as i32 >> 31) as u32).leading_zeros() };
            let want = [
                ((a as u64 * b as u64) >> 32) as u32,
                ((ai as i64 * bi as i64) >> 32) as u32,
                (u24(a) * u24(b)) as u32,
                ((u24(a) * u24(b)) >> 32) as u32,
                (i24(a) * i24(b)) as u32,
                ((i24(a) * i24(b)) >> 32) as u32,
                ((u24(a) * u24(b)) as u32).wrapping_add(c),
                ((i24(a) * i24(b)) as u32).wrapping_add(c),
                a >> (n & 31) & mask(8),
                sext((ai >> (n & 31)) as u32 & mask(8), 8),
                a >> 4 & mask(n & 31),
                a & b | !a & c,
                mask(n & 31) << 4,
                (((a as u64) << 32 | b as u64) >> (n & 31)) as u32,
                (((a as u64) << 32 | b as u64) >> (8 * (n & 3))) as u32,
                (a ^ b).wrapping_add(c),
                (sad8 as u32).wrapping_add(c),
                ((sad8 as u32) << 16).wrapping_add(c),
                (sad16 as u32).wrapping_add(c),
                a.abs_diff(b).wrapping_add(c),
                (msad8 as u32).wrapping_add(c),
                a.min(b).min(c),
                ai.max(bi).max(ci) as u32,
                med3(a, b, c),
                med3i(ai, bi, ci) as u32,
                ai.min(bi).min(ci) as u32,
                a.max(b).max(c),
                (a as u64 + b as u64 + c as u64).min(u32::MAX as u64) as u32,
                a.saturating_add(b),
                b.saturating_sub(a),
                ai.saturating_add(ci) as u32,
                bi.saturating_sub(ci) as u32,
                a.wrapping_add(c),
                a.count_ones() + n,
                first(a.leading_zeros(), a),
                first(b.trailing_zeros(), b),
                ffbh_i(b),
                a.reverse_bits(),
                !b,
                !(a ^ b),
                a.min(0xFFFF) | b.min(0xFFFF) << 16,
                bi.clamp(-32768, 32767) as u32 & 0xFFFF | (ci.clamp(-32768, 32767) as u32) << 16,
                (a >> 16).wrapping_mul(b & 0xFFFF).wrapping_add(c),
                (i16s(a) * i16s(b >> 16)).wrapping_add(ci) as u32,
                ai.min(bi) as u32,
                a.max(b),
                (a << (n & 31)).wrapping_add(c),
                a.wrapping_add(b) << (n & 31),
            ];
            for (k, w) in want.iter().enumerate() {
                assert_eq!(r[10 + k], *w, "lane {l} v{}: {:#x} against {w:#x}", 10 + k, r[10 + k]);
            }
        }
    }

    pub(crate) fn check_int_carries(s: &[u32], v: &[Vec<u32>]) {
        let mut masks = [0u32; 8];
        for (l, r) in v.iter().enumerate() {
            let (a, b, c) = int_inputs(l);
            let (a64, b64, c64) = (a as u64, b as u64, c as u64);
            let bit = |m: u32| (m >> l & 1) as u64;
            let carry = [a64 + b64 > u32::MAX as u64, a64 + c64 + bit(s[20]) > u32::MAX as u64, b > a, b64 + bit(s[24]) > c64, a > b, b64 + bit(s[28]) > c64];
            let want = [a.wrapping_add(b), (a64 + c64 + bit(s[20])) as u32, a.wrapping_sub(b), (c64.wrapping_sub(b64 + bit(s[24]))) as u32, b.wrapping_sub(a), (c64.wrapping_sub(b64 + bit(s[28]))) as u32];
            for k in 0..6 {
                assert_eq!(r[10 + k], want[k], "lane {l} v{}", 10 + k);
                masks[k] |= (carry[k] as u32) << l;
            }
            // the VCC chain
            let v0 = (a > b) as u64;
            let w16 = a64 + b64 + v0;
            let v1 = (w16 >> 32) as u64;
            let v2 = (a64 + v1 > c64) as u64;
            let v3 = (c64 + v2 > a64) as u64;
            assert_eq!([r[16], r[17], r[18]], [w16 as u32, c64.wrapping_sub(a64 + v1) as u32, a64.wrapping_sub(c64 + v2) as u32], "lane {l}: through vcc");
            masks[6] |= (v3 as u32) << l;
            assert_eq!(r[19], if bit(s[22]) == 1 { b } else { a }, "lane {l}: cndmask");
            let addend = c64 | a64 << 32;
            let (wu, cu) = (a64 * b64).overflowing_add(addend);
            let wi = ((a as i32 as i64 * b as i32 as i64) as u64).wrapping_add(addend);
            masks[7] |= (cu as u32) << l;
            let sh = (2 * l as u32 + 3) & 63;
            let pair = |i: usize| r[i] as u64 | (r[i + 1] as u64) << 32;
            assert_eq!([pair(20), pair(22), pair(24)], [wu, addend, wi], "lane {l}: mads");
            assert_eq!([pair(26), pair(28), pair(30)], [wu << sh, addend >> sh, ((wi as i64) >> sh) as u64], "lane {l}: 64-bit shifts");
        }
        let got = [s[20], s[22], s[24], s[26], s[28], s[30], s[106], s[32]];
        assert_eq!(got, masks, "carry-outs s20..s30, vcc, s32");
    }

    pub(crate) fn compute_cs() -> crate::gpu::recomp::CsInterface {
        crate::gpu::recomp::CsInterface { threads: [64, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 }
    }

    /// A compute program run on the CPU over `groups` groups (wave32): the memory after.
    pub(crate) fn run_compute(lines: &[I], cs: &crate::gpu::recomp::CsInterface, groups: u32, sgprs: &[u32], mem: &[u32]) -> Vec<u32> {
        run_compute_waves(lines, cs, false, groups, sgprs, mem)
    }

    /// … in wave32 or wave64: each wave takes the next `lanes` threads of its group, as on the
    /// console (the last one fewer).
    pub(crate) fn run_compute_waves(lines: &[I], cs: &crate::gpu::recomp::CsInterface, wave64: bool, groups: u32, sgprs: &[u32], mem: &[u32]) -> Vec<u32> {
        let ir = lower_compute(&decode(lines), cs, wave64, Interp::default()).unwrap_or_else(|e| panic!("{e}"));
        let lanes = if wave64 { 64 } else { 32 };
        let mut io = vec![0u32; KERNEL_SGPRS];
        io[..sgprs.len()].copy_from_slice(sgprs);
        let mut res = vec![0u32; KERNEL_SGPRS];
        let mut mem = mem.to_vec();
        let threads: u32 = cs.threads.iter().product();
        for g in 0..groups {
            for w in (0..threads).step_by(lanes as usize) {
                ir.run_with(lanes.min(threads - w) as usize, w, [g, 0, 0], &io, &mut res, &mut mem);
            }
        }
        mem
    }

    /// 2-D and 3-D groups of 64 threads.
    pub(crate) const ID_SHAPES: [[u32; 3]; 2] = [[8, 8, 1], [4, 4, 4]];

    /// Each thread of a 64-thread group of an `ID_SHAPES` shape stores its ids and its lane
    /// (`mbcnt` of EXEC) as `x | y << 8 | z << 16 | lane << 24` at word `group * 64 + flat index`.
    pub(crate) fn ids_program(threads: [u32; 3], wave64: bool) -> Vec<I> {
        let mut p: Vec<I> = vec![("S_MUL_I32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 192)])];
        // v4 = flat index: (y << 3) + x, or (z << 4) + (y << 2) + x
        if threads[2] == 1 {
            p.push(("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 1), ("SRC1", 131), ("SRC2", V)]));
        } else {
            p.push(("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 1), ("SRC1", 130), ("SRC2", V)]));
            p.push(("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 2), ("SRC1", 132), ("SRC2", V + 4)]));
        }
        // v[8:9] = (group * 64 + flat) * 4; v5 = lane
        p.extend_from_slice(&[
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 3), ("VSRC1", 4)]),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 130), ("VSRC1", 4)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)]),
            ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 126), ("SRC1", 128)]),
        ]);
        if wave64 {
            p.push(("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 127), ("SRC1", V + 5)]));
        }
        p.push(("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 1), ("SRC1", 136), ("SRC2", V)]));
        if threads[2] > 1 {
            p.push(("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 2), ("SRC1", 144), ("SRC2", V + 7)]));
        }
        p.extend_from_slice(&[
            ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 5), ("SRC1", 152), ("SRC2", V + 7)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 8), ("DATA", 7), ("SADDR", 125), ("OFFSET", 0)]),
            ENDPGM,
        ]);
        p
    }

    pub(crate) fn ids_cs(threads: [u32; 3]) -> crate::gpu::recomp::CsInterface {
        let tid_components = if threads[2] > 1 { 3 } else { 2 };
        crate::gpu::recomp::CsInterface { threads, tid_components, ..compute_cs() }
    }

    pub(crate) fn memory_image() -> Vec<u32> {
        (0..256).map(|i| 1000 + i).collect()
    }

    // f64 programs, as the f32 ones: `lower_dxil`'s `f64_programs_on_the_gpu` checks the HLSL
    // path's registers with `check_f64` and compares the per-lane form with it

    /// f64 on x = 1 + lane / 32 + lane · 2^-52 (v[2:3]): x + 2^-53 (ties to even), x · 2^-1040
    /// (a denormal in and out), x · f64::MAX (∞ from lane 1), fma(x, x, −x · x) (the rounding
    /// error: nonzero only fused), NaN into min / max, VOP3 abs / neg / omod / clamp; floor / ceil
    /// / trunc / rndne / fract / frexp of q = (lane − 16) / 4 (ties, −0 at lane 15); f64 ↔ f32 /
    /// i32 / u32 (a tie, overflow, saturation, NaN); rcp / rsq / sqrt.
    pub(crate) const F64_ARITHMETIC: &[I] = &[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 143), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3), ("LITERAL", 0x3FF0_0000)]),
        // v[4:5] = 2^-53, v[6:7] = 2^-1040, v[8:9] = f64::MAX, v[10:11] = NaN
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 128)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 255), ("LITERAL", 0x3CA0_0000)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 128)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 132)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 193)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 255), ("LITERAL", 0x7FEF_FFFF)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 129)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 255), ("LITERAL", 0x7FF8_0000)]),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("SRC1", V + 4)]),
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", V + 6)]),
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 2), ("SRC1", V + 8)]),
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 2), ("SRC1", V + 2)]),
        ("V_FMA_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 2), ("SRC1", V + 2), ("SRC2", V + 18), ("NEG", 4)]),
        ("V_MIN_F64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 10), ("SRC1", V + 2)]),
        ("V_MAX_F64", "ENC_VOP3", &[("VDST", 24), ("SRC0", V + 2), ("SRC1", V + 10), ("NEG", 2)]),
        // −x · |−4|, (x − 1) · 4, clamp(fma(x, x, −1))
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 2), ("SRC1", 247), ("ABS", 2), ("NEG", 1)]),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 2), ("SRC1", 243), ("OMOD", 2)]),
        ("V_FMA_F64", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 2), ("SRC1", V + 2), ("SRC2", 243), ("CLAMP", 1)]),
        // q = ldexp(lane − 16, −2)
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F64_I32", "ENC_VOP1", &[("VDST", 34), ("SRC0", V + 32)]),
        ("V_LDEXP_F64", "ENC_VOP3", &[("VDST", 36), ("SRC0", V + 34), ("SRC1", 194)]),
        ("V_FLOOR_F64", "ENC_VOP1", &[("VDST", 38), ("SRC0", V + 36)]),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 40), ("SRC0", V + 36)]),
        ("V_TRUNC_F64", "ENC_VOP1", &[("VDST", 42), ("SRC0", V + 36)]),
        ("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 44), ("SRC0", V + 36)]),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 46), ("SRC0", V + 36)]),
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 48), ("SRC0", V + 36)]),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 50), ("SRC0", V + 36)]),
        // f32 of z = 1 + 2^-24 + lane · 2^-52 (a tie at lane 0) and of f64::MAX; i32 of −4x, of
        // x · f64::MAX and of NaN, u32 of −4x and of NaN
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 52), ("SRC0", 255), ("VSRC1", 0), ("LITERAL", 0x1000_0000)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 53), ("SRC0", 255), ("LITERAL", 0x3FF0_0000)]),
        ("V_CVT_F32_F64", "ENC_VOP1", &[("VDST", 51), ("SRC0", V + 52)]),
        ("V_CVT_F32_F64", "ENC_VOP1", &[("VDST", 54), ("SRC0", V + 8)]),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 55), ("SRC0", V + 26)]),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 56), ("SRC0", V + 16)]),
        ("V_CVT_I32_F64", "ENC_VOP1", &[("VDST", 57), ("SRC0", V + 10)]),
        ("V_CVT_U32_F64", "ENC_VOP1", &[("VDST", 58), ("SRC0", V + 26)]),
        ("V_CVT_U32_F64", "ENC_VOP1", &[("VDST", 59), ("SRC0", V + 10)]),
        // f64 of u32::MAX − lane and of the f32 at v51
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 193), ("VSRC1", 0)]),
        ("V_CVT_F64_U32", "ENC_VOP1", &[("VDST", 60), ("SRC0", V + 1)]),
        ("V_CVT_F64_F32", "ENC_VOP1", &[("VDST", 62), ("SRC0", V + 51)]),
        ("V_RCP_F64", "ENC_VOP1", &[("VDST", 64), ("SRC0", V + 2)]),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 66), ("SRC0", V + 2)]),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 68), ("SRC0", V + 2)]),
        ENDPGM,
    ];

    pub(crate) fn check_f64(_: &[u32], v: &[Vec<u32>]) {
        for (l, r) in v.iter().enumerate() {
            let f = |i: usize| f64::from_bits(r[i] as u64 | (r[i + 1] as u64) << 32);
            let x = 1.0 + l as f64 / 32.0 + l as f64 * 2f64.powi(-52);
            let q = (l as f64 - 16.0) / 4.0;
            assert_eq!(f(2).to_bits(), x.to_bits(), "lane {l}");
            let p = x * x;
            let e = if q == 0.0 { 0 } else { (q.abs().to_bits() >> 52) as i32 - 1022 };
            let want = [
                x + 2f64.powi(-53),
                x * 2f64.powi(-1040),
                x * f64::MAX,
                p,
                x.mul_add(x, -p),
                x,
                x,
                -4.0 * x,
                (x - 1.0) * 4.0,
                x.mul_add(x, -1.0).clamp(0.0, 1.0),
                (l as f64 - 16.0),
                q,
                q.floor(),
                q.ceil(),
                q.trunc(),
                q.round_ties_even(),
                q - q.floor(),
                q / 2f64.powi(e),
            ];
            for (k, w) in want.iter().enumerate() {
                let i = 12 + 2 * k + if k >= 10 { 2 } else { 0 };
                assert_eq!(f(i).to_bits(), w.to_bits(), "lane {l} v[{i}:{}]: {} against {w}", i + 1, f(i));
            }
            assert_eq!(r[50], e as u32, "lane {l}: frexp_exp");
            let z = 1.0 + 2f64.powi(-24) + l as f64 * 2f64.powi(-52);
            assert_eq!([r[51], r[54]], [(z as f32).to_bits(), f32::INFINITY.to_bits()], "lane {l}: f32 of f64");
            assert_eq!([r[55], r[56], r[57], r[58], r[59]], [(-4.0 * x) as i32 as u32, i32::MAX as u32, 0, 0, 0], "lane {l}: i32 / u32 of f64");
            assert_eq!([f(60), f(62)], [(u32::MAX - l as u32) as f64, z as f32 as f64], "lane {l}: f64 of u32, f32");
            // the GPU's rcp / rsq / sqrt are close, not exact
            for (i, w) in [(64, 1.0 / x), (66, 1.0 / x.sqrt()), (68, x.sqrt())] {
                assert!((f(i) - w).abs() <= w * 1e-12, "lane {l} v[{i}:{}]: {} against {w}", i + 1, f(i));
            }
        }
    }

    /// A line's bytes; a `("LITERAL", k)` field is the instruction's literal dword.
    pub(crate) fn encode_line((n, e, f): &I) -> Vec<u8> {
        let lit = f.iter().find(|(k, _)| *k == "LITERAL").map(|&(_, v)| v as u32);
        let f: Vec<(&str, u64)> = f.iter().copied().filter(|(k, _)| *k != "LITERAL").collect();
        rdna2::encode(n, e, &f, lit).unwrap()
    }

    pub(crate) fn decode(lines: &[I]) -> Vec<Decoded> {
        let bytes: Vec<u8> = lines.iter().flat_map(encode_line).collect();
        let mut out = Vec::new();
        let mut off = 0;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        rdna2::resolve_far_jumps(&mut out);
        out
    }

    /// A branch or a scalar select on whether any lane of a mask is set decides for the whole
    /// wave on the console: with lanes in EXEC that lack the bit (VCC = lanes 0–15, EXEC = all),
    /// each lane deciding by its own bit would differ in lanes 16–31, so the program is refused
    /// (Kalmarv's five programs on #1745); where the mask is constant over EXEC it is taken.
    #[test]
    fn a_whole_wave_decision_on_a_mask_some_lanes_lack_is_refused() {
        const CMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]);
        const MOV7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)]);
        const AND: I = ("S_AND_B32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 106), ("SSRC1", 126)]);
        let mixed: I = ("S_AND_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 106), ("SSRC1", 8)]);
        let programs: [(&str, Vec<I>); 8] = [
            ("mask and number + scc0", vec![CMP, mixed, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM]),
            ("mask and number + select", vec![CMP, mixed, ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 133), ("SSRC1", 135)]), ENDPGM]),
            ("mask compared with number", vec![CMP, ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 106), ("SSRC1", 8)]), ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM]),
            ("vccnz", vec![CMP, ("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM]),
            ("vccz", vec![CMP, ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM]),
            ("and + scc1", vec![CMP, AND, ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM]),
            ("cmp_lg + scc1", vec![CMP, ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 106), ("SSRC1", 128)]), ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM]),
            ("and + cselect", vec![CMP, AND, ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 133), ("SSRC1", 135)]), ENDPGM]),
        ];
        for (name, p) in programs {
            let err = lower_kernel(&decode(&p), false, Interp::default()).err().unwrap_or_else(|| panic!("{name}: taken"));
            assert!(err.contains("any lane of a mask") || err.contains("SCC of a mask op"), "{name}: {err}");
        }
        // the mask op's SCC reaching the `s_cselect` only around the loop's back edge
        // (thehollowedone's program on #1745: 7 where the console gives 5 in lanes 16–31)
        let around: Vec<I> = vec![
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 130)]),
            CMP,
            ("S_CMP_GT_U32", "ENC_SOPC", &[("SSRC0", 129), ("SSRC1", 128)]),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 133), ("SSRC1", 135)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 1)]),
            ("S_SUB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)]),
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)]),
            AND,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFF9)]),
            ENDPGM,
        ];
        let err = lower_kernel(&decode(&around), false, Interp::default()).err().expect("around the loop: taken");
        assert!(err.contains("SCC of a mask op") || err.contains("any lane of a mask"), "around the loop: {err}");
        // VCC = EXEC: the same in every lane in EXEC, so each lane decides as the wave does
        let uniform: [I; 4] = [("S_MOV_B32", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 126)]), ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)]), MOV7, ENDPGM];
        let (s, v) = state(32);
        let (_, v) = run(&uniform, false, &s, &v);
        assert!(v.iter().all(|r| r[1] == 7));
    }

    /// Only the last lane has a VCC bit. The branches inspect VCC after EXEC has been
    /// narrowed to the other lanes. Each case stores the scalar decision at mem[thread].
    pub(crate) fn wave_branch_cases(wave64: bool) -> Vec<(Vec<I>, u32)> {
        let cmp: I = if wave64 {
            ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 191), ("VSRC1", 0)])
        } else {
            ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 159), ("VSRC1", 0)])
        };
        let (narrow, restore, eq, lg) = if wave64 {
            ("S_ANDN2_B64", "S_MOV_B64", "S_CMP_EQ_U64", "S_CMP_LG_U64")
        } else {
            ("S_ANDN2_B32", "S_MOV_B32", "S_CMP_EQ_U32", "S_CMP_LG_U32")
        };
        let mut cases: Vec<(Vec<I>, u32)> = Vec::new();
        for (branch, expected) in [("S_CBRANCH_VCCNZ", 5), ("S_CBRANCH_VCCZ", 9), ("S_CBRANCH_EXECNZ", 5),
            ("S_CBRANCH_EXECZ", 9), ("S_CBRANCH_SCC1", 5), ("S_CBRANCH_SCC0", 9)] {
            cases.push((vec![
                cmp,
                (narrow, "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 133)]),
                (branch, "ENC_SOPP", &[("SIMM16", 1)]),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 137)]),
                (restore, "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
            ], expected));
        }
        for (compare, expected) in [(eq, 9), (lg, 5)] {
            cases.push((vec![
                cmp,
                (compare, "ENC_SOPC", &[("SSRC0", 106), ("SSRC1", 128)]),
                ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 133), ("SSRC1", 137)]),
            ], expected));
        }
        if !wave64 {
            // The second iteration's CSELECT reads the mask-derived SCC on the
            // backedge (thehollowedone, #1745).
            cases.push((vec![
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 130)]),
                ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
                ("S_CMP_GT_U32", "ENC_SOPC", &[("SSRC0", 129), ("SSRC1", 128)]),
                ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 133), ("SSRC1", 135)]),
                ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 6)]),
                ("S_SUB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)]),
                ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]),
                ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)]),
                ("S_AND_B32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 106), ("SSRC1", 126)]),
                ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xfff9)]),
            ], 5));
            // The bitwise op reads both a mask and a number (Pipodi, #1745).
            cases.push((vec![
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 133)]),
                ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
                ("S_AND_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 106), ("SSRC1", 8)]),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 135)]),
                ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)]),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 129)]),
            ], 1));
        }
        for (lines, _) in &mut cases {
            lines.extend_from_slice(&[
                ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 6)]),
                ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 0)]),
                ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 2), ("DATA", 1), ("SADDR", 125)]),
                ENDPGM,
            ]);
        }
        cases
    }

    #[test]
    fn native_wave_branches_keep_inactive_lanes_in_scalar_control_flow() {
        for wave64 in [false, true] {
            let lanes = if wave64 { 64 } else { 32 };
            let mut cs = compute_cs();
            cs.threads = [lanes, 1, 1];
            for (lines, expected) in wave_branch_cases(wave64) {
                let got = run_compute_waves(&lines, &cs, wave64, 1, &[], &vec![0; lanes as usize]);
                assert!(got.iter().all(|&word| word == expected), "wave{lanes}, {lines:?}: {got:?}");
            }
        }
    }

    /// A cross-lane scalar op on s0 holding a number takes the number path (the check for the
    /// upper dword of a mask pair at s<r - 1> must not compute r - 1 for s0: nobodyttk, #1269).
    #[test]
    fn a_wave_op_on_s0_holding_a_number() {
        let lines: &[I] = &[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 255), ("LITERAL", 0xF0F0)]),
            ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 0)]),
            ENDPGM,
        ];
        // the CPU lowering has no number form of it (DXIL takes it from #1426): refused, no panic
        let err = lower_kernel(&decode(lines), false, Interp::default()).err().expect("refused");
        assert_eq!(err, "unsupported S_BCNT1_I32_B32");
    }

    /// `s_wqm` of a mask sets every lane of a quad with a bit; `s_cselect` picks a mask or a
    /// number by SCC.
    #[test]
    fn whole_quad_mode_and_conditional_selects() {
        let lines: &[I] = &[
            // vcc = lane 5 only; s0 = wqm(vcc) = lanes 4–7
            ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 133), ("VSRC1", 0)]),
            ("S_WQM_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 128), ("SRC1", 129), ("SRC2", 0)]),
            // scc = (0 == 0): s1 = s0 (a mask), s2 = 5
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 128)]),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 128)]),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 133), ("SSRC1", 135)]),
            ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 128), ("SRC1", 129), ("SRC2", 1)]),
            // scc = (0 == 1): s3 = 7, s4 = no lanes
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 129)]),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 133), ("SSRC1", 135)]),
            ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 0), ("SSRC1", 128)]),
            ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 128), ("SRC1", 129), ("SRC2", 4)]),
            ENDPGM,
        ];
        let (s, v) = state(32);
        let (s, v) = run(lines, false, &s, &v);
        assert_eq!((s[2], s[3]), (5, 7));
        for (l, regs) in v.iter().enumerate() {
            let quad = (4..8).contains(&l) as u32;
            assert_eq!((regs[1], regs[2], regs[3]), (quad, quad, 0), "lane {l}");
        }
    }

    /// `s_addc_u32` / `s_subb_u32` take SCC as the carry / borrow in and leave theirs in SCC.
    #[test]
    fn add_with_carry_and_subtract_with_borrow() {
        let lines: &[I] = &[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 255), ("LITERAL", 0xFFFF_FFFF)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 129)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 128), ("SSRC1", 128)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 0), ("SSRC1", 128)]),
            ("S_SUB_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 128), ("SSRC1", 129)]),
            ("S_SUBB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 133), ("SSRC1", 129)]),
            ("S_SUBB_U32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 133), ("SSRC1", 129)]),
            ENDPGM,
        ];
        let (s, v) = state(32);
        let (s, _) = run(lines, false, &s, &v);
        // 0xFFFFFFFF + 1 = 0 carry 1; 0 + 0 + 1 = 1; 0xFFFFFFFF + 0 + 0 = 0xFFFFFFFF
        assert_eq!((s[1], s[2], s[5]), (0, 1, 0xFFFF_FFFF));
        // 0 - 1 = 0xFFFFFFFF borrow 1; 5 - 1 - 1 = 3; 5 - 1 - 0 = 4
        assert_eq!((s[3], s[4], s[6]), (0xFFFF_FFFF, 3, 4));
    }

    /// A far jump (`s_getpc_b64`, a constant added, `s_setpc_b64`, as a retail program jumps
    /// over a long `then` block) is a plain jump: the lanes skip the instructions it jumps over.
    #[test]
    fn a_far_jump_skips_what_it_jumps_over() {
        let far: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)]),
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 2)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 2), ("SSRC1", 255), ("LITERAL", 32)]),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 3), ("SSRC1", 128)]),
            ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 2)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 136)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 137)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 138)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 139)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 0)]),
            ENDPGM,
        ];
        let (s, v) = state(32);
        let (s, v) = run(far, false, &s, &v);
        // the address it computed: after `s_getpc_b64` (8) plus 32, past the four moves
        assert_eq!((s[2], s[3]), (40, 0));
        for (l, regs) in v.iter().enumerate() {
            assert_eq!((regs[1], regs[2]), (7, 7 + l as u32), "lane {l}");
        }
    }

    /// The kernel run on the CPU: initial (sgpr, vgpr[lane]) → final ones.
    fn run(lines: &[I], wave64: bool, sgpr: &[u32], vgpr: &[Vec<u32>]) -> (Vec<u32>, Vec<Vec<u32>>) {
        let (s, v, _) = run_mem(lines, wave64, sgpr, vgpr, &[]);
        (s, v)
    }

    /// … with the flat memory too: (sgpr, vgpr[lane], memory after).
    pub(crate) fn run_mem(lines: &[I], wave64: bool, sgpr: &[u32], vgpr: &[Vec<u32>], mem: &[u32]) -> (Vec<u32>, Vec<Vec<u32>>, Vec<u32>) {
        let insts = decode(lines);
        let (ir, _) = lower_kernel(&insts, wave64, Interp::default()).unwrap_or_else(|e| panic!("{e}"));
        let lanes = if wave64 { 64 } else { 32 };
        let mut io = vec![0u32; KERNEL_SGPRS + 64 * KERNEL_VGPRS];
        io[..sgpr.len()].copy_from_slice(sgpr);
        for (l, regs) in vgpr.iter().enumerate().take(lanes) {
            io[KERNEL_SGPRS + l * KERNEL_VGPRS..][..regs.len()].copy_from_slice(regs);
        }
        let mut res = io.clone();
        let mut mem = mem.to_vec();
        ir.run(lanes, &io, &mut res, &mut mem);
        let v = (0..lanes).map(|l| res[KERNEL_SGPRS + l * KERNEL_VGPRS..][..KERNEL_VGPRS].to_vec()).collect();
        (res[..KERNEL_SGPRS].to_vec(), v, mem)
    }

    pub(crate) fn state(lanes: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut s = vec![0u32; KERNEL_SGPRS];
        s[126] = u32::MAX;
        s[127] = u32::MAX;
        // v0 = the lane, v1 = 0
        (s, (0..lanes).map(|l| { let mut v = vec![0u32; 8]; v[0] = l as u32; v }).collect())
    }

    #[test]
    fn straight_line_scalar_and_vector_arithmetic() {
        let (s, v) = state(32);
        let (s, v) = run(STRAIGHT, false, &s, &v);
        assert_eq!((s[0], s[1]), (5, 10));
        for (l, regs) in v.iter().enumerate() {
            assert_eq!((regs[1], regs[2]), (10, 10 + l as u32), "lane {l}");
        }
    }

    #[test]
    fn integer_min_and_max() {
        let lines: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 129)]), // 1
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 193)]), // -1 (0xFFFF_FFFF)
            ("V_MIN_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 2)]),
            ("V_MAX_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 1), ("VSRC1", 2)]),
            ("V_MIN_I32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 1), ("VSRC1", 2)]),
            ("V_MAX_I32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 1), ("VSRC1", 2)]),
            ENDPGM,
        ];
        let (s, v) = state(32);
        let (_, v) = run(lines, false, &s, &v);
        assert_eq!((v[0][3], v[0][4], v[0][5], v[0][6]), (1, u32::MAX, u32::MAX, 1));
    }

    /// The survey goes on past a refused instruction: both are reported, and a program the
    /// lowering takes gives none.
    #[test]
    fn the_survey_reports_every_refusal() {
        let mut lines = STRAIGHT.to_vec();
        lines.insert(2, ("V_ADD_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 1)]));
        lines.insert(4, ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V), ("VSRC1", 1)]));
        let cs = compute_cs();
        assert!(lower_compute(&decode(&lines), &cs, false, Interp::default()).is_err());
        assert_eq!(survey_compute(&decode(&lines), &cs, false, Interp::default()), ["unsupported V_ADD_F32", "unsupported V_MUL_F32"]);
        assert!(survey_compute(&decode(STRAIGHT), &cs, false, Interp::default()).is_empty());
    }

    #[test]
    fn a_skipped_if_runs_only_its_lanes() {
        let (s, v) = state(32);
        let (s, v) = run(SKIPPED_IF, false, &s, &v);
        for (l, regs) in v.iter().enumerate() {
            let want = if l < 16 { 7 } else { 0 };
            assert_eq!((regs[1], regs[2]), (want, want + 1), "lane {l}");
        }
        assert_eq!(s[106], 0xFFFF, "vcc: the lanes below 16");
        assert_eq!((s[0], s[126]), (u32::MAX, u32::MAX), "saved and restored EXEC");
    }

    #[test]
    fn lanes_leave_a_loop_one_by_one() {
        let (s, v) = state(32);
        let (s, v) = run(LEAVING_LOOP, false, &s, &v);
        for (l, regs) in v.iter().enumerate() {
            assert_eq!(regs[1], (l as u32).max(1), "lane {l}");
        }
        assert_eq!(s[126], u32::MAX);
    }

    #[test]
    fn a_uniform_loop_with_a_scalar_counter() {
        let (s, v) = state(32);
        let (s, v) = run(UNIFORM_LOOP, false, &s, &v);
        assert_eq!(s[1], 12);
        assert!(v.iter().all(|r| r[1] == 4));
    }

    #[test]
    fn lanes_outside_exec_at_launch_keep_their_registers() {
        let (mut s, v) = state(32);
        s[126] = 0x0000_00FF;
        let (s, v) = run(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 137)]), ENDPGM], false, &s, &v);
        for (l, regs) in v.iter().enumerate() {
            assert_eq!(regs[1], if l < 8 { 9 } else { 0 }, "lane {l}");
        }
        assert_eq!(s[126], 0xFF);
    }

    /// A VS / PS run on the CPU in one wave of `lanes` lanes (a wave64 one in 32 lanes: in
    /// 32-lane waves): `launch` its launch EXEC, `vgpr[lane]` the registers it starts with. The
    /// memory after.
    fn run_graphics(lines: &[I], pixel: bool, wave64: bool, lanes: usize, launch: u64, vgpr: &[Vec<u32>], mem: &[u32]) -> Vec<u32> {
        let ir = lower_graphics(&decode(lines), pixel, wave64, wave64 && lanes == 32, Interp::default()).unwrap_or_else(|e| panic!("{e}"));
        let mut io = vec![0u32; KERNEL_SGPRS + 64 * KERNEL_VGPRS];
        io[EXEC as usize] = launch as u32;
        io[EXEC as usize + 1] = (launch >> 32) as u32;
        for (l, regs) in vgpr.iter().enumerate().take(lanes) {
            io[KERNEL_SGPRS + l * KERNEL_VGPRS..][..regs.len()].copy_from_slice(regs);
        }
        let mut res = vec![0u32; KERNEL_SGPRS];
        let mut mem = mem.to_vec();
        ir.run(lanes, &io, &mut res, &mut mem);
        mem
    }

    /// v7 = 9, stored at v[8:9] (the lane's address: 4 × lane)
    const SET_V7: I = ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 137)]);
    const STORE_V7: I = ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 8), ("DATA", 7), ("SADDR", 125), ("OFFSET", 0)]);

    fn store_rows() -> Vec<Vec<u32>> {
        (0..64).map(|l| { let mut v = vec![0u32; 10]; v[0] = l; v[1] = 15; v[8] = 4 * l; v }).collect()
    }

    #[test]
    fn a_ps_helper_lane_runs_the_program_but_does_not_store() {
        // even lanes covered, odd lanes helper lanes
        let launch = 0x5555_5555u64;
        let want: Vec<u32> = memory_image().iter().enumerate().map(|(l, &m)| if l < 32 && l % 2 == 0 { 9 } else { m }).collect();
        let exact = run_graphics(&[SET_V7, STORE_V7, ENDPGM], true, false, 32, launch, &store_rows(), &memory_image());
        assert_eq!(exact, want, "before EXEC is written, the store keeps its guard");
        // whole-quad mode, then back to the launch EXEC before the store (as compiled PSs do)
        let wqm32: Vec<I> = vec![
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
            ("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)]),
            SET_V7,
            ("S_AND_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            STORE_V7,
            ENDPGM,
        ];
        assert_eq!(run_graphics(&wqm32, true, false, 32, launch, &store_rows(), &memory_image()), want);
        // the same in wave64, run in 32-lane waves
        let wqm64: Vec<I> = vec![
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
            ("S_WQM_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)]),
            SET_V7,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            STORE_V7,
            ENDPGM,
        ];
        assert_eq!(run_graphics(&wqm64, true, true, 32, launch, &store_rows(), &memory_image()), want);
        // a VS: each D3D vertex invocation is one lane, so every lane that runs is in EXEC at
        // launch and stores, whatever bit its start gives it
        let all: Vec<u32> = memory_image().iter().enumerate().map(|(l, &m)| if l < 32 { 9 } else { m }).collect();
        assert_eq!(run_graphics(&[SET_V7, STORE_V7, ENDPGM], false, false, 32, launch, &store_rows(), &memory_image()), all);
        // … so its store keeps no guard (no block of its own), where a PS's does
        let blocks = |pixel: bool| lower_graphics(&decode(&[SET_V7, STORE_V7, ENDPGM]), pixel, false, false, Interp::default()).unwrap().blocks.len();
        assert_eq!(blocks(false) + 2, blocks(true), "a guarded store adds two blocks: the PS's only");
    }

    #[test]
    fn whole_quad_instructions_need_every_lane_of_the_quad() {
        // v_cmp (lanes above 15); s_and_saveexec s0; s_cbranch_execz L; s_wqm exec, s0; v_mov;
        // L: exec |= s0 – the s_wqm of the launch EXEC, in a block only some lanes reach
        let p: Vec<I> = vec![
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)]),
            ("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
            SET_V7,
            ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            ENDPGM,
        ];
        let e = lower_graphics(&decode(&p), true, false, false, Interp::default()).err();
        assert_eq!(e.as_deref(), Some("S_WQM_B32: needs whole quads, in a block not every lane reaches"));
        // s_wqm of another mask: the lanes of the quad are not read
        let other: Vec<I> = vec![("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]), ("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 106)]), SET_V7, ENDPGM];
        let e = lower_graphics(&decode(&other), true, false, false, Interp::default()).err();
        assert_eq!(e.as_deref(), Some("S_WQM_B32 of a mask other than the launch EXEC"));
        for (name, quad) in [
            ("IMAGE_SAMPLE", true),
            ("IMAGE_SAMPLE_B_CL", true),
            ("IMAGE_GATHER4_C", true),
            ("IMAGE_GET_LOD", true),
            ("S_WQM_B64", true),
            ("IMAGE_SAMPLE_L", false),
            ("IMAGE_SAMPLE_C_LZ_O", false),
            ("IMAGE_SAMPLE_D_G16", false),
            ("IMAGE_SAMPLE_CD", false),
            ("IMAGE_GATHER4_LZ", false),
            ("IMAGE_LOAD", false),
        ] {
            assert_eq!(whole_quad(name), quad, "{name}");
        }
    }

    #[test]
    fn a_wave64_vs_in_32_lane_waves_that_reads_upper_lanes_is_refused() {
        // whole-wave code writes lanes 32–63 (v6 = 5), then v_readlane of lane 32
        let p: Vec<I> = vec![
            ("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 133)]),
            ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)]),
            ENDPGM,
        ];
        let insts = decode(&p);
        assert_eq!(super::super::upper_lanes::analyze(&insts).reads.values().copied().collect::<Vec<_>>(), vec![super::super::upper_lanes::Read::Const(5)]);
        let e = lower_graphics(&insts, false, true, true, Interp::default()).err();
        assert_eq!(e.as_deref(), Some("wave64 VS / PS in 32-lane waves: a read of lanes 32–63"));
        assert!(lower_graphics(&insts, false, true, false, Interp::default()).is_ok(), "in 64-lane waves the lanes are there");
    }

    #[test]
    fn wave64_masks_span_both_words() {
        // v_cmp_gt_u32 vcc, 40, v0 over 64 lanes: lanes 0–39
        let (s, v) = state(64);
        let (s, _) = run(&[("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]), ENDPGM], true, &s, &v);
        assert_eq!((s[106], s[107]), (u32::MAX, 0xFF));
    }

    /// The lowering run on the CPU and the HLSL path run on the GPU give the same registers.
    #[test]
    fn the_lowering_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Options};
        if !gpu_waves() || dxc::location().is_err() {
            runner::skip_test(if !gpu_waves() { "no GPU with 32-lane waves" } else { "no DXC" });
            return;
        }
        for (name, lines) in [("straight", STRAIGHT), ("skipped if", SKIPPED_IF), ("leaving loop", LEAVING_LOOP), ("uniform loop", UNIFORM_LOOP), ("memory", MEMORY)] {
            let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
            let (s, v) = state(32);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let t = translate(&bytes, &Options { wave64: false, ..Default::default() }).unwrap();
            assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
            let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap();
            let (cs, cv, cmem) = run_mem(lines, false, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let insts = decode(lines);
            // masks some lanes kept their own older bits of (`lower_kernel`'s `stale`): only the
            // kernel ABI's final store sees them
            let (_, stale) = lower_kernel(&insts, false, Interp::default()).unwrap();
            let mut sregs: Vec<usize> = vec![106, 126];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(cs[r], gpu.sgpr[r], "{name}: s{r}");
            }
            for l in 0..32 {
                for &r in &vregs {
                    assert_eq!(cv[l][r], gpu.vgpr[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    // wave64 versions of the programs above: 64-bit EXEC / VCC / saved masks, and thresholds
    // past lane 32 so that both words of every mask matter
    /// v_cmp_gt_u32 vcc, 40, v0; s_and_saveexec_b64 s[0:1], vcc; s_cbranch_execz L; v_mov v1, 7;
    /// L: s_or_b64 exec, exec, s[0:1]; v_add v2, v1, 1
    const SKIPPED_IF_W64: &[I] = &[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)]),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 1)]),
        ENDPGM,
    ];
    /// s_mov_b64 s[0:1], exec; L: v_add v1, v1, 1; v_cmp_ge_u32 vcc, v1, v0;
    /// exec &= ~vcc (b64); s_cbranch_execnz L; exec |= s[0:1]  →  v1 = max(1, lane), 64 lanes
    const LEAVING_LOOP_W64: &[I] = &[
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 0)]),
        ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
        ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ENDPGM,
    ];
    /// `MEMORY` over 64 lanes: lanes below 40 take the guarded store
    const MEMORY_W64: &[I] = &[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 8), ("SOFFSET", 125)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 1), ("SADDR", 125), ("OFFSET", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 4), ("VSRC1", 2)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 2), ("SADDR", 125), ("OFFSET", 0x100)]),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 6)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 0), ("SADDR", 125), ("OFFSET", 0x200)]),
        ENDPGM,
    ];

    #[test]
    fn a_wave64_mask_left_behind_is_reported_in_both_halves() {
        // VCC from the loop's last v_cmp: lanes that left earlier kept their own bits in both words
        let (_, stale) = lower_kernel(&decode(LEAVING_LOOP_W64), true, Interp::default()).unwrap();
        assert_eq!(stale, vec![106, 107]);
    }

    /// The wave64 lowering run on the CPU and the HLSL path's native wave64 form run on a GPU
    /// with 64-lane waves give the same registers and memory.
    #[test]
    fn the_wave64_lowering_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Options};
        if !gpu_waves() || dxc::location().is_err() {
            runner::skip_test(if !gpu_waves() { "no GPU with 32-lane waves" } else { "no DXC" });
            return;
        }
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        for (name, lines) in [
            ("straight", STRAIGHT),
            ("skipped if", SKIPPED_IF_W64),
            ("leaving loop", LEAVING_LOOP_W64),
            ("uniform loop", UNIFORM_LOOP),
            ("memory", MEMORY_W64),
        ] {
            let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
            let (s, v) = state(64);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let t = translate(&bytes, &Options { wave64: true, split_wave64: false, ..Default::default() }).unwrap();
            assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
            let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap();
            let (cs, cv, cmem) = run_mem(lines, true, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let insts = decode(lines);
            let (_, stale) = lower_kernel(&insts, true, Interp::default()).unwrap();
            let mut sregs: Vec<usize> = vec![106, 107, 126, 127];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(cs[r], gpu.sgpr[r], "{name}: s{r}");
            }
            for l in 0..64 {
                for &r in &vregs {
                    assert_eq!(cv[l][r], gpu.vgpr[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    /// The split form (two hardware wave32s per console wave64, `lower_dxil::kernel_split`) run
    /// on the GPU gives the registers and memory of the wave64 lowering run on the CPU: masks
    /// across both halves, lane reads through the exchange (readfirstlane, readlane, the counts
    /// of EXEC as numbers, mbcnt), and the reads that stay in their half (DPP, permlanex16,
    /// ds_swizzle). A whole-wave read where lanes leave one by one is refused.
    #[test]
    fn the_split_form_agrees_with_the_wave64_lowering_on_the_gpu() {
        use super::super::{lower_dxil, runner};
        if !gpu_waves() {
            runner::skip_test("no GPU with 32-lane waves");
            return;
        }
        let reads = lane_reads(true);
        // reads across the halves both ways: readlane of lane 41 (the lower half needs the
        // upper's), readfirstlane with EXEC from lane 40 up (the first lane is in the upper
        // half), and a 64-bit compare mask counted as a number
        const ACROSS: &[I] = &[
            ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V), ("SRC1", 169)]),
            ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 131)]),
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]),
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 106)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 168)]),
            ("V_CMPX_LE_U32", "ENC_VOPC", &[("SRC0", V + 2), ("VSRC1", 0)]),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 1)]),
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 13), ("SSRC0", 126)]),
            ENDPGM,
        ];
        for (name, lines) in [
            ("across the halves", ACROSS),
            ("straight", STRAIGHT),
            ("skipped if", SKIPPED_IF_W64),
            ("leaving loop", LEAVING_LOOP_W64),
            ("uniform loop", UNIFORM_LOOP),
            ("memory", MEMORY_W64),
            ("lane reads", &reads[..]),
        ] {
            let insts = decode(lines);
            let (dxil, stale) = lower_dxil::kernel_split(&insts).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (s, v) = state(64);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (cs, cv, cmem) = run_mem(lines, true, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let mut sregs: Vec<usize> = vec![106, 107, 126, 127];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(gpu.sgpr[r], cs[r], "{name}: s{r}");
            }
            for l in 0..64 {
                for &r in &vregs {
                    assert_eq!(gpu.vgpr[l][r], cv[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
        // compute mode: a 64-thread group as two hardware wave32s, threads numbered by arrival
        let cs = compute_cs();
        let dxil = lower_dxil::compute_split(&decode(COMPUTE), &cs).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&gpu[..cpu.len()], &cpu[..], "compute");
        // two console waves per group: their halves pair up through #1426's arrival counters
        // and meet in its spin loop. Each thread stores its lane (`mbcnt` of EXEC: the mask
        // words exchanged), its thread id and its console wave's first thread
        // (`readfirstlane`: a lane of the other half for the upper one)
        const TWO_WAVES: &[I] = &[
            ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 135)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 3), ("VSRC1", 0)]),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 130), ("VSRC1", 4)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)]),
            ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 126), ("SRC1", 128)]),
            ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 127), ("SRC1", V + 5)]),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)]),
            ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 5), ("SRC1", 136), ("SRC2", V)]),
            ("V_LSHL_OR_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", 10), ("SRC1", 144), ("SRC2", V + 7)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 8), ("DATA", 7), ("SADDR", 125), ("OFFSET", 0)]),
            ENDPGM,
        ];
        let big = crate::gpu::recomp::CsInterface { threads: [128, 1, 1], ..compute_cs() };
        let dxil = lower_dxil::compute_split(&decode(TWO_WAVES), &big).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [2, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute_waves(TWO_WAVES, &big, true, 2, &[100, 0], &mem);
        assert_eq!(&gpu[..256], &cpu[..256], "two console waves per group");
        assert!(cpu[..256].iter().any(|&w| w >> 16 == 64), "the CPU run's second wave starts at thread 64");
        // the waterfall's readfirstlane is in a block lanes leave one by one: native wave64
        // takes it, the split form cannot meet the other half there
        let insts = decode(&waterfall(true));
        assert!(lower_kernel(&insts, true, Interp::default()).is_ok());
        let e = lower_dxil::kernel_split(&insts).unwrap_err();
        assert!(e.contains("split form"), "{e}");
    }

    #[test]
    fn scalar_and_vector_memory_with_a_guarded_store() {
        let (s, v) = state(32);
        let (s, v, mem) = run_mem(MEMORY, false, &s, &v, &memory_image());
        assert_eq!((s[4], s[5]), (1002, 1003));
        for l in 0..32 {
            assert_eq!(v[l][2], 2002 + l as u32, "lane {l}");
            assert_eq!(mem[64 + l], if l < 16 { 2002 + l as u32 } else { 1064 + l as u32 }, "guarded store, lane {l}");
            assert_eq!(mem[128 + l], l as u32, "store, lane {l}");
        }
        // the store under the narrowed EXEC is the guarded one
        let insts = decode(MEMORY);
        let sum = lane_masks::analyze(&insts, false).unwrap();
        assert_eq!(sum.guarded.iter().filter(|&&pc| insts.iter().any(|d| d.pc == pc && is_store(d))).count(), 1);
    }

    #[test]
    fn a_compute_program_takes_its_group_and_thread_ids() {
        let mem = run_compute(COMPUTE, &compute_cs(), 3, &[100, 0], &vec![0u32; 256]);
        for g in 0..3u32 {
            for t in 0..64u32 {
                assert_eq!(mem[(64 * g + t) as usize], 3 * t + 100 + g, "group {g} thread {t}");
            }
        }
    }

    #[test]
    fn a_2d_or_3d_group_numbers_its_threads_as_the_console_does() {
        // a wave takes consecutive threads, lane l the l-th: x fastest, then y, then z
        for threads in ID_SHAPES {
            for wave64 in [false, true] {
                let lanes = if wave64 { 64 } else { 32 };
                let mem = run_compute_waves(&ids_program(threads, wave64), &ids_cs(threads), wave64, 2, &[100, 0], &[0u32; 256]);
                for g in 0..2u32 {
                    for t in 0..64u32 {
                        let (x, y, z) = (t % threads[0], t / threads[0] % threads[1], t / (threads[0] * threads[1]));
                        let want = x | y << 8 | z << 16 | (t % lanes) << 24;
                        assert_eq!(mem[(64 * g + t) as usize], want, "{threads:?} wave64 {wave64}: group {g} thread {t}");
                    }
                }
            }
        }
    }

    /// The same through the HLSL path on the GPU, which numbers the threads by arrival.
    #[test]
    fn a_2d_or_3d_group_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Mode, Options};
        if !gpu_waves() || dxc::location().is_err() {
            runner::skip_test(if !gpu_waves() { "no GPU with 32-lane waves" } else { "no DXC" });
            return;
        }
        for wave64 in [false, true] {
            if wave64 && runner::wave64_supported() != Ok(true) {
                runner::skip_test("no wave64");
                return;
            }
            for threads in ID_SHAPES {
                let lines = ids_program(threads, wave64);
                let cs = ids_cs(threads);
                let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
                let t = translate(&bytes, &Options { mode: Mode::Compute(cs.clone()), wave64, ..Default::default() }).unwrap();
                assert!(t.complete(), "{:?}", t.unsupported_why);
                let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
                let mem = vec![0u32; 256];
                let gpu = runner::dispatch(&dxil, [2, 1, 1], &[100, 0], &mem).unwrap();
                let cpu = run_compute_waves(&lines, &cs, wave64, 2, &[100, 0], &mem);
                assert_eq!(&gpu[..128], &cpu[..128], "{threads:?} wave64 {wave64}");
            }
        }
    }

    /// The compute program through the HLSL path on the GPU and through the lowering on the CPU.
    #[test]
    fn a_compute_program_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Mode, Options};
        if !gpu_waves() || dxc::location().is_err() {
            runner::skip_test(if !gpu_waves() { "no GPU with 32-lane waves" } else { "no DXC" });
            return;
        }
        let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        let cs = compute_cs();
        let t = translate(&bytes, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&gpu[..cpu.len()], &cpu[..]);
    }

    #[test]
    fn a_mask_left_behind_by_lanes_that_skipped_its_write() {
        // VCC written inside the loop that lanes leave one by one: the console's last write left
        // the earlier leavers' bits 0, each lane here kept its own. Stored at the end: reported.
        let insts = decode(LEAVING_LOOP);
        let (_, stale) = lower_kernel(&insts, false, Interp::default()).unwrap();
        assert_eq!(stale, vec![106]);
        // read after the loop (v_cndmask on VCC): refused
        let mut lines = LEAVING_LOOP.to_vec();
        lines.insert(6, ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 128), ("VSRC1", 0)]));
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes were elsewhere"), "{err}");
        // a mask written before the lanes split is read after they meet: fine
        let insts = decode(SKIPPED_IF);
        assert_eq!(lower_kernel(&insts, false, Interp::default()).unwrap().1, Vec::<u8>::new());
        // an older mask copied in the arm some lanes skip (s_mov s4, s0), read after they meet:
        // the copy is the arm's write, not the older value's (found by MiguelMachado-dev)
        let mut lines = SKIPPED_IF.to_vec();
        lines[2] = ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)]);
        lines.insert(3, ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 0)]));
        lines.insert(6, ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 128), ("SRC1", 129), ("SRC2", 4)]));
        // (s4 holds a mask before the lanes split, so the join is mask against mask)
        lines.insert(1, ("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]));
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes were elsewhere"), "{err}");
    }

    #[test]
    fn a_scalar_counted_in_a_loop_lanes_leave_is_not_each_lanes_own() {
        // s_add_u32 s1, s1, 1 in the loop lanes leave one by one: the console counts the wave's
        // passes, each lane here only its own (a waterfall loop's key is the same case)
        // s_mov s0, exec; s1 = 0; L: v_add v1, v1, 1; s1 += 1; v_cmp_ge vcc, v1, v0;
        // exec &= ~vcc; s_cbranch_execnz L; exec |= s0
        let mut lines: Vec<I> = vec![
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 128)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)]),
            ("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 0)]),
            ("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
            ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFB)]),
            ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            ENDPGM,
        ];
        let (_, stale) = lower_kernel(&decode(&lines), false, Interp::default()).unwrap();
        assert!(stale.contains(&1), "{stale:?}");
        // read after the loop: refused
        lines.insert(8, ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 1)]));
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes were elsewhere"), "{err}");
    }

    #[test]
    fn vop3_operand_selects_are_refused() {
        let op_sel: &[I] = &[("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", V), ("OP_SEL", 1)]), ENDPGM];
        let err = lower_kernel(&decode(op_sel), false, Interp::default()).err().expect("refused");
        assert!(err.contains("modifiers"), "{err}");
        let plain: &[I] = &[("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", V)]), ENDPGM];
        assert!(lower_kernel(&decode(plain), false, Interp::default()).is_ok());
    }

    #[test]
    fn a_mask_read_as_a_number_is_refused() {
        // s_bcnt1-like use: s_add_u32 s2, vcc_lo, 0 after a v_cmp
        let insts = decode(&[
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 106), ("SSRC1", 128)]),
            ENDPGM,
        ]);
        assert!(lower_kernel(&insts, false, Interp::default()).is_err());
    }

    /// Cross-lane reads with EXEC narrowed to the lanes above 4 (`v_cmpx`): readfirstlane,
    /// the counts of EXEC as a number, mbcnt, readlane / writelane, DPP16 / DPP8 sources,
    /// permlanex16 and ds_swizzle.
    fn lane_reads(wave64: bool) -> Vec<I> {
        let mut p: Vec<I> = vec![
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 132)]),
            ("V_CMPX_LT_U32", "ENC_VOPC", &[("SRC0", V + 2), ("VSRC1", 0)]),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)]),
            ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 126), ("SRC1", 128)]),
            ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", 131)]),
            ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 4), ("SRC1", 137)]),
            ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 2), ("SRC1", 130)]),
            // row_shr:1 with bound_ctrl; quad_perm [3,2,1,0] on rows 1 and 3; DPP8 reversed, added
            ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)]),
            ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 7), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0x1B), ("ROW_MASK", 0xA), ("BANK_MASK", 0xF)]),
            ("V_ADD_NC_U32", "VOP2_VOP_DPP8", &[("VDST", 8), ("SRC0", 0xE9), ("VSRC0", 4), ("VSRC1", 0), ("LANE_SEL_0", 7), ("LANE_SEL_1", 6), ("LANE_SEL_2", 5), ("LANE_SEL_3", 4), ("LANE_SEL_4", 3), ("LANE_SEL_5", 2), ("LANE_SEL_6", 1), ("LANE_SEL_7", 0)]),
            ("S_MOVK_I32", "ENC_SOPK", &[("SDST", 5), ("SIMM16", 0x3210)]),
            ("S_MOVK_I32", "ENC_SOPK", &[("SDST", 6), ("SIMM16", 0x7F54)]),
            ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 4), ("SRC1", 5), ("SRC2", 6)]),
            // xor 1 (and 0x1F, or 0, xor 1); quad mode [3,2,1,0]
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 4), ("OFFSET0", 0x1F), ("OFFSET1", 0x04)]),
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 4), ("OFFSET0", 0x1B), ("OFFSET1", 0x80)]),
        ];
        if wave64 {
            p.extend_from_slice(&[
                ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 127), ("SRC1", V + 3)]),
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 126)]),
                ("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 126)]),
            ]);
        } else {
            p.extend_from_slice(&[
                ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 126)]),
                ("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 126)]),
            ]);
        }
        p.push(ENDPGM);
        p
    }

    /// A waterfall loop: each pass takes the first lane's key (readfirstlane), the lanes with
    /// that key do the work and leave; the readfirstlane is in a block lanes leave one by one.
    fn waterfall(wave64: bool) -> Vec<I> {
        let w: [I; 4] = if wave64 {
            [
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
                ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
                ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 4)]),
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
            ]
        } else {
            [
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
                ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
                ("S_XOR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 4)]),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
            ]
        };
        vec![
            ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 131), ("VSRC1", 0)]),
            w[0],
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 2)]),
            ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 2)]),
            w[1],
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 2), ("VSRC1", 3)]),
            w[2],
            ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFA)]),
            w[3],
            ENDPGM,
        ]
    }

    /// The lowering (on the CPU) and the HLSL path (on the GPU) give the same registers.
    fn agrees_with_hlsl(name: &str, lines: &[I], wave64: bool, skip: &[usize]) {
        use super::super::{dxc, runner, translate, Options};
        let lanes = if wave64 { 64 } else { 32 };
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        let (s, v) = state(lanes);
        let mut st = runner::KernelState::default();
        st.sgpr.copy_from_slice(&s);
        for (l, regs) in v.iter().enumerate() {
            st.vgpr[l][..regs.len()].copy_from_slice(regs);
        }
        let t = translate(&bytes, &Options { wave64, ..Default::default() }).unwrap();
        assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let gpu = runner::run(&dxil, &st, &[]).unwrap();
        let (cs, cv) = run(lines, wave64, &s, &v);
        let insts = decode(lines);
        let (_, stale) = lower_kernel(&insts, wave64, Interp::default()).unwrap();
        let mut sregs: Vec<usize> = vec![106, 126];
        let mut vregs: Vec<usize> = Vec::new();
        for d in &insts {
            let (r, w) = rdna2::sgpr_uses(d);
            sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
            let (r, w) = rdna2::vgpr_uses(d);
            vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
        }
        if wave64 {
            sregs.extend([107, 127]);
        }
        // (a stale wave64 mask is reported by its first register; the pair's second is stale too)
        let stale_at = |r: usize| stale.contains(&(r as u8)) || (wave64 && r > 0 && r % 2 == 1 && stale.contains(&(r as u8 - 1)));
        for &r in sregs.iter().filter(|&&r| !stale_at(r) && !skip.contains(&r)) {
            assert_eq!(cs[r], gpu.sgpr[r], "{name}: s{r}");
        }
        for l in 0..lanes {
            for &r in &vregs {
                assert_eq!(cv[l][r], gpu.vgpr[l][r], "{name}: lane {l} v{r}");
            }
        }
    }

    #[test]
    fn cross_lane_reads_on_the_cpu() {
        for wave64 in [false, true] {
            let lanes = if wave64 { 64 } else { 32 };
            let (s, v) = state(lanes);
            let (s, v) = run(&lane_reads(wave64), wave64, &s, &v);
            assert_eq!(s[2], 5, "readfirstlane: the first lane in EXEC");
            assert_eq!(s[3], lanes as u32 - 5, "s_bcnt1 of EXEC");
            assert_eq!(s[4], 27, "readlane 9 of 3 * lane");
            assert_eq!(s[7], 5, "s_ff1 of EXEC");
            for (l, r) in v.iter().enumerate() {
                let (l32, active) = (l as u32, l > 4);
                assert_eq!(r[3], if active { l32 - 5 } else { 0 }, "lane {l}: mbcnt");
                assert_eq!(r[5], if l == 2 { 5 } else { 0 }, "lane {l}: writelane, whatever EXEC holds");
                // row_shr:1, bound_ctrl: 0 at a row's first lane or from a lane outside EXEC
                let want = if !active || l % 16 == 0 || l - 1 <= 4 { 0 } else { 3 * (l32 - 1) };
                assert_eq!(r[6], want, "lane {l}: DPP16 row_shr");
                let rev = (l32 & !7) | (7 - (l32 & 7));
                let want = if active { l32 + if rev > 4 { 3 * rev } else { 0 } } else { 0 };
                assert_eq!(r[8], want, "lane {l}: DPP8");
                let x = l32 ^ 1;
                assert_eq!(r[10], if active && x > 4 { 3 * x } else { 0 }, "lane {l}: ds_swizzle xor 1");
            }
        }
    }

    #[test]
    fn a_waterfall_loop_reads_the_first_lane_of_each_pass() {
        for wave64 in [false, true] {
            let lanes = if wave64 { 64 } else { 32 };
            let (s, v) = state(lanes);
            let (_, v) = run(&waterfall(wave64), wave64, &s, &v);
            for (l, r) in v.iter().enumerate() {
                assert_eq!(r[3], l as u32 & 3, "lane {l}");
            }
        }
        // a lane of any lane, where some lanes have left: refused
        let mut lines = waterfall(false);
        lines.insert(3, ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 2), ("SRC1", 129)]));
        lines[8] = ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFF8)]);
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes are elsewhere"), "{err}");
    }

    #[test]
    fn cross_lane_reads_agree_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner};
        if !gpu_waves() || dxc::location().is_err() {
            runner::skip_test(if !gpu_waves() { "no GPU with 32-lane waves" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("lane reads, wave32", &lane_reads(false), false, &[]);
        // s2 and the saved EXEC s4 after the loop: the last pass's on the console; each lane's own
        // last pass's in the lowering (written where lanes leave one by one; not refused yet)
        agrees_with_hlsl("waterfall, wave32", &waterfall(false), false, &[2, 4]);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        agrees_with_hlsl("lane reads, wave64", &lane_reads(true), true, &[]);
        agrees_with_hlsl("waterfall, wave64", &waterfall(true), true, &[2, 4, 5]);
    }

    /// Guarded straight-line run under narrowed EXEC (no skip branch): VALU ops, a guarded load,
    /// `s_waitcnt`, and two guarded stores all coalesce into one per-lane branch with 0 selects.
    pub(crate) fn coalesced_run(wave64: bool) -> Vec<I> {
        let (cmp, save, restore): (&'static [(&'static str, u64)], &str, &str) = if wave64 {
            (&[("SRC0", 168), ("VSRC1", 0)], "S_AND_SAVEEXEC_B64", "S_OR_B64")
        } else {
            (&[("SRC0", 144), ("VSRC1", 0)], "S_AND_SAVEEXEC_B32", "S_OR_B32")
        };
        vec![
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 130), ("VSRC1", 0)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)]),
            ("V_CMP_GT_U32", "ENC_VOPC", cmp),
            (save, "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 138), ("VSRC1", 0)]),
            ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 2), ("SRC1", 131)]),
            ("V_XOR_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 3), ("VSRC1", 0)]),
            ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 8), ("SADDR", 125), ("OFFSET", 0)]),
            ("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 5), ("VSRC1", 2)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 8), ("DATA", 4), ("SADDR", 125), ("OFFSET", 0x100)]),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 8), ("DATA", 6), ("SADDR", 125), ("OFFSET", 0x200)]),
            (restore, "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 4), ("VSRC1", 6)]),
            ENDPGM,
        ]
    }

    /// Guarded runs separated by a scalar write, a `v_cmp`, and a cross-lane `v_readlane`, plus
    /// `v_cndmask` and a twice-written VGPR inside coalesced runs.
    pub(crate) const COALESCED_BREAKS: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 133)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 135), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 0)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 2), ("SSRC1", 131)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 2), ("VSRC1", 1)]),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 3), ("VSRC1", 2)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 136), ("VSRC1", 0)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 3), ("VSRC1", 4)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 129), ("VSRC1", 5)]),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 1), ("SRC1", 131)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 3), ("VSRC1", 6)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 130), ("VSRC1", 7)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ENDPGM,
    ];

    /// Two coalesced runs separated by a scalar write inside an already-`partial` block (arm of a
    /// skipped `if`), reading SGPR `s2` and `vcc` written earlier in that same console block.
    pub(crate) const COALESCED_IN_PARTIAL: &[I] = &[
        ("S_AND_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 126), ("SSRC1", 128)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 152), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 10)]),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 139)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 140), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 106)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 2), ("VSRC1", 0)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 2), ("SSRC1", 129)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 2), ("VSRC1", 2)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 129), ("VSRC1", 3)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 2), ("VSRC1", 4)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 129), ("VSRC1", 5)]),
        ENDPGM,
    ];

    /// A 4-iteration uniform loop whose body narrows EXEC (no skip branch) around a coalesced run.
    pub(crate) const COALESCED_IN_LOOP: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 128)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 2), ("SSRC1", 129)]),
        ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 2), ("SSRC1", 132)]),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF9)]),
        ENDPGM,
    ];

    /// Guarded run of f32 instructions lowered through `Ir::foreign` (`Lw::inst_guarded`).
    pub(crate) const COALESCED_F32_RUN: &[I] = &[
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 242), ("VSRC1", 1)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 244), ("VSRC1", 2)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 3), ("SRC1", 240), ("SRC2", 242)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 242), ("VSRC1", 4)]),
        ENDPGM,
    ];

    #[test]
    fn guarded_run_coalesces_into_one_branch_with_no_selects() {
        for (wave64, lanes, cut) in [(false, 32usize, 16usize), (true, 64usize, 40usize)] {
            let lines = coalesced_run(wave64);
            let insts = decode(&lines);
            let sum = lane_masks::analyze(&insts, wave64).unwrap();
            assert_eq!(sum.guarded.len(), 7, "wave64={wave64}");
            let (ir, stale) = lower_kernel(&insts, wave64, Interp::default()).unwrap();
            assert!(stale.is_empty(), "{stale:?}");
            // wave64 has 1 prologue select to pick EXEC's lo/hi dword; 0 per-write selects in either mode
            assert_eq!(ir.selects(), usize::from(wave64), "wave64={wave64}: coalesced run removes all per-write selects");
            assert_eq!(ir.br_ifs(), 3, "wave64={wave64}: entry + 1 coalesced run + exit");

            let (s, v) = state(lanes);
            let (s, v, mem) = run_mem(&lines, wave64, &s, &v, &memory_image());
            assert_eq!(s[126], u32::MAX);
            if wave64 {
                assert_eq!(s[127], u32::MAX);
            }
            for l in 0..lanes {
                let l32 = l as u32;
                if l < cut {
                    let (v2, v5) = (l32 + 10, 1000 + l32);
                    let (v3, v6) = (v2 * 3, v5 + v2);
                    let v4 = v3 ^ l32;
                    assert_eq!((v[l][2], v[l][3], v[l][4], v[l][5], v[l][6], v[l][7]), (v2, v3, v4, v5, v6, v4 + v6), "wave64={wave64} lane {l}");
                    assert_eq!((mem[64 + l], mem[128 + l]), (v4, v6), "wave64={wave64} lane {l} mem");
                } else {
                    assert_eq!((v[l][2], v[l][3], v[l][4], v[l][5], v[l][6], v[l][7]), (0, 0, 0, 0, 0, 0), "wave64={wave64} lane {l}");
                    assert_eq!((mem[64 + l], mem[128 + l]), (1064 + l32, 1128 + l32), "wave64={wave64} lane {l} mem");
                }
            }
        }

        // a single guarded VALU or load keeps its select rather than adding a branch
        for single_op in [
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)][..]),
            ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 1), ("ADDR", 0), ("SADDR", 125), ("OFFSET", 0)][..]),
        ] {
            let single = decode(&[
                ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
                ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
                single_op,
                ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
                ENDPGM,
            ]);
            let (ir, _) = lower_kernel(&single, false, Interp::default()).unwrap();
            assert_eq!((ir.selects(), ir.br_ifs()), (1, 2), "{}", single_op.0);
        }
    }

    #[test]
    fn guard_coalescing_breaks_at_scalar_compare_and_cross_lane_ops() {
        let (ir, stale) = lower_kernel(&decode(COALESCED_BREAKS), false, Interp::default()).unwrap();
        assert!(stale.is_empty(), "{stale:?}");
        assert_eq!((ir.selects(), ir.br_ifs()), (1, 6)); // only v_cndmask's select; 4 runs + entry/exit
        let (s, v) = state(32);
        let (s, v) = run(COALESCED_BREAKS, false, &s, &v);
        assert_eq!((s[2], s[3], s[106], s[126]), (8, 10, 0xFF, u32::MAX));
        for l in 0..32usize {
            let l32 = l as u32;
            let want = if l < 16 {
                let (v1, v3) = (l32 + 7, l32 + 15);
                let v4 = v3 ^ (v1 + l32);
                (v1, v4, 10 + (if l < 8 { v4 } else { v3 }) + 3)
            } else {
                (0, 0, 0)
            };
            assert_eq!((v[l][1], v[l][4], v[l][7]), want, "lane {l}");
        }

        // two coalesced runs inside a partial block (with a scalar write in between) stay non-stale
        let (s, v) = state(32);
        let (_, v) = run(COALESCED_IN_PARTIAL, false, &s, &v);
        for l in 0..32usize {
            let l32 = l as u32;
            let v4 = if l < 12 { l32 + 24 } else { 0 };
            let v5 = if l < 24 { 12 + v4 } else { 0 };
            assert_eq!((v[l][4], v[l][5], v[l][6]), (v4, v5, v5 + 1), "partial lane {l}");
        }

        // coalesced run inside a loop: 0 selects, SSA phis carry loop values cleanly
        let (ir, _) = lower_kernel(&decode(COALESCED_IN_LOOP), false, Interp::default()).unwrap();
        assert_eq!(ir.selects(), 0);
        let (s, v) = state(32);
        let (s, v) = run(COALESCED_IN_LOOP, false, &s, &v);
        assert_eq!((s[2], s[126]), (4, u32::MAX));
        for l in 0..32 {
            assert_eq!((v[l][1], v[l][2]), if l < 16 { (4, 10) } else { (0, 0) }, "loop lane {l}");
        }

        // guarded VALU carry-out (VOP3_SDST_ENC) and cross-lane ds_swizzle split runs
        let cs = decode(&[
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
            ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 0)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 1)]),
            ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 3), ("SDST", 106), ("SRC0", V + 1), ("SRC1", V + 2)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 129), ("VSRC1", 3)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 130), ("VSRC1", 4)]),
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 5), ("OFFSET0", 0x1F), ("OFFSET1", 0x04)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 129), ("VSRC1", 6)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 130), ("VSRC1", 7)]),
            ENDPGM,
        ]);
        let sum = lane_masks::analyze(&cs, false).unwrap();
        let guarded: HashSet<u64> = sum.guarded.iter().copied().collect();
        assert_eq!(coalesced_runs(&cs, &sum.blocks[0].insts, &guarded), vec![2..4, 5..7, 8..10]);
    }

    #[test]
    fn coalesced_guarded_runs_agree_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Options};
        if !gpu_waves() || dxc::location().is_err() {
            runner::skip_test(if !gpu_waves() { "no GPU with 32-lane waves" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("coalesced breaks, wave32", COALESCED_BREAKS, false, &[]);
        agrees_with_hlsl("coalesced in partial, wave32", COALESCED_IN_PARTIAL, false, &[]);
        agrees_with_hlsl("coalesced in loop, wave32", COALESCED_IN_LOOP, false, &[]);
        let check_mem = |wave64: bool| {
            let (lines, lanes) = (coalesced_run(wave64), if wave64 { 64 } else { 32 });
            let bytes: Vec<u8> = lines.iter().flat_map(encode_line).collect();
            let (s, v) = state(lanes);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let t = translate(&bytes, &Options { wave64, split_wave64: false, ..Default::default() }).unwrap();
            assert!(t.complete(), "{:?}", t.unsupported_why);
            let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap();
            let (cs, cv, cmem) = run_mem(&lines, wave64, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "wave64={wave64}: memory");
            for l in 0..lanes {
                for r in 0..8 {
                    assert_eq!(cv[l][r], gpu.vgpr[l][r], "wave64={wave64}: lane {l} v{r}");
                }
            }
            assert_eq!(cs[126], gpu.sgpr[126]);
            if wave64 {
                assert_eq!(cs[127], gpu.sgpr[127]);
            }
        };
        check_mem(false);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        check_mem(true);
    }
}
