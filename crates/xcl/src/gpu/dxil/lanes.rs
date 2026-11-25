//! Lane masks and cross-lane values.
//!
//! Every lane mask (EXEC, VCC, compare results, carries) is a wave-uniform
//! value built with `WaveActiveBallot`; a lane tests its own bit
//! ([`Lw::lanebit`], cached per block). In the split form (a wave64 console
//! program on a GPU with 32-lane waves only, `docs/wave64-on-wave32.md`) the
//! two hardware halves of the console wave are the whole thread group and
//! meet at a group barrier to exchange one word each: ballot words and
//! `v_readlane` values. Lane reads the ISA keeps inside a 32-lane half (DPP,
//! `ds_swizzle`, permlane) need no exchange.

use super::lower::{Lw, Partner};
use super::module::{BinOp, Pred, RmwOp, Value};
use super::ops::barrier;

impl Lw {
    /// Groupshared storage, created in the entry block before anything else:
    /// the program's LDS (`lds_words` > 0) and, in the split form, the halves'
    /// exchange words — in the top of the LDS block when there is one (D3D12
    /// allows 32 KB, the whole LDS), as the HLSL kernel lays it out.
    pub fn setup_storage(&mut self, lds: bool, slot: u32) {
        let i32 = self.s.m.i32();
        self.xslot = slot.max(1);
        if lds {
            let g = self.s.groupshared("xcl_lds", i32, 8192);
            self.lds = Some(g);
            if self.split {
                // HLSL: exchange base = 8192 - slots - counters - 1 (slots = 4 · slot per console
                // wave, 2 counters each); the program's LDS stops below it
                let slots = 4 * self.xslot * self.waves;
                let base = 8192 - slots - 2 * self.waves - 1;
                self.lds_top = base - 1;
                self.xs = Some((g, base));
                self.xc = Some((g, base + slots));
                self.xw = Some((g, 8191));
            } else {
                self.lds_top = 8191;
            }
        } else if self.split {
            let w = self.s.groupshared("xcl_xw", i32, 1);
            self.xw = Some((w, 0));
            if self.waves > 1 {
                let x = self.s.groupshared("xcl_xs", i32, (4 * self.xslot * self.waves) as u64);
                self.xs = Some((x, 0));
                let c = self.s.groupshared("xcl_xc", i32, 2 * self.waves as u64);
                self.xc = Some((c, 0));
            }
        }
    }

    /// Lane setup at the start of the entry block (after [`Lw::setup_storage`]).
    pub fn setup_lanes(&mut self) {
        let lane = if self.split {
            // each hardware wave takes the next 32 console lanes as it arrives (the PC need
            // not form waves from consecutive thread indices): a groupshared counter
            let (xw, at) = self.xw.expect("split storage");
            let z = self.c(0);
            let ai = self.c(at);
            let p = self.s.gs_ptr(xw, ai);
            self.s.m.store(p, z, 4);
            self.s.barrier(barrier::GROUP_SYNC);
            let first = self.s.wave_is_first_lane();
            let k = self.selc(first, 32, z);
            let got = self.s.m.atomic_rmw(RmwOp::Add, p, k);
            let base = self.s.wave_read_lane_first(got);
            let hw = self.s.wave_lane_index();
            self.add(base, hw)
        } else if self.wave64 {
            self.s.flattened_thread_id_in_group()
        } else {
            self.s.wave_lane_index()
        };
        self.lane = lane;
        self.lane31 = self.andc(lane, 31);
        self.upper = self.cmpc(Pred::UGe, lane, 32);
        self.half = self.lshrc(lane, 5);
    }

    /// Compute-mode lane setup (the HLSL compute entry's): the console thread index in the
    /// group (`gi` as the program sees it) and the thread id components; also sets the
    /// console lane. Split wave64: hardware waves take the next 32 console threads as they
    /// arrive (`xcl_x_thread`); 2-D / 3-D groups in native waves: the next `lanes` threads
    /// of full waves (`XCL_WT`).
    pub fn setup_lanes_compute(&mut self, threads: [u32; 3], lds_top_word: bool) -> (Value, [Value; 3]) {
        let [x, y, z] = threads;
        let total = x * y * z;
        let lanes: u32 = if self.wave64 { 64 } else { 32 };
        let zero = self.c(0);
        let tid = if self.split {
            let (xw, at) = self.xw.expect("split storage");
            let ai = self.c(at);
            let p = self.s.gs_ptr(xw, ai);
            self.s.m.store(p, zero, 4);
            if let Some((xc, cb)) = self.xc {
                // the arrival counters start at 0 (`if (gi < 2 * waves) XCL_XC(gi) = 0`)
                let gi = self.s.flattened_thread_id_in_group();
                let last = self.c(2 * self.waves - 1);
                let i = self.umin(gi, last);
                let i = self.addc(i, cb);
                let q = self.s.gs_ptr(xc, i);
                self.s.m.store(q, zero, 4);
            }
            self.s.barrier(barrier::GROUP_SYNC);
            let t = self.s.m.i1c(true);
            let n = self.s.wave_all_bit_count(t);
            let full = self.cmpc(Pred::Eq, n, 32);
            let first = self.s.wave_is_first_lane();
            let take = self.land(first, full);
            let k = self.selc(take, 32, zero);
            let got = self.s.m.atomic_rmw(RmwOp::Add, p, k);
            let dflt = self.c(total & !31);
            let base = self.sel(take, got, dflt);
            let base = self.s.wave_read_lane_first(base);
            let t = self.s.m.i1c(true);
            let rank = self.s.wave_prefix_bit_count(t);
            self.add(base, rank)
        } else if y * z > 1 {
            let i32 = self.s.m.i32();
            let (g, at) = match (self.lds, lds_top_word) {
                (Some(l), true) => (l, 8191),
                _ => (self.s.groupshared("xcl_wt", i32, 1), 0),
            };
            let ai = self.c(at);
            let p = self.s.gs_ptr(g, ai);
            self.s.m.store(p, zero, 4);
            self.s.barrier(barrier::GROUP_SYNC);
            let hw = self.s.wave_lane_index();
            let t = self.s.m.i1c(true);
            let n = self.s.wave_all_bit_count(t);
            let full = self.cmpc(Pred::Eq, n, lanes);
            let first = self.s.wave_is_first_lane();
            let take = self.land(first, full);
            let k = self.selc(take, lanes, zero);
            let got = self.s.m.atomic_rmw(RmwOp::Add, p, k);
            let dflt = self.c(total & !(lanes - 1));
            let base = self.sel(take, got, dflt);
            let base = self.s.wave_read_lane_first(base);
            let tid = self.add(base, hw);
            self.s.barrier(barrier::GROUP_SYNC);
            tid
        } else {
            self.s.flattened_thread_id_in_group()
        };
        if self.split {
            self.cw = self.lshrc(tid, 6);
            // lanes of the partner half: threads > ((tid & ~31) ^ 32) ? min(32, threads - ..) : 0
            let other = self.andc(tid, !31);
            let k32 = self.c(32);
            let other = self.xor(other, k32);
            let t = self.c(total);
            let has = self.cmp(Pred::UGt, t, other);
            let left = self.sub(t, other);
            let n = self.umin(left, k32);
            let z = self.c(0);
            self.xn = self.sel(has, n, z);
        }
        let lane = if self.split {
            self.andc(tid, 63)
        } else {
            self.s.wave_lane_index()
        };
        self.lane = lane;
        self.lane31 = self.andc(lane, 31);
        self.upper = self.cmpc(Pred::UGe, lane, 32);
        self.half = self.lshrc(lane, 5);
        let gtid = if self.split || y * z > 1 {
            let kx = self.c(x);
            let kxy = self.c(x * y);
            let ky = self.c(y);
            let gx = self.bin(BinOp::URem, tid, kx);
            let q = self.bin(BinOp::UDiv, tid, kx);
            let gy = self.bin(BinOp::URem, q, ky);
            let gz = self.bin(BinOp::UDiv, tid, kxy);
            [gx, gy, gz]
        } else {
            [self.s.thread_id_in_group(0), self.s.thread_id_in_group(1), self.s.thread_id_in_group(2)]
        };
        (tid, gtid)
    }

    /// This lane's bit of the mask (lo, hi), cached per block.
    pub fn lanebit(&mut self, lo: Value, hi: Value) -> Value {
        let key = (self.cur(), lo, hi);
        if let Some(&v) = self.exec_cache.get(&key) {
            return v;
        }
        let w = if self.wave64 {
            let u = self.upper;
            self.sel(u, hi, lo)
        } else {
            lo
        };
        let l = self.lane31;
        let sh = self.bin(BinOp::LShr, w, l);
        let b = self.andc(sh, 1);
        let v = self.nz(b);
        self.exec_cache.insert(key, v);
        v
    }

    /// The bit of lane `l` (uniform or per-lane i32) of the mask (lo, hi).
    pub fn lanebit_at(&mut self, lo: Value, hi: Value, l: Value) -> Value {
        let w = if self.wave64 {
            let up = self.cmpc(Pred::UGe, l, 32);
            self.sel(up, hi, lo)
        } else {
            lo
        };
        let l31 = self.andc(l, 31);
        let sh = self.bin(BinOp::LShr, w, l31);
        let b = self.andc(sh, 1);
        self.nz(b)
    }

    /// Group barrier for the split form's exchanges.
    fn meet(&mut self) {
        self.s.barrier(barrier::GROUP_SYNC);
    }

    /// Split form: this half gives `vals` (uniform in the half), gets the other half's.
    pub fn exchange_n(&mut self, vals: &[Value]) -> Vec<Value> {
        match self.partner.clone() {
            Partner::Present => {}
            Partner::Spin => return self.exchange_spin(vals),
            Partner::Missing => return vals.iter().map(|_| self.c(0)).collect(),
            Partner::Unsupported(why) => {
                self.exchange_error.get_or_insert_with(|| format!("pc {:#x}: wave64 exchange: {why}", self.pc));
                return vals.iter().map(|_| self.c(0)).collect();
            }
        }
        let (xs, base) = self.xs_storage();
        let slot = self.xslot;
        assert!(vals.len() as u32 <= slot, "exchange of {} words, slot {slot}", vals.len());
        let h = self.half;
        let k = self.c(slot);
        let mine = self.mul(h, k);
        let mine = self.addc(mine, base);
        for (i, &v) in vals.iter().enumerate() {
            let at = self.addc(mine, i as u32);
            let p = self.s.gs_ptr(xs, at);
            self.s.m.store(p, v, 4);
        }
        self.meet();
        let one = self.c(1);
        let o = self.xor(h, one);
        let theirs = self.mul(o, k);
        let theirs = self.addc(theirs, base);
        let mut out = Vec::new();
        for i in 0..vals.len() {
            let at = self.addc(theirs, i as u32);
            let q = self.s.gs_ptr(xs, at);
            out.push(self.s.m.load(q, 4));
        }
        self.meet();
        out
    }

    /// The spin exchange (several console waves per group): the HLSL `xcl_x_meet` with
    /// `XCL_X_SPIN`. Each half stores its words in its slot (double-buffered by the parity
    /// of the exchange count), fences, adds its lane count to its arrival counter and waits
    /// (one lane, bounded) for the partner's counter; a partner that never arrives marks
    /// the wave dead and its exchanges give 0 from then on.
    fn exchange_spin(&mut self, vals: &[Value]) -> Vec<Value> {
        let stores: Vec<(Value, Value)> = vals.iter().enumerate().map(|(i, &v)| (self.c(i as u32), v)).collect();
        let loads: Vec<Value> = (0..vals.len()).map(|i| self.c(i as u32)).collect();
        self.spin_exchange(&stores, &loads)
    }

    /// The spin exchange with explicit slot offsets: this half stores `(offset, value)`
    /// pairs (per lane allowed), then reads the partner's slot at `loads`.
    fn spin_exchange(&mut self, stores: &[(Value, Value)], loads: &[Value]) -> Vec<Value> {
        use super::lower::{VAR_XDEAD, VAR_XK};
        let (xs, base) = self.xs.expect("spin slots");
        let (xc, cbase) = self.xc.expect("spin counters");
        let slot = self.xslot;
        let k = self.rd(VAR_XK);
        let k1 = self.addc(k, 1);
        self.wr(VAR_XK, k1);
        let (cw, h, xn) = (self.cw, self.half, self.xn);
        let par = self.andc(k, 1);
        let at = |lw: &mut Lw, half: Value| -> Value {
            let x = lw.shlc(cw, 1);
            let x = lw.add(x, half);
            let x = lw.shlc(x, 1);
            let x = lw.add(x, par);
            let ks = lw.c(slot);
            let x = lw.mul(x, ks);
            lw.addc(x, base)
        };
        let mine = at(self, h);
        for &(o, v) in stores {
            let a = self.add(mine, o);
            let p = self.s.gs_ptr(xs, a);
            self.s.m.store(p, v, 4);
        }
        let met = self.tmp_var();
        let z = self.c(0);
        self.wr(met, z);
        let dead = self.rd(VAR_XDEAD);
        let alive = self.cmpc(Pred::Eq, dead, 0);
        let has = self.cmpc(Pred::Ne, xn, 0);
        let go = self.land(alive, has);
        let _ = self.guarded(go, &mut |lw| {
            lw.s.barrier(barrier::TGSM_FENCE);
            let t = lw.s.m.i1c(true);
            let lanes = lw.s.wave_all_bit_count(t);
            let ok = lw.tmp_var();
            let one = lw.c(1);
            lw.wr(ok, one);
            let first = lw.s.wave_is_first_lane();
            let need = lw.mul(k1, xn);
            let own = lw.shlc(cw, 1);
            let own = lw.add(own, h);
            let own = lw.addc(own, cbase);
            let one_ = lw.c(1);
            let ho = lw.xor(h, one_);
            let other = lw.shlc(cw, 1);
            let other = lw.add(other, ho);
            let other = lw.addc(other, cbase);
            lw.guarded(first, &mut |lw| {
                lw.s.gs_atomic(super::module::RmwOp::Add, xc, own, lanes);
                let pre = lw.cur();
                let lp = lw.block(Some(2));
                let ex = lw.block(Some(1));
                lw.br(lp);
                lw.s.m.set_block(lp);
                let i32 = lw.s.m.i32();
                let n = lw.s.m.phi(i32);
                let z = lw.c(0);
                lw.s.m.add_incoming(n, z, pre);
                let c = lw.s.gs_atomic(super::module::RmwOp::Add, xc, other, z);
                let d = lw.sub(c, need);
                let done = lw.cmp(Pred::SGe, d, z);
                let n1 = lw.addc(n, 1);
                let out = lw.cmpc(Pred::UGe, n1, 1 << 20);
                let stop = lw.lor(done, out);
                let me = lw.cur();
                lw.s.m.add_incoming(n, n1, me);
                lw.cond_br(stop, ex, lp);
                lw.s.m.set_block(ex);
                let okv = lw.u(done);
                lw.wr(ok, okv);
                Ok(())
            })?;
            let okv = lw.rd(ok);
            let okf = lw.s.wave_read_lane_first(okv);
            let fail = lw.cmpc(Pred::Eq, okf, 0);
            let d0 = lw.rd(VAR_XDEAD);
            let d1 = lw.selc(fail, 1, d0);
            lw.wr(VAR_XDEAD, d1);
            lw.s.barrier(barrier::TGSM_FENCE);
            lw.wr(met, okf);
            Ok(())
        });
        let one = self.c(1);
        let ho = self.xor(h, one);
        let theirs = at(self, ho);
        let metv = self.rd(met);
        let ok = self.nz(metv);
        let mut out = Vec::new();
        for &o in loads {
            let a = self.add(theirs, o);
            let q = self.s.gs_ptr(xs, a);
            let v = self.s.m.load(q, 4);
            let z = self.c(0);
            out.push(self.sel(ok, v, z));
        }
        out
    }

    /// The exchange words (created on first use when the program has no LDS).
    fn xs_storage(&mut self) -> (super::ops::GroupShared, u32) {
        if let Some(x) = self.xs {
            return x;
        }
        let i32 = self.s.m.i32();
        let x = self.s.groupshared("xcl_xs", i32, 2 * self.xslot as u64);
        self.xs = Some((x, 0));
        (x, 0)
    }

    pub fn exchange(&mut self, v: Value) -> Value {
        self.exchange_n(&[v])[0]
    }

    /// Split form: `v` of any lane `l` of the console wave (`ds_bpermute`): every lane
    /// gives its value, 32 words per half.
    pub fn read_lane_any(&mut self, v: Value, l: Value) -> Value {
        if !self.split {
            return self.rla(v, l);
        }
        if self.partner == Partner::Spin {
            let l31 = self.lane31;
            let src31 = self.andc(l, 31);
            let other = self.spin_exchange(&[(l31, v)], &[src31])[0];
            let own = self.s.wave_read_lane_at(v, src31);
            let lh = self.lshrc(l, 5);
            let h = self.half;
            let same = self.cmp(Pred::Eq, lh, h);
            return self.sel(same, own, other);
        }
        if self.partner != Partner::Present {
            // no partner half: only this half's lanes (the others read 0)
            let _ = self.exchange_n(&[]);
            let src31 = self.andc(l, 31);
            let own = self.s.wave_read_lane_at(v, src31);
            let lh = self.lshrc(l, 5);
            let h = self.half;
            let same = self.cmp(Pred::Eq, lh, h);
            let z = self.c(0);
            return self.sel(same, own, z);
        }
        let (xs, base) = self.xs_storage();
        let h = self.half;
        let mine = self.shlc(h, 5);
        let mine = self.addc(mine, base);
        let l31 = self.lane31;
        let at = self.add(mine, l31);
        let p = self.s.gs_ptr(xs, at);
        self.s.m.store(p, v, 4);
        self.meet();
        let one = self.c(1);
        let o = self.xor(h, one);
        let theirs = self.shlc(o, 5);
        let theirs = self.addc(theirs, base);
        let src31 = self.andc(l, 31);
        let at = self.add(theirs, src31);
        let q = self.s.gs_ptr(xs, at);
        let other = self.s.m.load(q, 4);
        self.meet();
        let own = self.s.wave_read_lane_at(v, src31);
        let lh = self.lshrc(l, 5);
        let same = self.cmp(Pred::Eq, lh, h);
        self.sel(same, own, other)
    }

    /// The program's ballot of `cond` → (lo, hi) mask words.
    pub fn ballot(&mut self, cond: Value) -> (Value, Value) {
        if self.gfx.as_ref().is_some_and(|g| g.ps.is_some()) {
            return self.ballot_ps(cond);
        }
        let w = self.s.wave_ballot(cond);
        if !self.wave64 {
            let z = self.c(0);
            // a wave32 program in 64-lane hardware: each half is its own wave
            if let Some(h) = self.gfx_half {
                return (self.sel(h, w[1], w[0]), z);
            }
            return (w[0], z);
        }
        if !self.split {
            return (w[0], w[1]);
        }
        let m = w[0];
        let o = self.exchange(m);
        let u = self.upper;
        let lo = self.sel(u, o, m);
        let hi = self.sel(u, m, o);
        (lo, hi)
    }

    /// Pixel-shader ballot from a precomputed 4-bit quad mask `q` (bit `p` = member `p`'s
    /// predicate, identical across all 4 lanes of each quad; parity with HLSL `ballot_px_quad`
    /// in `recomp/emit.rs`): 4 `WaveActiveBallot` passes spread each quad's member `p` from
    /// its live lane(s) back to bit `4k + p`.
    pub fn ballot_ps_quad(&mut self, q: Value) -> (Value, Value) {
        let z = self.c(0);
        if self.s.m.const_u64(q) == Some(0) {
            return (z, z);
        }
        let need_hi = self.wave64 || self.gfx_half.is_some();
        let mut m0 = z;
        let mut m1 = z;
        for p in 0..4u32 {
            let qp = self.lshrc(q, p);
            let qp = self.andc(qp, 1);
            let bit = self.nz(qp);
            let w = self.s.wave_ballot(bit);
            let spread = |lw: &mut Lw, mut v: Value| -> Value {
                let s1 = lw.lshrc(v, 1);
                v = lw.or(v, s1);
                let s2 = lw.lshrc(v, 2);
                v = lw.or(v, s2);
                let v = lw.andc(v, 0x1111_1111);
                lw.shlc(v, p)
            };
            let b0 = spread(self, w[0]);
            m0 = self.or(m0, b0);
            if need_hi {
                let b1 = spread(self, w[1]);
                m1 = self.or(m1, b1);
            }
        }
        if !self.wave64 {
            if let Some(h) = self.gfx_half {
                return (self.sel(h, m1, m0), z);
            }
            return (m0, z);
        }
        (m0, m1)
    }

    /// Pixel-shader ballot including helper lanes (parity with HLSL `ballot_px` in
    /// `recomp/emit.rs`): each quad collects its 4 bits through `QuadReadAcross*`
    /// (which see helper lanes) and feeds them to [`Lw::ballot_ps_quad`]. Entry `EXEC` in
    /// `setup_pixel` runs before `self.gfx` is set, so it stays live-lanes-only.
    fn ballot_ps(&mut self, cond: Value) -> (Value, Value) {
        let z = self.c(0);
        if self.s.m.const_u64(cond) == Some(0) {
            return (z, z);
        }
        let qbit = self.andc(self.lane, 3);
        let own_b = self.u(cond);
        let own = self.bin(BinOp::Shl, own_b, qbit);
        let qx = self.s.quad_op(own, super::ops::QuadDir::ReadAcrossX);
        let qy = self.s.quad_op(own, super::ops::QuadDir::ReadAcrossY);
        let qd = self.s.quad_op(own, super::ops::QuadDir::ReadAcrossDiagonal);
        let q0 = self.or(own, qx);
        let q1 = self.or(qy, qd);
        let q = self.or(q0, q1);
        self.ballot_ps_quad(q)
    }

    /// The hardware lane of console lane `l` (`l + 32 · xcl_half` for a wave32 program in
    /// 64-lane hardware).
    fn hw_lane(&mut self, l: Value) -> Value {
        match self.gfx_half {
            Some(h) => {
                let k = self.c(32);
                let z = self.c(0);
                let o = self.sel(h, k, z);
                self.add(l, o)
            }
            None => l,
        }
    }

    /// `v` of lane `l` (the same index in every lane: v_readlane, v_readfirstlane).
    pub fn read_lane_uniform(&mut self, v: Value, l: Value) -> Value {
        if !self.split {
            return self.rla(v, l);
        }
        let l31 = self.andc(l, 31);
        let m = self.s.wave_read_lane_at(v, l31);
        let o = self.exchange(m);
        let lh = self.lshrc(l, 5);
        let mine = self.cmp(Pred::Eq, lh, self.half);
        self.sel(mine, m, o)
    }

    /// `v` of lane `l` within the reader's 32-lane half (DPP, swizzle, permlane).
    pub fn read_lane_half(&mut self, v: Value, l: Value) -> Value {
        if self.split {
            let l31 = self.andc(l, 31);
            self.s.wave_read_lane_at(v, l31)
        } else {
            self.rla(v, l)
        }
    }

    /// The HLSL `rla(v, l)`: `WaveReadLaneAt(v, l + 32 · xcl_half)`; in a pixel shader, lanes
    /// of the reader's own quad through quad reads (they see helper lanes, wave reads do
    /// not), and a helper lane of another quad through a live lane of that quad.
    pub fn rla(&mut self, v: Value, l: Value) -> Value {
        let h = self.hw_lane(l);
        let w = self.s.wave_read_lane_at(v, h);
        if !self.gfx.as_ref().is_some_and(|g| g.ps.is_some()) {
            return w;
        }
        let q: Vec<Value> = (0..4u32).map(|k| {
            let kk = self.c(k);
            self.s.quad_read_lane_at(v, kk)
        }).collect();
        let pick = |lw: &mut Lw, r: &[Value]| -> Value {
            let b1 = lw.andc(l, 1);
            let b1 = lw.nz(b1);
            let b2 = lw.andc(l, 2);
            let b2 = lw.nz(b2);
            let hi = lw.sel(b1, r[3], r[2]);
            let lo = lw.sel(b1, r[1], r[0]);
            lw.sel(b2, hi, lo)
        };
        let qv = pick(self, &q);
        let t = self.s.m.i1c(true);
        let lv = self.s.wave_ballot(t);
        let wi = self.lshrc(h, 5);
        let k3 = self.c(3);
        let wi = self.umin(wi, k3);
        let lvw = {
            let mut x = lv[3];
            for k in (0..3).rev() {
                let is = self.cmpc(Pred::Eq, wi, k);
                x = self.sel(is, lv[k as usize], x);
            }
            x
        };
        let sh = self.andc(h, 28);
        let nib = self.bin(BinOp::LShr, lvw, sh);
        let nib = self.andc(nib, 15);
        let qbase = self.andc(h, !3);
        let fl = self.firstbitlow(nib);
        let src = self.add(qbase, fl);
        let r: Vec<Value> = q.iter().map(|&qq| self.s.wave_read_lane_at(qq, src)).collect();
        let rv = pick(self, &r);
        let h3 = self.andc(h, 3);
        let own = self.bin(BinOp::LShr, nib, h3);
        let own = self.andc(own, 1);
        let helper = self.cmpc(Pred::Eq, own, 0);
        let nnz = self.nz(nib);
        let use_r = self.land(nnz, helper);
        let w = self.sel(use_r, rv, w);
        let lq = self.lshrc(l, 2);
        let mq = self.lshrc(self.lane, 2);
        let same = self.cmp(Pred::Eq, lq, mq);
        self.sel(same, qv, w)
    }

    /// First set lane of (lo, hi), 0 when none (the recompiler's `first_lane`).
    pub fn first_lane(&mut self, lo: Value, hi: Value) -> Value {
        let fl = self.firstbitlow(lo);
        let fh = self.firstbitlow(hi);
        let fh = self.addc(fh, 32);
        let hnz = self.nz(hi);
        let z = self.c(0);
        let hv = self.sel(hnz, fh, z);
        let lnz = self.nz(lo);
        self.sel(lnz, fl, hv)
    }
}

#[cfg(test)]
mod ps_ballot_tests {
    use super::super::kernel::{self, KernelOptions};
    use super::super::validate;
    use crate::gpu::dxbc::SigElement;
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::{dxc, runner, PsInterface};

    const V: u64 = 256;

    fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> SigElement {
        SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
    }

    fn ps_interface() -> PsInterface {
        PsInterface { inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x302, attrs: None }
    }

    /// Pixel shader exercising `s_wqm`, `v_cndmask` + DPP `ddx`, `s_and_saveexec` + DPP `ddy`,
    /// and a waterfall loop over `v_readfirstlane` of `pos.x` (parity with
    /// `recomp::tests::helper_lane_masks_ps`).
    fn helper_lane_masks_ps_code(wave64: bool) -> Vec<u8> {
        let (mov, wqm, and_save, xor) = if wave64 {
            ("S_MOV_B64", "S_WQM_B64", "S_AND_SAVEEXEC_B64", "S_XOR_B64")
        } else {
            ("S_MOV_B32", "S_WQM_B32", "S_AND_SAVEEXEC_B32", "S_XOR_B32")
        };
        let sop1 = |n: &'static str, d: u64, s: u64| (n, "ENC_SOP1", vec![("SDST", d), ("SSRC0", s)], None);
        type Inst = (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>);
        let head: Vec<Inst> = vec![
            sop1(mov, 20, 126),
            sop1(wqm, 126, 126),
            ("V_CMP_LT_F32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 2)], None),
            ("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 128), ("VSRC1", 2)], None),
            ("V_MOV_B32", "VOP1_VOP_DPP16", vec![("VDST", 5), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0xB1), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            ("V_SUB_F32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", V + 5), ("VSRC1", 4)], None),
            ("V_CMP_LT_F32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 3)], None),
            sop1(and_save, 22, 106),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", V + 3)], None),
            sop1(mov, 126, 22),
            ("V_MOV_B32", "VOP1_VOP_DPP16", vec![("VDST", 8), ("SRC0", 0xFA), ("VSRC0", 7), ("DPP_CTRL", 0x4E), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            ("V_SUB_F32", "ENC_VOP2", vec![("VDST", 9), ("SRC0", V + 8), ("VSRC1", 7)], None),
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 8), ("SSRC0", 128)], None),
            sop1(mov, 24, 126),
        ];
        let body: Vec<Inst> = vec![
            ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 4), ("SRC0", V + 2)], None),
            ("V_CMP_EQ_F32", "ENC_VOPC", vec![("SRC0", 4), ("VSRC1", 2)], None),
            sop1(and_save, 26, 106),
            ("S_ADD_U32", "ENC_SOP2", vec![("SDST", 8), ("SSRC0", 8), ("SSRC1", 129)], None),
            (xor, "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 26), ("SSRC1", 126)], None),
        ];
        let tail: Vec<Inst> = vec![
            sop1(mov, 126, 24),
            ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 10), ("SRC0", 8)], None),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 11), ("SRC0", 242)], None),
            sop1(mov, 126, 20),
            ("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 6), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11), ("DONE", 1), ("VM", 1)], None),
        ];
        let enc = |l: &[Inst]| -> Vec<u8> { l.iter().flat_map(|(n, e, f, x)| encode(n, e, f, *x).unwrap_or_else(|err| panic!("{n}: {err}"))).collect() };
        let body_code = enc(&body);
        let back = (-((body_code.len() as i64 + 4) / 4)) as u16 as u64;
        let mut ps_code = enc(&head);
        ps_code.extend(&body_code);
        ps_code.extend(encode("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", back)], None).unwrap());
        ps_code.extend(enc(&tail));
        ps_code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        ps_code
    }

    #[test]
    fn direct_ps_ballot_uses_quad_ops_and_keeps_entry_exec_live_only() {
        let psi = ps_interface();
        for wave64 in [false, true] {
            let opt = KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 };
            let k = kernel::translate_pixel(&helper_lane_masks_ps_code(wave64), opt, &psi).expect("translate PS");
            validate::validate(&k.dxil).expect("validate PS");
            let dis = validate::disassemble(&k.dxil).expect("disassemble PS");
            // #134: VOPC masks quad-read the VGPR operands (`quadReadLaneAt`), never the
            // compare predicate (`quadOp`), avoiding AMD RDNA2/3/4 amdxc64 PSO crashes.
            assert!(dis.contains("dx.op.quadReadLaneAt.i32"), "wave64 {wave64}: PS VOPC must quad-read VGPR operands via quadReadLaneAt:\n{dis}");
            assert!(!dis.contains("dx.op.quadOp"), "wave64 {wave64}: PS VOPC must not emit quadOp on compare results (#134):\n{dis}");

            // Standalone VOPC with no DPP / readfirstlane (so `rla` is never called):
            // proves VOPC itself emits the 4 VGPR operand `quadReadLaneAt` reads and zero `quadOp`.
            let mut vopc_only = Vec::new();
            vopc_only.extend(encode(if wave64 { "S_WQM_B64" } else { "S_WQM_B32" }, "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None).unwrap());
            vopc_only.extend(encode("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 2)], None).unwrap());
            vopc_only.extend(encode("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 128), ("VSRC1", 2)], None).unwrap());
            vopc_only.extend(encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 3), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None).unwrap());
            vopc_only.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
            let kv = kernel::translate_pixel(&vopc_only, opt, &psi).expect("translate VOPC-only PS");
            validate::validate(&kv.dxil).expect("validate VOPC-only PS");
            let vdis = validate::disassemble(&kv.dxil).expect("disassemble VOPC-only PS");
            assert_eq!(vdis.matches("call i32 @dx.op.quadReadLaneAt.i32(").count(), 4, "wave64 {wave64}: expected 4 operand quad reads for V_CMP_LT_F32(0, v2):\n{vdis}");
            assert!(!vdis.contains("dx.op.quadOp"), "wave64 {wave64}: VOPC-only PS must not emit quadOp:\n{vdis}");

            // Entry EXEC alone (no post-entry ballot) must stay live-lanes-only (no quad reads).
            let mut trivial = Vec::new();
            trivial.extend(encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 2), ("VSRC1", 3), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None).unwrap());
            trivial.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
            let kt = kernel::translate_pixel(&trivial, opt, &psi).expect("translate trivial PS");
            validate::validate(&kt.dxil).expect("validate trivial PS");
            let tdis = validate::disassemble(&kt.dxil).expect("disassemble trivial PS");
            assert!(
                !tdis.contains("dx.op.quadOp") && !tdis.contains("dx.op.quadReadLaneAt"),
                "wave64 {wave64}: entry EXEC ballot must stay live-lanes-only:\n{tdis}"
            );
        }
    }

    #[test]
    fn direct_ps_constant_false_ballot_skips_quad_ops() {
        let psi = ps_interface();
        let mut code = Vec::new();
        code.extend(encode("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None).unwrap());
        code.extend(encode("V_CMP_F_F32", "ENC_VOPC", &[("SRC0", V + 2), ("VSRC1", 3)], None).unwrap());
        code.extend(encode("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 128), ("VSRC1", 2)], None).unwrap());
        code.extend(encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 3), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let k = kernel::translate_pixel(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }, &psi).expect("translate false-cmp PS");
        validate::validate(&k.dxil).expect("validate false-cmp PS");
        let dis = validate::disassemble(&k.dxil).expect("disassemble false-cmp PS");
        assert!(
            !dis.contains("dx.op.quadOp") && !dis.contains("dx.op.quadReadLaneAt"),
            "constant-false PS ballot should short-circuit to 0:\n{dis}"
        );
    }

    #[test]
    fn direct_ps_non_vopc_carry_ballot_uses_quad_op() {
        let psi = ps_interface();
        let mut code = Vec::new();
        code.extend(encode("S_WQM_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None).unwrap());
        code.extend(encode("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 4), ("SDST", 106), ("SRC0", V + 2), ("SRC1", V + 3)], None).unwrap());
        code.extend(encode("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 128), ("VSRC1", 4)], None).unwrap());
        code.extend(encode("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 5), ("VSRC1", 3), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1), ("VM", 1)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let k = kernel::translate_pixel(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }, &psi).expect("translate carry PS");
        validate::validate(&k.dxil).expect("validate carry PS");
        let dis = validate::disassemble(&k.dxil).expect("disassemble carry PS");
        assert!(dis.contains("dx.op.quadOp.i32"), "non-VOPC PS carry ballot must preserve helper lanes via quadOp:\n{dis}");
    }

    #[test]
    fn gpu_direct_ps_masks_hold_helper_lanes_on_triangle_edges() {
        match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => {}
            (a, w, d) => {
                runner::skip_test(if a.is_err() || w.is_err() {
                    "no GPU"
                } else if d.is_err() {
                    "no DXC"
                } else {
                    "no wave64"
                });
                return;
            }
        }
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
        let vs = dxc::compile("float4 main(float2 p : POSITION) : SV_Position { return float4(p, 0.5, 1.0); }\n", "main", "vs_6_6", &[]).unwrap();
        let psi = ps_interface();
        let quad = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        let layout = [runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 }];
        let size = 32usize;
        for wave64 in [true, false] {
            let opt = KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 };
            let ps = kernel::translate_pixel(&helper_lane_masks_ps_code(wave64), opt, &psi).expect("translate PS");
            let px = runner::draw(&vs, &ps.dxil, &layout, &quad, 8, &[], &[], size as u32, [9.0; 4]).unwrap();
            let lanes = if wave64 { 64.0 } else { 32.0 };
            let bad: Vec<(usize, usize, [f32; 4])> = (0..size * size)
                .filter(|&i| {
                    let p = px[i];
                    p[0].abs() != 1.0 || p[1].abs() != 1.0 || !(1.0..=lanes).contains(&p[2]) || p[3] != 1.0
                })
                .map(|i| (i % size, i / size, px[i]))
                .collect();
            let count = |k: usize, ok: &dyn Fn(f32) -> bool| px.iter().filter(|p| !ok(p[k])).count();
            let (dx, dy, it) = (count(0, &|v| v.abs() == 1.0), count(1, &|v| v.abs() == 1.0), count(2, &|v| (1.0..=lanes).contains(&v)));
            assert!(
                bad.is_empty(),
                "wave64 {wave64}: {} pixels wrong (ddx {dx}, ddy {dy}, loop {it}), first {:?}",
                bad.len(),
                &bad[..bad.len().min(6)]
            );
        }
    }
}
