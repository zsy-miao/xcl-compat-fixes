"""Generate synthetic q=0 and q=15 BCPack vectors using the encoder's byte counts.

The multiplexer never decodes rANS. Each inverse step records the number
of bytes the forward step will need later.
"""

import random
import struct
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BC1 = (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8)
BC7 = (4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5)


WIDTHS = {
    1: BC1,
    3: (4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8),
    4: (8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    5: (4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    6: (4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5),
    7: BC7,
}


def record_header(mode, body_size):
    if body_size <= 256:
        return bytes((mode, body_size - 1))
    return bytes((mode | 32,)) + struct.pack('<H', body_size - 257)


class EncoderLane:
    def __init__(self, symbols, width, reduction, model, carry=False, trim=False):
        state = 1 << 24
        emitted = bytearray()
        counts = []
        step = 1 << (16 - width + reduction)
        for symbol in reversed(symbols):
            lo = max(1, symbol * step)
            count = (symbol + 1) * step - lo
            if model == 15:
                alphabet = 1 << (width - reduction)
                count = 65536 - alphabet if symbol == 0 else 1
                lo = 1 if symbol == 0 else 65536 - alphabet + symbol
            start = len(emitted)
            while state >= count << 16:
                emitted.append(state & 255)
                state >>= 8
            counts.append(len(emitted) - start)
            whole, remainder = divmod(state, count)
            state = (whole << 16) + lo + remainder
        mode = 16 | model | (reduction << 6)
        size = 4 + len(emitted)
        state_bytes = state.to_bytes(4, 'big')
        self.continuation = None
        if carry and emitted:
            # First record ends immediately after the initial state. Its first
            # unfulfilled normalisation request starts the continuation header.
            self.header = deque(bytes((mode & ~16, 3)) + state_bytes)
            self.payload = deque()
            self.continuation = (deque(record_header(mode, len(emitted))),
                                 deque(reversed(emitted)))
        elif trim:
            assert len(symbols) == 1 and not any(emitted)
            state_bytes = state_bytes.rstrip(b'\x00')
            self.header = deque(record_header(mode, len(state_bytes)) + state_bytes)
            self.payload = deque()
            counts = [0]
        else:
            self.header = deque(record_header(mode, size) + state_bytes)
            self.payload = deque(reversed(emitted))
        self.counts = deque(reversed(counts))
        self.buffered = False
        self.owed = 0
        self.ahead = False

    def wants(self):
        return self.header or (self.payload and not self.buffered
                               and (self.owed or not self.ahead))

    def tick(self, served, stalled, out):
        if self.header:
            if served:
                out.append(self.header.popleft())
            return
        if served:
            out.append(self.payload.popleft())
            self.buffered = True
        if self.owed:
            if self.buffered:
                self.owed -= 1
                self.buffered = False
            return
        if self.counts and not (self.ahead or stalled):
            self.owed = self.counts.popleft()
            self.ahead = True
            if self.buffered and self.owed:
                self.buffered = False
                self.owed -= 1


def encode(symbols, widths, reductions, model=0, carry=False, trim=False):
    lanes = [EncoderLane(*spec, model, carry, trim) for spec in zip(symbols, widths, reductions)]
    out = bytearray()
    while any(lane.header or lane.counts or lane.owed for lane in lanes):
        wanted = [i for i, lane in enumerate(lanes) if lane.wants()]
        for i, lane in enumerate(lanes):
            lane.tick(i in wanted[:8], i in wanted[8:], out)
            if lane.owed and lane.continuation:
                lane.header, lane.payload = lane.continuation
                lane.continuation = None
        if all(lane.ahead for lane in lanes):
            for lane in lanes:
                lane.ahead = False
    assert all(not lane.payload and not lane.buffered for lane in lanes)
    return bytes(out)


def generate(mode, units, seed, reduced, **options):
    widths = WIDTHS[mode]
    rng = random.Random(seed)
    reductions = [rng.randrange(4) if reduced else 0 for width in widths]
    symbols = [[rng.randrange(1 << (width - reduction)) for _ in range(units)]
               for width, reduction in zip(widths, reductions)]
    if mode == 1:
        symbols[1] = [0] * units
    path = ROOT / f'bc{mode}-{units}-{seed}'
    path.with_suffix('.bin').write_bytes(encode(symbols, widths, reductions, **options))
    path.with_suffix('.lanes').write_bytes(b''.join(map(bytes, symbols)))
    return symbols, path


def seed_channels(slot):
    return [(7 * slot + 13 * k) % (1 << w) for k, w in enumerate(BC1[2:8])]


def assemble(symbols):
    blocks = []
    raw = bytearray()
    for i, (ref, _, *fields) in enumerate(zip(*symbols)):
        d = 255 - ref // 2 if ref & 1 else ref // 2
        predictor = ([0] * 6 if d == 0 else
                     blocks[i-d] if i >= d else seed_channels((i-d) % 256))
        channels = []
        for v, base, w in zip(fields, predictor, BC1[2:8]):
            delta = ~(v >> 1) if v & 1 else v >> 1
            channels.append((base + delta) % (1 << w))
        blocks.append(channels)
        for r, g, b in (channels[:3], channels[3:]):
            raw.extend(struct.pack('<H', r << 11 | g << 5 | b))
        indices = 0
        for y in range(4):
            for x in range(4):
                code = fields[6 + 2*(y//2) + x//2]
                px = (code >> (2 * (2*(y%2) + x%2))) & 3
                indices |= px << (2 * (4*y + x))
        raw.extend(struct.pack('<I', indices))
    return bytes(raw)


if __name__ == '__main__':
    for mode in WIDTHS:
        cases = [(2, 0, False), (16, 1, True), (700, 2, False)]
        if mode not in (1, 7):
            cases = [(16, 1, True)]
        for units, seed, reduced in cases:
            syms, path = generate(mode, units, seed, reduced)
            if mode == 1:
                path.with_suffix('.bc1').write_bytes(assemble(syms))

    for units, seed, extra in [(1, 8, dict(trim=True)), (16, 9, dict(carry=True))]:
        syms, path = generate(1, units, seed, False, **extra)
        path.with_suffix('.bc1').write_bytes(assemble(syms))

    for mode, widths in [(1, BC1), (7, BC7)]:
        units = 8192
        syms = [[0] * units for _ in widths]
        if mode == 1:
            for i in range(0, units, 511):
                syms[2][i] = 30
                syms[8][i] = 5
            for i in range(31, units, 31):
                syms[0][i] = 2
        path = ROOT / f'bc{mode}-{units}-3.bin'
        path.write_bytes(encode(syms, widths, [0]*len(widths), model=15))
