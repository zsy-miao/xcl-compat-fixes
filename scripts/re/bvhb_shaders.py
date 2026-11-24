"""Extract the DXBC containers embedded in XboxBvhBuilder.dll and print each one's root
signature (RTS0) parameters, with b/t/u registers for root CBV/SRV/UAV descriptors.

usage: python bvhb_shaders.py <XboxBvhBuilder.dll> [out dir, default ./bvhb_shaders]"""
import os, struct, sys

DLL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else 'bvhb_shaders'
os.makedirs(OUT, exist_ok=True)
b = open(DLL, 'rb').read()
TYPES = {0: 'table', 1: 'consts', 2: 'CBV', 3: 'SRV', 4: 'UAV'}
RANGE = {0: 'SRV', 1: 'UAV', 2: 'CBV', 3: 'Sampler'}


def parts(blob):
    n = struct.unpack_from('<I', blob, 28)[0]
    for k in range(n):
        off = struct.unpack_from('<I', blob, 32 + 4 * k)[0]
        cc, size = blob[off:off + 4].decode('latin1'), struct.unpack_from('<I', blob, off + 4)[0]
        yield cc, blob[off + 8:off + 8 + size]


def rts0(r):
    ver, n, poff, ns, soff, flags = struct.unpack_from('<6I', r, 0)
    out = []
    for k in range(n):
        ty, vis, pay = struct.unpack_from('<3I', r, poff + 12 * k)
        if ty == 0:
            nr, roff = struct.unpack_from('<2I', r, pay)
            rs = []
            for j in range(nr):
                w = struct.unpack_from('<6I' if ver >= 2 else '<5I', r, roff + (24 if ver >= 2 else 20) * j)
                rs.append(f'{RANGE[w[0]]}x{w[1] if w[1] != 0xFFFFFFFF else "∞"}@{w[2]}')
            out.append('table[' + ','.join(rs) + ']')
        elif ty == 1:
            reg, space, num = struct.unpack_from('<3I', r, pay)
            out.append(f'consts b{reg} x{num}')
        else:
            reg, space = struct.unpack_from('<2I', r, pay)
            out.append(f'{TYPES[ty]} {"btu"[ty - 2] if ty < 5 else "?"}{reg}')
    return f'v{ver} flags {flags:#x} samplers {ns}: ' + ' | '.join(out)


i = b.find(b'DXBC')
k = 0
while i >= 0:
    size = struct.unpack_from('<I', b, i + 24)[0]
    blob = b[i:i + size]
    if 32 < size < 4 << 20:
        names = [cc for cc, _ in parts(blob)]
        rs = next((p for cc, p in parts(blob) if cc == 'RTS0'), None)
        path = os.path.join(OUT, f'{k:02d}.dxbc')
        open(path, 'wb').write(blob)
        print(f'{k:02d} @{i:#x} {size:6} B {",".join(names)}  RS: {rts0(rs) if rs else "-"}')
        k += 1
    i = b.find(b'DXBC', i + 4)
