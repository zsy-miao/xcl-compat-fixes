"""Print a PE's exports: ordinal, all names for that ordinal, RVA (forwarders shown)."""
import struct, sys

for path in sys.argv[1:]:
    b = open(path, 'rb').read()
    pe = struct.unpack_from('<I', b, 0x3C)[0]
    nsec = struct.unpack_from('<H', b, pe + 6)[0]
    opt = pe + 24
    magic = struct.unpack_from('<H', b, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    exp_rva, exp_size = struct.unpack_from('<II', b, dd)
    secs = []
    so = opt + struct.unpack_from('<H', b, pe + 20)[0]
    for i in range(nsec):
        vs, va, raw_sz, raw = struct.unpack_from('<IIII', b, so + 40 * i + 8)
        secs.append((va, max(vs, raw_sz), raw))

    def off(rva):
        for va, sz, raw in secs:
            if va <= rva < va + sz:
                return rva - va + raw
        raise ValueError(hex(rva))

    def cstr(rva):
        o = off(rva)
        return b[o:b.index(b'\0', o)].decode('latin1')

    print(f'== {path}')
    if not exp_rva:
        print('  (no exports)')
        continue
    e = off(exp_rva)
    name_rva, base, nfun, nnames, afun, anames, aords = struct.unpack_from('<IIIIIII', b, e + 12)
    names = {}
    for i in range(nnames):
        nr = struct.unpack_from('<I', b, off(anames) + 4 * i)[0]
        o = struct.unpack_from('<H', b, off(aords) + 2 * i)[0]
        names.setdefault(o, []).append(cstr(nr))
    print(f'  {cstr(name_rva)}: {nfun} functions, {nnames} names, ordinal base {base}')
    for i in range(nfun):
        rva = struct.unpack_from('<I', b, off(afun) + 4 * i)[0]
        if not rva:
            continue
        fwd = f' -> {cstr(rva)}' if exp_rva <= rva < exp_rva + exp_size else ''
        print(f'  #{base + i:<4} {", ".join(names.get(i, ["(ordinal only)"]))} @{rva:#x}{fwd}')
