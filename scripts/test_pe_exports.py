from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "pe_exports.py"


class ExportAliasTests(unittest.TestCase):
    def test_all_names_for_an_ordinal_are_printed(self):
        for magic in (0x10B, 0x20B):
            with self.subTest(magic=hex(magic)), tempfile.TemporaryDirectory() as tmp:
                image = bytearray(0x400)
                image[:2] = b"MZ"
                struct.pack_into("<I", image, 0x3C, 0x80)
                image[0x80:0x84] = b"PE\0\0"
                optional_size = 0xF0 if magic == 0x20B else 0xE0
                machine = 0x8664 if magic == 0x20B else 0x14C
                struct.pack_into("<HHIIIHH", image, 0x84, machine, 1, 0, 0, 0, optional_size, 0x2002)
                optional = 0x98
                struct.pack_into("<H", image, optional, magic)
                struct.pack_into("<II", image, optional + 32, 0x1000, 0x200)
                struct.pack_into("<II", image, optional + 56, 0x2000, 0x200)
                directory = optional + (112 if magic == 0x20B else 96)
                struct.pack_into("<I", image, directory - 4, 16)
                struct.pack_into("<II", image, directory, 0x1000, 0x100)
                section = optional + optional_size
                image[section:section + 8] = b".edata\0\0"
                struct.pack_into("<IIII", image, section + 8, 0x200, 0x1000, 0x200, 0x200)
                struct.pack_into("<I", image, section + 36, 0x40000040)
                struct.pack_into("<7I", image, 0x20C, 0x1080, 7, 3, 3, 0x1040, 0x1050, 0x1060)
                struct.pack_into("<3I", image, 0x240, 0x1100, 0x1110, 0x10C0)
                struct.pack_into("<3I", image, 0x250, 0x1090, 0x1098, 0x10A0)
                struct.pack_into("<3H", image, 0x260, 0, 0, 2)
                for offset, text in ((0x280, b"synthetic.dll\0"), (0x290, b"Alpha\0"),
                                     (0x298, b"Beta\0"), (0x2A0, b"Forwarded\0"),
                                     (0x2C0, b"other.Target\0")):
                    image[offset:offset + len(text)] = text
                source = Path(tmp) / "synthetic.dll"
                source.write_bytes(image)
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(source)],
                    capture_output=True, text=True, check=True,
                )
                self.assertRegex(result.stdout, r"(?m)^  #7\s+Alpha, Beta @0x1100$")
                self.assertRegex(result.stdout, r"(?m)^  #8\s+\(ordinal only\) @0x1110$")
                self.assertRegex(result.stdout, r"(?m)^  #9\s+Forwarded @0x10c0 -> other.Target$")


if __name__ == "__main__":
    unittest.main()
