from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "bvhb_shaders.py"


class RootDescriptorTests(unittest.TestCase):
    def test_root_descriptors_print_their_register_classes(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                descriptors = ((2, 7), (3, 11), (4, 13))
                payload_size = 12 if version == 2 else 8
                payload_offset = 24 + 12 * len(descriptors)
                root = bytearray(struct.pack("<6I", version, len(descriptors), 24, 0, 0, 0))
                for index, (kind, register) in enumerate(descriptors):
                    root += struct.pack("<3I", kind, 0, payload_offset + index * payload_size)
                for kind, register in descriptors:
                    root += struct.pack("<2I", register, 0)
                    if version == 2:
                        root += struct.pack("<I", 0)
                chunk = b"RTS0" + struct.pack("<I", len(root)) + root
                blob = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1)
                blob += struct.pack("<I", 36) + chunk
                source = Path(tmp) / "synthetic.dll"
                source.write_bytes(blob)
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(source), str(Path(tmp) / "out")],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("CBV b7 | SRV t11 | UAV u13", result.stdout)


if __name__ == "__main__":
    unittest.main()
