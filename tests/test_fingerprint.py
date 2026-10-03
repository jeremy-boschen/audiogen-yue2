"""Stack fingerprints: measured once per stack, looked up without loading anything."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audiogen import fingerprint


class FingerprintTests(unittest.TestCase):
    def test_measured_once_per_key_then_looked_up(self):
        record = {"probe": fingerprint.PROBE, "hash": "aaaa", "parts": {"pcm": "p"}, "stack": {}, "settings": {}}
        with tempfile.TemporaryDirectory() as folder:
            cache, models = Path(folder) / "fp.json", Path(folder)
            self.assertIsNone(fingerprint.lookup(models, {"profile": "x"}, cache))
            with patch.object(fingerprint, "measure", return_value=record) as measure:
                self.assertEqual(fingerprint.current(models, {"profile": "x"}, cache)["hash"], "aaaa")
                self.assertEqual(fingerprint.current(models, {"profile": "x"}, cache)["hash"], "aaaa")
                self.assertEqual(measure.call_count, 1)
                # Other engine settings are another key: measured on their own.
                fingerprint.current(models, {"profile": "y"}, cache)
                self.assertEqual(measure.call_count, 2)
            self.assertEqual(len(json.loads(cache.read_text())), 2)
            self.assertEqual(fingerprint.lookup(models, {"profile": "x"}, cache)["hash"], "aaaa")

    def test_the_key_moves_with_the_stack_and_not_with_the_machine_name(self):
        models = Path(tempfile.gettempdir())
        base = fingerprint.key(models, {"profile": "x"})
        self.assertEqual(base, fingerprint.key(models, {"profile": "x", "offload_ar": True}))
        with patch.object(fingerprint, "stack", return_value={**fingerprint.stack(), "machine": "other"}):
            self.assertEqual(base, fingerprint.key(models, {"profile": "x"}))
        with patch.object(fingerprint, "stack", return_value={**fingerprint.stack(), "os": "macOS 99"}):
            self.assertNotEqual(base, fingerprint.key(models, {"profile": "x"}))
        with patch.object(fingerprint, "PROBE", fingerprint.PROBE + 1):
            self.assertNotEqual(base, fingerprint.key(models, {"profile": "x"}))

    def test_a_lookup_loads_neither_torch_nor_numpy(self):
        import subprocess
        code = ("import sys; from pathlib import Path; from audiogen import fingerprint; "
                "fingerprint.lookup(Path('.'), {}, Path('missing.json')); "
                "print('torch' in sys.modules, 'numpy' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent / "src")
        self.assertEqual(out.stdout.split(), ["False", "False"], out.stderr)

    def test_same_and_differs(self):
        a = {"probe": 1, "hash": "a", "parts": {"rms_norm": "1", "pcm": "2"}}
        self.assertTrue(fingerprint.same(a, dict(a)))
        self.assertFalse(fingerprint.same(a, {**a, "hash": "b"}))
        self.assertIsNone(fingerprint.same(a, {**a, "probe": 2}))
        self.assertIsNone(fingerprint.same(a, None))
        self.assertEqual(fingerprint.differs(a, {**a, "parts": {"rms_norm": "1", "pcm": "3"}}), ["pcm"])


if __name__ == "__main__":
    unittest.main()
