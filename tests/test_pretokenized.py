"""Offline checks for llm.c token conversion and interrupted preparation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_fineweb_tokens as prep
from data import TokenLoader


class PretokenizedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokens = np.array([50256] + list(range(32)), dtype='<u2')
        self.source = self.root / 'source.bin'
        self.write_source()

    def write_source(self, magic=20240520, version=1, tokens=None):
        tokens = self.tokens if tokens is None else tokens
        header = np.zeros(256, dtype='<i4')
        header[:3] = [magic, version, len(tokens)]
        self.source.write_bytes(header.tobytes() + tokens.tobytes())

    def test_conversion_reuse_and_corruption(self):
        target = self.root / 'tokens.npy'
        entry = prep.convert_shard(self.source, target, len(self.tokens))
        np.testing.assert_array_equal(np.load(target), self.tokens)
        self.assertEqual(entry, prep.convert_shard(self.source, target, len(self.tokens)))
        changed = self.tokens.copy()
        changed[2] = 123
        np.save(target, changed)
        with self.assertRaisesRegex(ValueError, 'content mismatch'):
            prep.convert_shard(self.source, target, len(self.tokens))
        np.testing.assert_array_equal(np.load(target), changed)  # Never silently replace it.

    def test_reject_invalid_bin(self):
        target = self.root / 'tokens.npy'
        for magic, version in [(0, 1), (20240520, 7)]:
            self.write_source(magic, version)
            with self.assertRaisesRegex(ValueError, 'header'):
                prep.convert_shard(self.source, target, len(self.tokens))
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'token count'):
            prep.convert_shard(self.source, target, 999)
        self.source.write_bytes(self.source.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'file size'):
            prep.convert_shard(self.source, target, len(self.tokens))
        bad = self.tokens.copy()
        bad[0] = 65535
        self.write_source(tokens=bad)
        with self.assertRaisesRegex(ValueError, 'token ID'):
            prep.convert_shard(self.source, target, len(self.tokens))
        self.assertFalse(target.exists())

    def test_interrupted_prepare_resume_and_loader(self):
        out = self.root / 'out'
        real_convert = prep.convert_shard
        calls = []
        def download(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise ConnectionError('simulated interruption')
            return self.source
        def small_convert(source, target):
            return real_convert(source, target, len(self.tokens))
        with patch.object(prep, 'convert_shard', side_effect=small_convert):
            with self.assertRaises(ConnectionError):
                prep.prepare(out, self.root / 'cache', 1, download=download)
            self.assertFalse((out / 'manifest.json').exists())
            self.assertTrue((out / 'fineweb_val_000000.npy').exists())
            manifest = prep.prepare(out, self.root / 'cache', 1, download=download)
            self.assertEqual(len(manifest['shards']), 2)
            self.assertEqual(manifest, json.loads((out / 'manifest.json').read_text()))
            self.assertEqual(manifest, prep.prepare(out, self.root / 'cache', 1, download=download))
        loader = TokenLoader(1, 4, 0, 2, data_dir=out)
        x, y = loader.next_batch()
        np.testing.assert_array_equal(x.numpy().ravel(), self.tokens[:4])
        np.testing.assert_array_equal(y.numpy().ravel(), self.tokens[1:5])
        with self.assertRaisesRegex(ValueError, 'selection differs'):
            prep.prepare(out, self.root / 'cache', 2, download=download)
        self.assertTrue((out / 'manifest.json').exists())
        self.assertTrue(all(c['revision'] == prep.REVISION for c in calls))


if __name__ == '__main__':
    unittest.main()
