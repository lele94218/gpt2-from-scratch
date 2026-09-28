"""Offline regression tests: shard boundaries, resume state and actual training restarts."""
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tiktoken
import torch

from data import TokenLoader
from prepare_fineweb import write_shards
from checkpoint import rng_state, restore_rng

ROOT = Path(__file__).resolve().parents[1]


def assert_nested_equal(test, a, b):
    test.assertEqual(type(a), type(b))
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_nested_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.documents = [{'text': ' hello world' * 250}, {'text': ' other document' * 100}]
        self.manifest = write_shards(self.documents, self.data, 65)

    def tearDown(self):
        self.temp.cleanup()

    def test_long_document_shards_and_rank_cursor(self):
        enc = tiktoken.get_encoding('gpt2')
        expected = sum(([enc.eot_token] + enc.encode_ordinary(d['text']) for d in self.documents), [])
        actual = np.concatenate([np.load(self.data / s['file']) for s in self.manifest['shards']])
        np.testing.assert_array_equal(actual, expected)
        loaders = [TokenLoader(1, 4, rank, 2, data_dir=self.data) for rank in range(2)]
        for _ in range(25):
            shard, pos = loaders[0].shard, loaders[0].position
            array = loaders[0].arrays[shard]
            for rank, loader in enumerate(loaders):
                x, y = loader.next_batch()
                start = pos + rank * 4
                np.testing.assert_array_equal(x.numpy().ravel(), array[start:start+4])
                np.testing.assert_array_equal(y.numpy().ravel(), array[start+1:start+5])
            self.assertEqual(loaders[0].state_dict(), loaders[1].state_dict())
        restored = TokenLoader(1, 4, 0, 2, data_dir=self.data)
        restored.load_state_dict(loaders[0].state_dict())
        assert_nested_equal(self, restored.next_batch(), loaders[0].next_batch())
        with self.assertRaisesRegex(ValueError, 'world_size'):
            TokenLoader(1, 4, 0, 1, data_dir=self.data).load_state_dict(restored.state_dict())
        path = self.data / self.manifest['shards'][1]['file']
        array = np.load(path)
        array[0] = 1
        np.save(path, array)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            TokenLoader(1, 4, 0, 1, data_dir=self.data)

    def test_token_cap_and_short_shard(self):
        capped = self.root / 'capped'
        manifest = write_shards(self.documents, capped, 16, max_tokens=35)
        self.assertEqual([s['tokens'] for s in manifest['shards']], [16, 16, 3])
        loader = TokenLoader(1, 4, 0, 1, data_dir=capped)
        self.assertEqual(len(loader.arrays), 1)
        with self.assertRaisesRegex(ValueError, 'No usable'):
            TokenLoader(4, 8, 0, 1, data_dir=capped)
        with self.assertRaisesRegex(ValueError, 'empty'):
            write_shards(self.documents, capped, 16)

    def test_text_training_and_generation(self):
        result = subprocess.run([sys.executable, str(ROOT/'train-gpt2.py'),
            '--device', 'cpu', '--no-compile', '--max-steps', '1',
            '--batch-size', '1', '--seq-len', '4', '--total-batch-size', '8',
            '--n-layer', '1', '--n-head', '1', '--n-embd', '16',
            '--output-dir', str(self.root/'text')],
            env=dict(os.environ, OMP_NUM_THREADS='1'), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(re.findall(r'^> ', result.stdout, re.MULTILINE)), 5)
        state = torch.load(self.root/'text/latest.pt', map_location='cpu', weights_only=False)
        self.assertEqual(state['next_step'], 1)

    def test_rank_rng_roundtrip(self):
        device = torch.device(os.environ.get('TEST_DEVICE', 'cpu'))
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        state = rng_state(device)
        def draw():
            return (random.random(), np.random.rand(4), torch.rand(4), torch.rand(4, device=device))
        expected = draw()
        restore_rng(state, device)
        assert_nested_equal(self, expected, draw())

    def test_restart_matches_uninterrupted(self):
        device = os.environ.get('TEST_DEVICE', 'cpu')
        worlds = [int(x) for x in os.environ.get('TEST_WORLDS', '1,2').split(',')]
        compiled = os.environ.get('TEST_COMPILE') == '1'
        for world in worlds:
            with self.subTest(world=world, device=device, compiled=compiled):
                prefix = [sys.executable]
                # Exercise torchrun/NCCL even at world size one when requested.
                if world > 1 or os.environ.get('TEST_TORCHRUN') == '1':
                    prefix += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
                base = prefix + [str(ROOT/'train-gpt2.py'), '--device', device,
                    '--compile' if compiled else '--no-compile', '--no-generate',
                    '--n-layer', '1', '--n-head', '1', '--n-embd', '16',
                    '--batch-size', '1', '--seq-len', '4', '--total-batch-size', '16',
                    '--max-steps', '4', '--warmup-steps', '1', '--checkpoint-every', '2',
                    '--data-dir', str(self.data)]
                full, part = self.root/f'full-{world}', self.root/f'part-{world}'
                env = dict(os.environ, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                def run(extra, ok=True):
                    result = subprocess.run(base+extra, env=env, capture_output=True, text=True, timeout=240)
                    if ok:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                    return result.stdout + result.stderr
                full_log = run(['--output-dir', str(full)])
                first_log = run(['--output-dir', str(part), '--stop-after', '2'])
                checkpoint = str(part/'latest.pt')
                second_log = run(['--output-dir', str(part), '--resume', checkpoint])
                pattern = r'step (\d+), loss: ([^,\n]+), lr: ([^ ]+)'
                expected = re.findall(pattern, full_log)
                self.assertEqual(len(expected), 4)
                self.assertEqual(expected, re.findall(pattern, first_log+second_log))
                a = torch.load(full/'latest.pt', map_location='cpu', weights_only=False)
                b = torch.load(part/'latest.pt', map_location='cpu', weights_only=False)
                assert_nested_equal(self, a, b)
                self.assertEqual(a['next_step'], 4)
                if world == 1:
                    error = run(['--output-dir', str(part), '--resume', checkpoint, '--max-steps', '5'], ok=False)
                    self.assertIn('Resume configuration differs', error)
                    error = run(['--output-dir', str(part)], ok=False)
                    self.assertIn('exists', error)


if __name__ == '__main__':
    unittest.main()
