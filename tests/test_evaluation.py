import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import unittest
import numpy as np
import torch
from evaluation import batches, evaluate, validation_data
from prepare_fineweb import write_shards

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TargetMean(torch.nn.Module):
    def forward(self, x, y):
        # Consumes RNG to exercise isolation even for a stochastic forward.
        torch.rand(1)
        return None, y.float().mean()


class EvaluationTests(unittest.TestCase):
    def test_weighted_partial_batch_no_wrap_and_state_restoration(self):
        arrays = [np.arange(12, dtype=np.uint16)]
        model = TargetMean().train()
        state = torch.get_rng_state().clone()
        result = evaluate(model, arrays, 2, 2, torch.device('cpu'))
        self.assertEqual(result['tokens'], 10)  # 2+2+1 sequences; one trailing input discarded
        self.assertEqual(result['loss'], 5.5)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        result = evaluate(model, arrays, 2, 2, torch.device('cpu'), 1)
        self.assertEqual(result['tokens'], 4)
        self.assertEqual(result['loss'], 2.5)
        self.assertEqual(len(list(batches(arrays, 2, 2, 99))), 3)

    def test_val_only_and_standalone_single_vs_two_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            manifest = write_shards([{'text': ' hello world' * 100}], data, 25)
            val = validation_data(data, 4)
            self.assertEqual(len(val), 1)
            np.testing.assert_array_equal(val[0], np.load(data/manifest['shards'][0]['file']))
            # A validation-only directory must work; training files are unnecessary.
            for shard in manifest['shards']:
                if shard['split'] == 'train':
                    (data/shard['file']).unlink()
            env = dict(os.environ, OMP_NUM_THREADS='1')
            def run(args):
                r = subprocess.run([sys.executable]+args, cwd=ROOT, env=env,
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
                return r.stdout
            run(['train-gpt2.py', '--device', 'cpu', '--no-compile', '--no-generate',
                 '--n-layer', '1', '--n-head', '1', '--n-embd', '16', '--batch-size', '1',
                 '--seq-len', '4', '--total-batch-size', '4', '--max-steps', '1',
                 '--output-dir', str(root/'ckpt')])
            base = ['eval-gpt2.py', '--checkpoint', str(root/'ckpt/latest.pt'),
                    '--data-dir', str(data), '--device', 'cpu', '--batch-size', '2',
                    '--seq-len', '4', '--max-batches', '0']
            one = json.loads(run(base).strip().splitlines()[-1])
            two = json.loads(run(['-m','torch.distributed.run','--master-addr=127.0.0.1', f'--master-port={free_port()}','--nproc_per_node=2']+base).strip().splitlines()[-1])
            self.assertEqual(one['tokens'], 24)
            self.assertEqual(one['tokens'], two['tokens'])
            self.assertAlmostEqual(one['loss'], two['loss'], places=6)
            # Fewer global batches than ranks: idle ranks must still reduce successfully.
            short = json.loads(run(['-m','torch.distributed.run','--master-addr=127.0.0.1', f'--master-port={free_port()}','--nproc_per_node=2']+base+['--max-batches','1']).strip().splitlines()[-1])
            self.assertEqual(short['tokens'], 8)
            (data/manifest['shards'][0]['file']).unlink()
            with self.assertRaises(FileNotFoundError):
                validation_data(data, 4)
