import tempfile
from pathlib import Path
import unittest

import torch

from checkpoint import load_checkpoint
from export_weights import export_weights


class ExportTests(unittest.TestCase):
    def test_export_preserves_tensors_and_removes_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root/'full.pt', root/'weights.pt'
            tied = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            payload = dict(format_version=1, next_step=7, config=dict(stage='sft', model={}),
                           model=dict(embedding=tied, output=tied),
                           optimizer={'private_training_state': 1}, ranks=[{'rng': 'not exported'}])
            torch.save(payload, source)
            original = source.read_bytes()
            info = export_weights(source, target)
            result = torch.load(target, weights_only=True)
            self.assertEqual(result['artifact_type'], 'weights_only')
            self.assertEqual(result['next_step'], 7)
            self.assertEqual(result['config'], payload['config'])
            self.assertNotIn('optimizer', result)
            self.assertNotIn('ranks', result)
            self.assertTrue(torch.equal(result['model']['embedding'], tied))
            self.assertEqual(result['model']['embedding'].data_ptr(), result['model']['output'].data_ptr())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(len(info['sha256']), 64)
            self.assertEqual(result['provenance']['source_checkpoint_sha256'], info['source_checkpoint_sha256'])
            with self.assertRaisesRegex(ValueError, 'Weights-only'):
                load_checkpoint(target, None, None, None, {}, torch.device('cpu'))
            with self.assertRaises(FileExistsError):
                export_weights(source, target)
            with self.assertRaises(FileExistsError):
                export_weights(source, source)


if __name__ == '__main__':
    unittest.main()
