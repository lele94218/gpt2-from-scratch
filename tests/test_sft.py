"""SFT contracts: target alignment, token weighting, data integrity and restart."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import tiktoken
import torch

from chat_data import (IGNORE, TEMPLATE, ConversationLoader, collate,
                       encode_conversation, encode_prompt, load_data, render)
from prepare_chat import write_split
from train_sft import backward_update, evaluate
from chat import reply
from test_training import assert_nested_equal

ROOT = Path(__file__).resolve().parents[1]


def messages(question='Hi', answer='Hello!'):
    return [dict(role='user', content=question), dict(role='assistant', content=answer)]


def prepare_fixture(root):
    root.mkdir()
    seen = set()
    manifest = dict(format_version=1, template=TEMPLATE, tokenizer='gpt2', splits={})
    for split, count in [('train', 5), ('val', 3)]:
        rows = [dict(messages=messages(f'{split} question {i}', 'A short answer.' if i % 2 else 'Yes.'))
                for i in range(count)]
        manifest['splits'][split] = write_split(rows, root/f'{split}.jsonl', count, 32, seen)
    (root/'manifest.json').write_text(json.dumps(manifest))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8)

    def forward(self, x):
        return self.output(self.embedding(x)), None


class SFTTests(unittest.TestCase):
    def test_zero_count_scans_all_and_rejects_empty_or_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = dict(messages=messages())
            rows = [good, good, dict(messages=messages(answer='long ' * 100)),
                    dict(messages=[]), dict(messages=messages('Why?', 'Because.'))]
            stats = write_split(rows, root/'all.jsonl', 0, 32, set())
            self.assertEqual(stats['scanned'], 5)
            self.assertEqual(stats['examples'], 2)
            self.assertEqual(stats['duplicate'], 1)
            self.assertEqual(stats['too_long'], 1)
            self.assertEqual(stats['invalid'], 1)
            self.assertEqual(len((root/'all.jsonl').read_text().splitlines()), 2)
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                write_split([], root/'empty.jsonl', 0, 32, set())
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                write_split(rows, root/'negative.jsonl', -1, 32, set())
            with self.assertRaisesRegex(ValueError, 'Only found 2/3'):
                write_split(rows, root/'short.jsonl', 3, 32, set())

    def test_chat_stops_at_eot_and_excludes_padded_vocab(self):
        enc = tiktoken.get_encoding('gpt2')
        class EndingModel(torch.nn.Module):
            config = SimpleNamespace(block_size=64)

            def forward(self, x):
                logits = torch.zeros(1, x.shape[1], 50304)
                logits[..., enc.eot_token] = 10
                logits[..., 50303] = 100  # Must never be sampled: no tokenizer entry.
                return logits, None
        text, stopped = reply(EndingModel(), messages()[:1], enc, torch.device('cpu'), 5, 0, 50)
        self.assertEqual(text, '')
        self.assertTrue(stopped)

    def test_shift_mask_eot_padding_and_prompt_prefix(self):
        enc = tiktoken.get_encoding('gpt2')
        conversation = [dict(role='system', content='Be concise.')] + messages()
        conversation += messages('Why?', 'Because it works.')
        record = encode_conversation(conversation)
        ids, mask = render(conversation, enc)
        self.assertEqual(record['input_ids'], ids[:-1])
        expected_answer_ids = (enc.encode_ordinary('Hello!') + [enc.eot_token]
                               + enc.encode_ordinary('Because it works.') + [enc.eot_token])
        self.assertEqual([t for t in record['labels'] if t != IGNORE], expected_answer_ids)
        # Inference supplies exactly the prefix that training used before the final answer.
        prompt = encode_prompt(conversation[:-1], enc)
        self.assertEqual(ids[:len(prompt)], prompt)
        self.assertEqual(record['labels'][len(prompt)-1], enc.encode_ordinary('Because it works.')[0])
        self.assertEqual(record['labels'][-1], enc.eot_token)
        x, y = collate([record], len(ids)+4)
        self.assertTrue((y[0, len(ids)-1:] == IGNORE).all())
        self.assertTrue((x[0, len(ids)-1:] == enc.eot_token).all())
        self.assertEqual(record['labels'].count(enc.eot_token), 2)
        with self.assertRaises(ValueError):
            encode_conversation([dict(role='user', content='No answer')])

    def test_filter_complete_examples_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = dict(messages=messages())
            seen = set()
            rows = [dict(messages=messages(answer='too long ' * 100)), good, good,
                    dict(messages=messages('Another?', 'Yes.'))]
            stats = write_split(rows, root/'selection.jsonl', 2, 32, seen)
            self.assertEqual(stats['too_long'], 1)
            self.assertEqual(stats['duplicate'], 1)
            self.assertEqual(stats['examples'], 2)
            # Same conversation is also rejected when preparing the validation split.
            stats = write_split([good, dict(messages=messages('Held out?', 'Indeed.'))],
                                root/'heldout.jsonl', 1, 32, seen)
            self.assertEqual(stats['duplicate'], 1)
            prepare_fixture(root/'data')
            data, _ = load_data(root/'data', 32)
            self.assertEqual(len(data['train']), 5)
            with self.assertRaises(ValueError):
                load_data(root/'data', 2)
            with (root/'data/train.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_data(root/'data', 32)

    def test_unequal_answer_lengths_match_full_batch_gradient(self):
        torch.manual_seed(9)
        a = TinyModel()
        b = copy.deepcopy(a)
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [2, 3, 4]])
        y = torch.tensor([[IGNORE, IGNORE, 2], [1, 2, 3], [IGNORE, 4, 5]])
        one, count = backward_update(a, [(x, y)], torch.device('cpu'))
        many, other_count = backward_update(b, [(x[:1], y[:1]), (x[1:], y[1:])], torch.device('cpu'))
        self.assertEqual(count, 6)
        self.assertEqual(count, other_count)
        self.assertAlmostEqual(one, many, places=6)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad)

    def test_partial_epoch_loader_and_evaluation_state(self):
        records = [encode_conversation(messages(str(i), 'Hi')) for i in range(5)]
        loader = ConversationLoader(records, 2, 2, 16, 7)
        self.assertEqual(sum(len(x) for x, _ in loader.next_update()), 4)
        state = loader.state_dict()
        last = loader.next_update()
        self.assertEqual(sum(len(x) for x, _ in last), 1)
        restored = ConversationLoader(records, 2, 2, 16, 7)
        restored.load_state_dict(state)
        assert_nested_equal(self, last, restored.next_update())
        self.assertEqual(sum(len(x) for x, _ in restored.next_update()), 4)
        self.assertEqual(restored.epoch, 1)
        # Model deliberately consumes RNG; evaluation must restore both RNG and mode.
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return torch.rand(*x.shape, 50257), None
        model = Dummy().train()
        rng = torch.get_rng_state().clone()
        result = evaluate(model, records, 2, 16, torch.device('cpu'))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(result['answer_tokens'], 10)

    def test_restart_and_chat_with_real_gpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_fixture(root/'data')
            env = dict(os.environ, OMP_NUM_THREADS='1')
            def run(argv, ok=True):
                result = subprocess.run([sys.executable] + argv, cwd=ROOT, env=env,
                                        text=True, capture_output=True, timeout=120)
                if ok:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result.stdout+result.stderr
            run(['train-gpt2.py', '--device', 'cpu', '--no-compile', '--no-generate',
                 '--n-layer', '1', '--n-head', '1', '--n-embd', '16', '--batch-size', '1',
                 '--seq-len', '4', '--total-batch-size', '4', '--max-steps', '1',
                 '--output-dir', str(root/'base')])
            base = str(root/'base/latest.pt')
            common = ['train_sft.py', '--device', 'cpu', '--data-dir', str(root/'data'),
                      '--batch-size', '2', '--accum-steps', '2', '--seq-len', '32',
                      '--epochs', '2', '--checkpoint-every', '1', '--eval-every', '1']
            run(common + ['--init-from', base, '--output-dir', str(root/'full')])
            run(common + ['--init-from', base, '--output-dir', str(root/'part'), '--stop-after', '1'])
            run(common + ['--resume', str(root/'part/latest.pt'), '--output-dir', str(root/'part')])
            a = torch.load(root/'full/latest.pt', weights_only=False)
            b = torch.load(root/'part/latest.pt', weights_only=False)
            assert_nested_equal(self, a, b)
            self.assertEqual(a['next_step'], 4)
            self.assertEqual(a['config']['stage'], 'sft')
            # Eval does not change training trajectory; final update remains partial.
            run(common + ['--init-from', base, '--output-dir', str(root/'no-eval'), '--eval-every', '0'])
            assert_nested_equal(self, a, torch.load(root/'no-eval/latest.pt', weights_only=False))
            text = run(['chat.py', '--checkpoint', str(root/'full/latest.pt'), '--device', 'cpu',
                        '--prompt', 'Hi', '--max-new-tokens', '2', '--temperature', '0'])
            self.assertIn('Assistant:', text)
            text = run(common + ['--resume', base, '--output-dir', str(root/'bad')], ok=False)
            self.assertIn('Use --init-from', text)
            text = run(common + ['--resume', str(root/'part/latest.pt'), '--output-dir', str(root/'part'),
                                 '--lr', '0.001'], ok=False)
            self.assertIn('Resume configuration differs', text)
            text = run(common + ['--init-from', base, '--output-dir', str(root/'full')], ok=False)
            self.assertIn('Output checkpoint exists', text)


if __name__ == '__main__':
    unittest.main()
