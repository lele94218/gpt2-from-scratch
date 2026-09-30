"""One conversation per example; no packing, truncation, or vocabulary changes."""
import hashlib
import json
from array import array
from pathlib import Path

import tiktoken
import torch

IGNORE = -100
TEMPLATE = 'gpt2-text-roles-eot-v1'


def fingerprint(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def conversation_key(messages):
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def validate_messages(messages, completed=True):
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected a nonempty messages list')
    expected = 'user'
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Every message needs nonempty text')
        role = message.get('role')
        if i == 0 and role == 'system':
            continue
        if role != expected:
            raise ValueError('Expected optional system, then alternating user/assistant messages')
        expected = 'assistant' if role == 'user' else 'user'
    if messages[-1]['role'] != ('assistant' if completed else 'user'):
        raise ValueError('Conversation has an incomplete final turn')


def render(messages, enc):
    """Return token IDs and a same-length assistant-target mask (before shifting)."""
    ids, mask = [], []
    for message in messages:
        header = enc.encode_ordinary(message['role'].capitalize() + ': ')
        body = enc.encode_ordinary(message['content'])
        assistant = message['role'] == 'assistant'
        tail = [enc.eot_token] if assistant else enc.encode_ordinary('\n')
        ids.extend(header + body + tail)
        # Assistant header is supplied by the chat program, not learned as a target.
        mask.extend([False] * len(header) + [assistant] * (len(body) + len(tail)))
    return ids, mask


def encode_conversation(messages, enc=None):
    validate_messages(messages)
    enc = enc or tiktoken.get_encoding('gpt2')
    ids, mask = render(messages, enc)
    # Shift ONCE here. GPT.forward does not shift labels a second time.
    return dict(input_ids=ids[:-1],
                labels=[token if keep else IGNORE for token, keep in zip(ids[1:], mask[1:])])


def encode_prompt(messages, enc):
    validate_messages(messages, completed=False)
    ids, _ = render(messages, enc)
    return ids + enc.encode_ordinary('Assistant: ')


def collate(records, seq_len):
    """Right padding is visible only to padding positions under causal attention."""
    enc = tiktoken.get_encoding('gpt2')
    x = torch.full((len(records), seq_len), enc.eot_token, dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    for i, record in enumerate(records):
        n = len(record['input_ids'])
        if not 0 < n <= seq_len:
            raise ValueError('Example exceeds seq-len; prepare matching data instead of truncating')
        x[i, :n] = torch.tensor(record['input_ids'])
        y[i, :n] = torch.tensor(record['labels'])
    return x, y


class IndexedConversations:
    """Keep byte offsets in RAM; decode only the examples needed by each batch."""
    def __init__(self, path, seq_len):
        self.path = Path(path)
        self.offsets = array('Q')
        with self.path.open('rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                record = json.loads(line)
                x, y = record['input_ids'], record['labels']
                if not 0 < len(x) == len(y) <= seq_len:
                    raise ValueError('Invalid example length; increase seq-len or prepare shorter data')
                if any(type(t) is not int or not 0 <= t < 50257 for t in x):
                    raise ValueError('Invalid GPT-2 input token')
                if any(type(t) is not int or not (t == IGNORE or 0 <= t < 50257) for t in y):
                    raise ValueError('Invalid target token')
                if not any(t != IGNORE for t in y) or y[-1] != 50256:
                    raise ValueError('Example needs assistant targets including final EOT')
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            offsets = self.offsets[index]
            with self.path.open('rb') as f:
                records = []
                for offset in offsets:
                    f.seek(offset)
                    records.append(json.loads(f.readline()))
            return records
        offset = self.offsets[index]  # also handles negative indices and IndexError
        with self.path.open('rb') as f:
            f.seek(offset)
            return json.loads(f.readline())


def load_data(directory, seq_len):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest.get('template') != TEMPLATE or manifest.get('tokenizer') != 'gpt2':
        raise ValueError('Unsupported chat data format/template/tokenizer')
    result = {}
    for split in ('train', 'val'):
        path = root / f'{split}.jsonl'
        entry = manifest['splits'][split]
        if fingerprint(path) != entry['sha256']:
            raise ValueError(f'{split} checksum mismatch')
        records = IndexedConversations(path, seq_len)
        if not records or len(records) != entry['examples']:
            raise ValueError(f'Invalid {split} size')
        result[split] = records
        print(f'Indexed {len(records):,} {split} conversations (on-disk JSONL)', flush=True)
    return result, fingerprint(root / 'manifest.json')


class ConversationLoader:
    """Deterministic epoch shuffle; retain partial last updates instead of wrapping."""
    def __init__(self, records, batch_size, accum_steps, seq_len, seed):
        self.records = records
        self.batch_size, self.accum_steps = batch_size, accum_steps
        self.seq_len, self.seed = seq_len, seed
        self.epoch, self.cursor = 0, 0
        self._shuffle()

    def _shuffle(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(len(self.records), generator=generator).tolist()

    def next_update(self):
        if self.cursor == len(self.records):
            self.epoch += 1
            self.cursor = 0
            self._shuffle()
        end = min(self.cursor + self.batch_size * self.accum_steps, len(self.records))
        selected = [self.records[i] for i in self.order[self.cursor:end]]
        self.cursor = end
        return [collate(selected[i:i+self.batch_size], self.seq_len)
                for i in range(0, len(selected), self.batch_size)]

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor)

    def load_state_dict(self, state):
        if state['epoch'] < 0 or not 0 <= state['cursor'] <= len(self.records):
            raise ValueError('Invalid conversation cursor')
        self.epoch, self.cursor = state['epoch'], state['cursor']
        self._shuffle()
