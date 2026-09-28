"""Sequential token loading with a shared DDP cursor and resumable shard position."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken
import torch


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class TokenLoader:
    def __init__(self, B, T, rank, world_size, input_file=None, data_dir=None):
        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.stride = B * T * world_size
        self.shard, self.position = 0, 0  # position is the global round start
        self.arrays = []
        identity = []
        if data_dir:
            root = Path(data_dir)
            manifest = json.loads((root / 'manifest.json').read_text())
            if manifest['tokenizer'] != 'gpt2':
                raise ValueError('Expected GPT-2 tokenization')
            for entry in manifest['shards']:
                if entry['split'] != 'train':
                    continue
                path = root / entry['file']
                if file_digest(path) != entry['sha256']:
                    raise ValueError(f'Shard checksum mismatch: {path}')
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.ndim != 1 or array.dtype != np.uint16 or len(array) != entry['tokens']:
                    raise ValueError(f'Invalid shard: {path}')
                if len(array) and int(array.max()) > 50256:
                    raise ValueError(f'Invalid GPT-2 token ID: {path}')
                # A short last shard cannot supply one synchronized DDP round.
                if len(array) < self.stride + 1:
                    if rank == 0:
                        print(f'skipping short train shard: {path.name}', flush=True)
                    continue
                self.arrays.append(array)
                identity.append((path.name, entry['sha256']))
        else:
            path = Path(input_file)
            tokens = tiktoken.get_encoding('gpt2').encode(path.read_text())
            self.arrays = [np.asarray(tokens, dtype=np.uint16)]
            identity = [('text', file_digest(path))]
        if not self.arrays or any(len(a) < self.stride + 1 for a in self.arrays):
            raise ValueError('No usable training data: need at least B*T*world_size+1 tokens per shard')
        self.fingerprint = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        if rank == 0:
            print(f'loaded {len(self.arrays)} train shard(s), {sum(map(len, self.arrays))} tokens', flush=True)

    def next_batch(self):
        start = self.position + self.B * self.T * self.rank
        buf = torch.from_numpy(np.array(self.arrays[self.shard][start:start + self.B*self.T + 1], dtype=np.int64))
        x, y = buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)
        self.position += self.stride
        if self.position + self.stride + 1 > len(self.arrays[self.shard]):
            self.shard = (self.shard + 1) % len(self.arrays)
            self.position = 0
        return x, y

    def state_dict(self):
        return dict(shard=self.shard, position=self.position, fingerprint=self.fingerprint,
                    B=self.B, T=self.T, world_size=self.world_size)

    def load_state_dict(self, state):
        for key in ('fingerprint', 'B', 'T', 'world_size'):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f'Data loader resume mismatch: {key}')
        shard, position = state['shard'], state['position']
        if not (0 <= shard < len(self.arrays)) or position < 0 or position % self.stride:
            raise ValueError('Invalid checkpoint data cursor')
        if position + self.stride + 1 > len(self.arrays[shard]):
            raise ValueError('Checkpoint cursor exceeds shard')
        self.shard, self.position = shard, position
