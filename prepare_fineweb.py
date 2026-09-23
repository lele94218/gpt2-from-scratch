"""Stream FineWeb(-Edu), tokenize with GPT-2, write uint16 .npy shards.

The first shard is held out as validation data. Training uses only train shards.
The manifest is written last: an interrupted preparation is not a usable dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken

from data import file_digest


def write_shards(documents, output_dir, shard_size, max_tokens=None, source=None):
    if shard_size < 2 or (max_tokens is not None and max_tokens <= shard_size):
        raise ValueError('Need shard_size >= 2 and max_tokens > shard_size for train + val')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Output directory must be empty; do not mix dataset preparations')
    enc = tiktoken.get_encoding('gpt2')
    buffer = np.empty(shard_size, dtype=np.uint16)
    filled = total = 0
    shards = []

    def flush(count):
        split = 'val' if not shards else 'train'
        path = root / f'fineweb_{split}_{len(shards):06d}.npy'
        np.save(path, buffer[:count], allow_pickle=False)
        shards.append(dict(file=path.name, split=split, tokens=count, sha256=file_digest(path)))
        print(f'{path.name}: {count} tokens', flush=True)

    for doc in documents:
        tokens = [enc.eot_token] + enc.encode_ordinary(doc['text'])
        if max_tokens is not None:
            tokens = tokens[:max_tokens - total]
        offset = 0
        # One document may span arbitrarily many shards.
        while offset < len(tokens):
            n = min(shard_size - filled, len(tokens) - offset)
            buffer[filled:filled+n] = tokens[offset:offset+n]
            filled += n
            total += n
            offset += n
            if filled == shard_size:
                flush(filled)
                filled = 0
        if max_tokens is not None and total >= max_tokens:
            break
    if filled:
        flush(filled)
    if len(shards) < 2:
        raise ValueError('Not enough tokens for train and val; retry in an empty directory with smaller shards')
    manifest = dict(tokenizer='gpt2', source=source, total_tokens=total, shards=shards)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['HuggingFaceFW/fineweb-edu', 'HuggingFaceFW/fineweb'], default='HuggingFaceFW/fineweb-edu')
    parser.add_argument('--config', default='sample-10BT')
    parser.add_argument('--revision', default='main', help='Use a dataset commit for a reproducible download')
    parser.add_argument('--output-dir', default='data/fineweb-edu')
    parser.add_argument('--shard-size', type=int, default=100_000_000)
    parser.add_argument('--max-tokens', type=int, help='Optional cap for a small pilot run')
    args = parser.parse_args()
    from datasets import load_dataset
    docs = load_dataset(args.dataset, name=args.config, revision=args.revision, split='train', streaming=True)
    write_shards(docs, args.output_dir, args.shard_size, args.max_tokens,
                 dict(dataset=args.dataset, config=args.config, revision=args.revision))


if __name__ == '__main__':
    main()
