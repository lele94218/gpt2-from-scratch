"""Select complete short GPT-2 conversations from pinned Smol-SmolTalk data."""
import argparse
from collections import Counter
import json
from pathlib import Path

import tiktoken

from chat_data import TEMPLATE, conversation_key, encode_conversation, fingerprint

DATASET = 'HuggingFaceTB/smol-smoltalk'
REVISION = 'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'


def write_split(rows, path, count, seq_len, seen):
    if count < 0:
        raise ValueError('Example count must be nonnegative; 0 selects all eligible examples')
    enc = tiktoken.get_encoding('gpt2')
    stats = Counter()
    sources = Counter()
    with open(path, 'x') as out:
        for row in rows:
            stats['scanned'] += 1
            try:
                messages = row['messages']
                record = encode_conversation(messages, enc)
            except (KeyError, ValueError, TypeError):
                stats['invalid'] += 1
                continue
            if len(record['input_ids']) > seq_len:
                stats['too_long'] += 1
                continue
            key = conversation_key(messages)
            if key in seen:
                stats['duplicate'] += 1
                continue
            seen.add(key)
            out.write(json.dumps(record) + '\n')
            sources[row.get('source', 'local')] += 1
            stats['examples'] += 1
            stats['input_tokens'] += len(record['input_ids'])
            stats['answer_tokens'] += sum(t != -100 for t in record['labels'])
            if count > 0 and stats['examples'] == count:
                break
    if not stats['examples']:
        raise ValueError('No eligible examples found; refusing to publish an empty split')
    if count > 0 and stats['examples'] != count:
        raise ValueError(f'Only found {stats["examples"]}/{count} eligible examples; use a smaller count or longer seq-len')
    return dict(stats, sources=dict(sources), sha256=fingerprint(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--train-examples', type=int, default=10000, help='Number of eligible training examples; 0 = all')
    p.add_argument('--val-examples', type=int, default=500, help='Number of eligible validation examples; 0 = all')
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.train_examples, args.val_examples) < 0 or not 1 <= args.seq_len <= 1024:
        p.error('Counts must be nonnegative (0 = all); seq-len must be in 1..1024')
    from datasets import load_dataset
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        p.error('Use an empty output directory (incomplete preparations are not resumable)')
    manifest = dict(format_version=1, tokenizer='gpt2', template=TEMPLATE,
                    dataset=DATASET, revision=REVISION, license='apache-2.0',
                    seq_len=args.seq_len, seed=args.seed, shuffle_buffer=10000,
                    requested_examples=dict(train=args.train_examples, val=args.val_examples), splits={})
    seen = set()
    for split, source_split, count in [('train', 'train', args.train_examples), ('val', 'test', args.val_examples)]:
        rows = load_dataset(DATASET, revision=REVISION, split=source_split, streaming=True)
        rows = rows.shuffle(seed=args.seed, buffer_size=10000)
        manifest['splits'][split] = write_split(rows, root / f'{split}.jsonl', count, args.seq_len, seen)
        print(split, json.dumps(manifest['splits'][split]), flush=True)
    # Publish completion last. The loader refuses directories without this manifest.
    temporary = root / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(root / 'manifest.json')


if __name__ == '__main__':
    main()
