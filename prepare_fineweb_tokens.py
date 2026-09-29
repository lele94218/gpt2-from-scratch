"""Download Karpathy's GPT-2 token shards and convert llm.c BIN to loader-ready NPY.

No tokenization is performed. Re-running the same selection reuses downloads and
checks existing NPY content. The completion manifest is published last.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DATASET = 'karpathy/fineweb-edu-100B-gpt2-token-shards'
REVISION = 'a33f75d78c7f74236fb03754ec1eb3cc77507d64'
TOKENS_PER_SHARD = 100_000_000
CHUNK = 1_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def convert_shard(source, target, expected_tokens=TOKENS_PER_SHARD):
    """Validate a GPT-2 llm.c shard and preserve its token IDs exactly."""
    source, target = Path(source), Path(target)
    with source.open('rb') as f:
        header = np.fromfile(f, dtype='<i4', count=256)
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        raise ValueError(f'Invalid GPT-2 header: {source}')
    count = int(header[2])
    if count <= 0 or count != expected_tokens:
        raise ValueError(f'Unexpected token count {count}: {source}')
    if source.stat().st_size != 1024 + count * 2:
        raise ValueError(f'Invalid file size: {source}')
    tokens = np.memmap(source, mode='r', dtype='<u2', offset=1024, shape=(count,))
    try:
        for start in range(0, count, CHUNK):
            if int(tokens[start:start + CHUNK].max()) > 50256:
                raise ValueError(f'Invalid GPT-2 token ID: {source}')
        if target.exists():
            existing = np.load(target, mmap_mode='r', allow_pickle=False)
            try:
                if existing.shape != (count,) or existing.dtype != np.uint16:
                    raise ValueError(f'Existing NPY shape/dtype mismatch: {target}')
                for start in range(0, count, CHUNK):
                    if not np.array_equal(existing[start:start + CHUNK], tokens[start:start + CHUNK]):
                        raise ValueError(f'Existing NPY content mismatch: {target}')
            finally:
                del existing
        else:
            temporary = target.with_name(target.name + '.tmp')
            with temporary.open('wb') as f:
                np.save(f, tokens, allow_pickle=False)
            temporary.replace(target)
    finally:
        del tokens
    return dict(file=target.name, tokens=count, sha256=digest(target))


def prepare(output_dir, cache_dir, train_shards, revision=REVISION, download=None):
    if not 1 <= train_shards <= 999:
        raise ValueError('train-shards must be between 1 and 999')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('revision must be a full lowercase dataset commit SHA')
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selection = [('val', 0)] + [('train', i) for i in range(1, train_shards + 1)]
    filenames = [f'fineweb_{split}_{i:06d}.npy' for split, i in selection]
    plan = dict(dataset=DATASET, revision=revision, files=filenames)
    plan_path = root / '.preparation.json'
    manifest_path = root / 'manifest.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Preparation selection differs; use a new output directory')
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        source = old.get('source', {})
        if (old.get('tokenizer') != 'gpt2' or source.get('dataset') != DATASET
                or source.get('revision') != revision
                or [s['file'] for s in old['shards']] != filenames):
            raise ValueError('Existing manifest differs; use a new output directory')
    allowed = set(filenames) | {f + '.tmp' for f in filenames}
    allowed |= {'.preparation.json', '.preparation.json.tmp', 'manifest.json', 'manifest.json.tmp'}
    if any(p.name not in allowed for p in root.iterdir()):
        raise ValueError('Unexpected files in output directory; use a separate directory')
    atomic_json(plan_path, plan)
    # A manifest must never advertise completion of a failed/in-progress verification.
    manifest_path.unlink(missing_ok=True)
    entries = []
    for done, (split, index) in enumerate(selection, 1):
        name = f'edu_fineweb_{split}_{index:06d}.bin'
        print(f'[{done}/{len(selection)}] {name}', flush=True)
        source = download(repo_id=DATASET, repo_type='dataset', filename=name,
                          revision=revision, local_dir=str(cache_dir))
        entry = convert_shard(source, root / filenames[done - 1])
        entry['split'] = split
        entries.append(entry)
        print(f"  verified: {entry['tokens']:,} tokens", flush=True)
    manifest = dict(tokenizer='gpt2', source=dict(dataset=DATASET, revision=revision,
                    format='llm.c GPT-2', train_shards=train_shards),
                    total_tokens=sum(s['tokens'] for s in entries), shards=entries)
    atomic_json(manifest_path, manifest)
    for split in ('train', 'val'):
        shards = [s for s in entries if s['split'] == split]
        print(f"{split}: {len(shards)} shards, {sum(s['tokens'] for s in shards):,} tokens")
    print(f'READY: {manifest_path}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-shards', type=int, default=100,
                        help='100M tokens per shard; 100 = 10B training tokens, plus one val shard')
    parser.add_argument('--output-dir', default='data/fineweb-pretokenized-10B')
    parser.add_argument('--cache-dir', default='data/fineweb-bin-cache')
    parser.add_argument('--revision', default=REVISION, help='Pinned dataset commit SHA')
    args = parser.parse_args()
    prepare(args.output_dir, args.cache_dir, args.train_shards, args.revision)


if __name__ == '__main__':
    main()
