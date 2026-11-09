"""Export a trusted training checkpoint for generation and weight initialization."""
import argparse
import hashlib
from pathlib import Path

import torch


def export_weights(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError('Choose a new output file; the original checkpoint is never replaced')
    payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format_version') != 1 or not {'model', 'config', 'next_step'} <= payload.keys():
        raise ValueError('Expected this repository\'s format-version-1 checkpoint')
    with source.open('rb') as stream:
        source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(format_version=1, artifact_type='weights_only', next_step=payload['next_step'],
                  config=payload['config'], model=payload['model'],
                  provenance=dict(source_checkpoint_sha256=source_sha))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(f'{temporary} already exists; inspect the interrupted export first')
    try:
        torch.save(result, temporary)
        # Release artifacts must also load with PyTorch's restricted weights-only loader.
        verified = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
        if verified['model'].keys() != payload['model'].keys():
            raise ValueError('Exported model keys differ')
        for key, value in payload['model'].items():
            if not torch.equal(value, verified['model'][key]):
                raise ValueError(f'Exported tensor differs: {key}')
        del verified
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with destination.open('rb') as stream:
        output_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=destination.stat().st_size, sha256=output_sha,
                source_checkpoint_sha256=source_sha, next_step=result['next_step'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted full training checkpoint')
    parser.add_argument('--output', required=True, help='New weights-only .pt file')
    args = parser.parse_args()
    import json
    print(json.dumps(export_weights(args.checkpoint, args.output), indent=2))


if __name__ == '__main__':
    main()
