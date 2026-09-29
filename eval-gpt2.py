"""Evaluate a trusted training checkpoint without resuming or updating it."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from evaluation import evaluate, validation_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Trusted checkpoint; may contain Python pickle objects')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=1024)
    p.add_argument('--max-batches', type=int, default=20, help='Global batches, independent of rank count; 0 = all')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = p.parse_args()
    if min(args.batch_size, args.seq_len) <= 0 or args.max_batches < 0:
        p.error('Invalid batch dimensions or limit')
    ddp = 'RANK' in os.environ
    kind = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if ddp and kind == 'mps':
        p.error('MPS evaluation supports one process only')
    device = torch.device(f'cuda:{os.environ.get("LOCAL_RANK", "0")}' if kind == 'cuda' else kind)
    if kind == 'cuda':
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            p.error('CUDA evaluation requires BF16 support')
    try:
        if ddp:
            dist.init_process_group('nccl' if kind == 'cuda' else 'gloo')
        spec = importlib.util.spec_from_file_location('gpt2_training', Path(__file__).with_name('train-gpt2.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if payload.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint format')
        config = module.GPTConfig(**payload['config']['model'])
        if args.seq_len > config.block_size:
            p.error('seq-len exceeds checkpoint context length')
        model = module.GPT(config)
        model.load_state_dict(payload['model'], strict=True)
        step = payload['next_step']
        del payload
        model.to(device)
        arrays = validation_data(args.data_dir, args.seq_len)
        result = evaluate(model, arrays, args.batch_size, args.seq_len, device, args.max_batches)
        if not ddp or dist.get_rank() == 0:
            print(json.dumps(dict(checkpoint_step=step, split='val', device=kind,
                                  seq_len=args.seq_len, batch_size=args.batch_size,
                                  max_batches=args.max_batches, **result)), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
