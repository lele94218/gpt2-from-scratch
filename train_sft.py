"""Single-GPU full-parameter SFT: same GPT-2, assistant-only next-token loss."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chat_data import IGNORE, TEMPLATE, ConversationLoader, collate, fingerprint, load_data
from checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from model_io import load_gpt


def precision_context(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def loss_sum(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                           ignore_index=IGNORE, reduction='sum')


def backward_update(model, batches, device):
    # Average over ALL answer tokens in this optimizer update. Micro-batches can
    # have unequal answer lengths, so mean(loss per micro-batch) would be wrong.
    count = sum(int((y != IGNORE).sum()) for _, y in batches)
    if not count:
        raise ValueError('Update has no assistant targets')
    total = torch.zeros((), device=device)
    for x, y in batches:
        with precision_context(device):
            summed = loss_sum(model, x.to(device), y.to(device))
        (summed / count).backward()
        total += summed.detach().float()
    return float(total / count), count


@torch.no_grad()
def evaluate(model, records, batch_size, seq_len, device):
    state, was_training = rng_state(device), model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for i in range(0, len(records), batch_size):
            x, y = collate(records[i:i+batch_size], seq_len)
            with precision_context(device):
                summed = loss_sum(model, x.to(device), y.to(device))
            total += float(summed)
            count += int((y != IGNORE).sum())
        return dict(loss=total / count, answer_tokens=count)
    finally:
        model.train(was_training)
        restore_rng(state, device)


def learning_rate(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-from', help='Trusted base checkpoint: load weights only; start a new optimizer')
    source.add_argument('--resume', help='Trusted SFT checkpoint: restore optimizer, cursor and RNG too')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--accum-steps', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--warmup-fraction', type=float, default=0.03)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=100, help='Also evaluate at exit; 0 disables')
    p.add_argument('--stop-after', type=int, help='Stop after this absolute update; retain full LR horizon for resume')
    args = p.parse_args()
    if 'RANK' in os.environ:
        p.error('This learning implementation is single-process: use python, not torchrun')
    if min(args.batch_size, args.seq_len, args.accum_steps, args.epochs, args.checkpoint_every) <= 0:
        p.error('Batch dimensions, epochs and checkpoint interval must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.warmup_fraction < 1 or args.eval_every < 0:
        p.error('Invalid learning rate, warmup fraction or evaluation interval')
    return args


def main():
    args = parse_args()
    kind = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(kind)
    if kind == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA SFT requires BF16 support')
    output = Path(args.output_dir) / 'latest.pt'
    if output.exists() and (not args.resume or output.resolve() != Path(args.resume).resolve()):
        raise FileExistsError('Output checkpoint exists; resume that file or use a new output directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    data, data_hash = load_data(args.data_dir, args.seq_len)
    raw_model, payload = load_gpt(args.resume or args.init_from)
    if args.seq_len > raw_model.config.block_size or raw_model.config.vocab_size < 50257:
        raise ValueError('Checkpoint context/vocabulary incompatible with GPT-2 chat data')
    if args.resume and payload['config'].get('stage') != 'sft':
        raise ValueError('Use --init-from for a base checkpoint, not --resume')
    initial_sha = payload['config']['initial_checkpoint_sha256'] if args.resume else fingerprint(args.init_from)
    del payload
    raw_model.to(device)
    optimizer = raw_model.configure_optimizers(0.0, args.lr, kind, True)
    model = torch.compile(raw_model) if args.compile else raw_model
    loader = ConversationLoader(data['train'], args.batch_size, args.accum_steps, args.seq_len, args.seed)
    per_epoch = math.ceil(len(data['train']) / (args.batch_size * args.accum_steps))
    steps = per_epoch * args.epochs
    warmup = min(steps - 1, math.ceil(steps * args.warmup_fraction))
    config = dict(stage='sft', model=asdict(raw_model.config), template=TEMPLATE,
                  initial_checkpoint_sha256=initial_sha,
                  data_sha256=data_hash, batch_size=args.batch_size, seq_len=args.seq_len,
                  accum_steps=args.accum_steps, epochs=args.epochs, max_steps=steps,
                  lr=args.lr, warmup_steps=warmup, seed=args.seed, world_size=1,
                  device=kind, precision='bf16' if kind == 'cuda' else 'fp32',
                  compile=args.compile, torch_version=str(torch.__version__))
    start = load_checkpoint(args.resume, raw_model, optimizer, loader, config, device) if args.resume else 0
    end = args.stop_after if args.stop_after is not None else steps
    if not 0 < end <= steps or end < start:
        raise ValueError(f'stop-after must be in 1..{steps} and not precede checkpoint step {start}')
    print(f'SFT: {len(data["train"])} conversations; {steps} updates; steps {start}..{end-1}', flush=True)
    model.train()
    for step in range(start, end):
        if kind == 'cuda':
            torch.cuda.synchronize()
        began = time.perf_counter()
        batches = loader.next_update()
        optimizer.zero_grad(set_to_none=True)
        loss, answer_tokens = backward_update(model, batches, device)
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        lr = learning_rate(step, steps, args.lr, warmup)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if kind == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        print(json.dumps(dict(step=step, loss=loss, lr=lr, grad_norm=float(norm),
                              answer_tokens=answer_tokens, input_slots=sum(x.numel() for x, _ in batches),
                              seconds=seconds)), flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == end:
            save_checkpoint(output, raw_model, optimizer, loader, step + 1, config, device)
            print(f'checkpoint: {output} (next step {step+1})', flush=True)
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step + 1 == end):
            result = evaluate(raw_model, data['val'], args.batch_size, args.seq_len, device)
            print(json.dumps(dict(validation_step=step + 1, **result)), flush=True)


if __name__ == '__main__':
    main()
