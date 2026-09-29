"""Finite, token-weighted validation; no training cursor or optimizer is touched."""
from contextlib import nullcontext
import math
import numpy as np
import torch
import torch.distributed as dist
from data import TokenLoader
from checkpoint import rng_state, restore_rng


def validation_data(data_dir, seq_len):
    # B=world_size=1 retains every shard containing a full validation sequence.
    return TokenLoader(1, seq_len, 0, 1, data_dir=data_dir, split='val').arrays


def batches(arrays, batch_size, seq_len, max_batches=0):
    """Same global batches for every world size; never wrap or cross shard ends."""
    index = 0
    for array in arrays:
        sequences = (len(array) - 1) // seq_len
        for start in range(0, sequences, batch_size):
            if max_batches and index >= max_batches:
                return
            count = min(batch_size, sequences - start)
            offset = start * seq_len
            buf = torch.from_numpy(np.array(array[offset:offset + count*seq_len + 1], dtype=np.int64))
            yield index, buf[:-1].reshape(count, seq_len), buf[1:].reshape(count, seq_len)
            index += 1


def evaluate(model, arrays, batch_size, seq_len, device, max_batches=0):
    if batch_size <= 0 or seq_len <= 0 or max_batches < 0:
        raise ValueError('Invalid evaluation batch dimensions or limit')
    ddp = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    training = model.training
    saved_rng = rng_state(device)
    # float64 accumulation avoids losing token-count precision on large evaluations.
    # MPS lacks float64, so reduce on CPU there (standalone only).
    totals = torch.zeros(2, dtype=torch.float64, device='cpu' if device.type == 'mps' else device)
    model.eval()
    try:
        with torch.no_grad():
            for index, x, y in batches(arrays, batch_size, seq_len, max_batches):
                if index % world != rank:
                    continue
                context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
                with context:
                    _, loss = model(x.to(device), y.to(device))
                totals[0] += loss.detach().to(totals) * y.numel()
                totals[1] += y.numel()
        if ddp:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() == 0:
            raise ValueError('No complete validation sequences')
        loss = (totals[0] / totals[1]).item()
        return dict(loss=loss, tokens=int(totals[1].item()),
                    perplexity=math.exp(loss) if loss < 700 else float('inf'))
    finally:
        model.train(training)
        restore_rng(saved_rng, device)
