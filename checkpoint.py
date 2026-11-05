"""Save only at optimizer-step boundaries; keep optimizer, cursor and rank-local RNG."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def load_initial_weights(path, raw_model, model_config):
    """Start a new pretraining phase from weights, without restoring training state."""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported initialization checkpoint format')
    source = payload['config']
    if source.get('model') != model_config:
        raise ValueError('Initialization model configuration differs')
    if source.get('stage') == 'sft' or 'total_batch_size' not in source:
        raise ValueError('--init-from requires a pretraining checkpoint')
    step = payload['next_step']
    if not 0 <= step <= source['max_steps']:
        raise ValueError('Invalid initialization checkpoint step')
    raw_model.load_state_dict(payload['model'], strict=True)
    prior_tokens = source.get('initialization', {}).get('tokens_before', 0)
    with Path(path).open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(checkpoint_sha256=sha256, source_next_step=step,
                tokens_before=prior_tokens + step * source['total_batch_size'])


def save_checkpoint(path, raw_model, optimizer, loader, next_step, run_config, device):
    ddp = dist.is_initialized()
    rank = dist.get_rank() if ddp else 0
    state = dict(loader=loader.state_dict(), rng=rng_state(device))
    states = [None] * dist.get_world_size() if ddp else [state]
    if ddp:
        dist.all_gather_object(states, state)  # every rank must participate
    error = [None]
    if rank == 0:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp')
            payload = dict(format_version=1, next_step=next_step, config=run_config,
                           model=raw_model.state_dict(), optimizer=optimizer.state_dict(), ranks=states)
            with open(temporary, 'wb') as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)  # readers see either the old or the complete new file
        except Exception as exc:
            error[0] = str(exc)
    if ddp:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f'Checkpoint write failed: {error[0]}')


def load_checkpoint(path, raw_model, optimizer, loader, run_config, device):
    # Includes Python/NumPy RNG objects. Load only your own trusted checkpoints.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['format_version'] != 1 or payload['config'] != run_config:
        raise ValueError('Resume configuration differs: retain model, batch, LR horizon, world size, device and precision settings')
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(payload['ranks']) != run_config['world_size']:
        raise ValueError('Resume world size differs')
    if not 0 <= payload['next_step'] <= run_config['max_steps']:
        raise ValueError('Invalid checkpoint step')
    loader.load_state_dict(payload['ranks'][rank]['loader'])
    raw_model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['ranks'][rank]['rng'], device)
    return payload['next_step']
