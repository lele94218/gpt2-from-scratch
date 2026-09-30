"""Reuse the learning model without moving or editing train-gpt2.py."""
import importlib.util
from pathlib import Path
import sys

import torch


def load_gpt(path):
    # These checkpoints include Python/NumPy RNG objects: only load trusted files.
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    name = 'gpt2_learning_model'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('train-gpt2.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    model = module.GPT(module.GPTConfig(**payload['config']['model']))
    model.load_state_dict(payload['model'], strict=True)
    return model, payload
