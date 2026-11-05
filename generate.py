"""Generate raw text continuations from a trusted GPT-2 checkpoint."""
import argparse
from contextlib import nullcontext
import math
import sys

import tiktoken
import torch

from model_io import load_gpt


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8, help='0 selects greedy decoding')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.top_k <= 0:
        parser.error('max-new-tokens and top-k must be positive')
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('temperature must be finite and nonnegative')
    kind = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(kind)
    if kind == 'cuda' and not torch.cuda.is_bf16_supported():
        parser.error('CUDA generation requires BF16 support')
    model, payload = load_gpt(args.checkpoint)
    del payload
    enc = tiktoken.get_encoding('gpt2')
    ids = enc.encode_ordinary(args.prompt) or [enc.eot_token]
    room = model.config.block_size - len(ids)
    if room <= 0:
        parser.error(f'Prompt must be shorter than {model.config.block_size} tokens')
    if model.config.vocab_size < enc.n_vocab:
        parser.error('Checkpoint vocabulary is smaller than the GPT-2 tokenizer')
    model.to(device).eval()
    torch.manual_seed(args.seed)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    stopped_at_eot = False
    for _ in range(min(args.max_new_tokens, room)):
        context = torch.autocast('cuda', dtype=torch.bfloat16) if kind == 'cuda' else nullcontext()
        with context:
            logits, _ = model(x)
        scores = logits[0, -1, :enc.n_vocab].float()
        if args.temperature == 0:
            token = int(scores.argmax().item())
        else:
            values, indices = torch.topk(scores / args.temperature, min(args.top_k, enc.n_vocab))
            token = int(indices[torch.multinomial(values.softmax(-1), 1)].item())
        if token == enc.eot_token:
            stopped_at_eot = True
            break
        generated.append(token)
        x = torch.cat([x, torch.tensor([[token]], dtype=torch.long, device=device)], dim=1)
    print(args.prompt + enc.decode(generated))
    if not stopped_at_eot and args.max_new_tokens > room:
        print(f'[Reached the {model.config.block_size}-token context limit.]', file=sys.stderr)


if __name__ == '__main__':
    main()
