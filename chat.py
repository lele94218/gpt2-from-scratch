"""Terminal chat with the exact GPT-2 role template used for SFT."""
import argparse

import tiktoken
import torch

from chat_data import TEMPLATE, encode_prompt
from model_io import load_gpt
from train_sft import precision_context


@torch.no_grad()
def reply(model, messages, enc, device, max_new_tokens, temperature, top_k):
    ids = encode_prompt(messages, enc)
    if len(ids) >= model.config.block_size:
        raise ValueError('Conversation fills the context window; use /reset or a shorter prompt')
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(min(max_new_tokens, model.config.block_size - len(ids))):
        with precision_context(device):
            logits, _ = model(x)
        # The model has 50304 rows, but GPT-2 can only decode IDs 0..50256.
        scores = logits[0, -1, :enc.n_vocab].float()
        if temperature == 0:
            token = int(scores.argmax())
        else:
            values, indices = torch.topk(scores / temperature, min(top_k, enc.n_vocab))
            token = int(indices[torch.multinomial(values.softmax(-1), 1)].item())
        if token == enc.eot_token:
            return enc.decode(generated), True
        generated.append(token)
        x = torch.cat((x, torch.tensor([[token]], device=device)), dim=1)
    return enc.decode(generated), False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Trusted checkpoint')
    p.add_argument('--prompt', help='One-shot prompt; omit for interactive chat')
    p.add_argument('--allow-base', action='store_true', help='Use a base checkpoint for a before/after comparison')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.7, help='0 = greedy')
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.max_new_tokens <= 0 or args.top_k <= 0 or not 0 <= args.temperature < float('inf'):
        p.error('Invalid generation settings')
    kind = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(kind)
    if kind == 'cuda' and not torch.cuda.is_bf16_supported():
        p.error('CUDA chat requires BF16 support')
    model, payload = load_gpt(args.checkpoint)
    config = payload['config']
    if config.get('stage') == 'sft':
        if config.get('template') != TEMPLATE:
            p.error('Checkpoint uses a different chat template')
    elif not args.allow_base:
        p.error('Use an SFT checkpoint, or --allow-base for comparison')
    del payload
    enc = tiktoken.get_encoding('gpt2')
    if model.config.vocab_size < enc.n_vocab:
        p.error('Checkpoint vocabulary is smaller than GPT-2')
    model.to(device).eval()
    torch.manual_seed(args.seed)
    history = []
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input('You (/reset, /quit): ')
        except (EOFError, KeyboardInterrupt):
            break
        if args.prompt is None and prompt == '/quit':
            break
        if args.prompt is None and prompt == '/reset':
            history = []
            continue
        messages = history + [dict(role='user', content=prompt)]
        try:
            answer, stopped = reply(model, messages, enc, device, args.max_new_tokens, args.temperature, args.top_k)
        except ValueError as exc:
            if args.prompt is not None:
                p.error(str(exc))
            print(exc)
            continue
        print('Assistant:', answer)
        if not stopped:
            print('[Generation limit reached; this turn was not added to history.]')
        elif answer.strip():
            history = messages + [dict(role='assistant', content=answer)]
        if args.prompt is not None:
            break


if __name__ == '__main__':
    main()
