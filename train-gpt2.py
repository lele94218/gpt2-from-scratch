import math
import os
import tiktoken
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


from dataclasses import dataclass
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        # k, q, v for all head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1   # 随便挂一个属性当 flag

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()

        # (B, T, C) @ (C, 3*C) -> (B, T, 3*C)
        qkv = self.c_attn(x)
        # q,k,v = (B, T, C)
        q,k,v = qkv.split(self.n_embd, dim=2)
        # q,k,v = (B, T, nh, C/head) -> (B, nh, T, C/head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        # att:（B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # y = att @ v
        # y: (B, nh, T, hs) -> (B, T, nh, hs) -> （B, T, nh*hs=C)

        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
        

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1   # 随便挂一个属性当 flag

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def configure_optimizers(self, weight_decay, learning_rate, device, master_process):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 2D 参数(matmul 权重、embedding)做 decay;1D(bias、layernorm)不做
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"decay params: {num_decay:,} | no-decay params: {num_nodecay:,}")
        # fused AdamW: 把逐参数的小 kernel 融合成单个 CUDA kernel
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                      betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward lenght {T}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        loss = None
        
        logits = self.lm_head(x)
        if target is not None:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, self.config.vocab_size), target.view(B*T))

        return logits, loss
        

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
    
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
    
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
    
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
    
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # HuggingFace 用 Conv1D，weight 是转置的
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
    
        return model
        
# The model above is unchanged. The runtime below adds data/checkpoint lifecycle.
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np

from data import TokenLoader
from checkpoint import save_checkpoint, load_checkpoint


def get_lr(it, max_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def parse_args():
    parser = argparse.ArgumentParser(description='GPT-2 training with resumable text/FineWeb data')
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--input-file', default=str(Path(__file__).with_name('input.txt')))
    data.add_argument('--data-dir', help='Directory produced by prepare_fineweb.py')
    parser.add_argument('--max-steps', type=int, default=30, help='Total LR/training horizon; keep unchanged on resume')
    parser.add_argument('--stop-after', type=int, help='Stop at this absolute completed-step count, keeping the LR horizon')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--total-batch-size', type=int, default=524288, help='Global tokens per optimizer update')
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--max-lr', type=float, default=6e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Steps between atomic latest.pt saves; also save at normal exit')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--resume', help='Trusted latest.pt file; same config and data required')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--generate', action=argparse.BooleanOptionalAction, default=True)
    # Defaults remain GPT-2 124M; smaller shapes make lifecycle tests inexpensive.
    parser.add_argument('--n-layer', type=int, default=12)
    parser.add_argument('--n-head', type=int, default=12)
    parser.add_argument('--n-embd', type=int, default=768)
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.seq_len, args.total_batch_size,
           args.checkpoint_every, args.n_layer, args.n_head, args.n_embd) <= 0:
        parser.error('Steps, batch dimensions, checkpoint interval and model dimensions must be positive')
    if args.seq_len > 1024 or args.n_embd % args.n_head:
        parser.error('seq-len must be <= 1024 and n-embd divisible by n-head')
    if args.warmup_steps < 0 or args.max_lr <= 0 or args.weight_decay < 0:
        parser.error('Invalid optimizer/scheduler configuration')
    if args.stop_after is not None and not 0 < args.stop_after <= args.max_steps:
        parser.error('stop-after must be between 1 and max-steps')
    return args


def generate(raw_model, device):
    raw_model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode('yes we can continue'), dtype=torch.long, device=device)
    x = tokens.unsqueeze(0).repeat(5, 1)
    torch.manual_seed(42)
    with torch.no_grad():
        while x.size(1) < 30:
            logits, _ = raw_model(x)
            # The padded vocabulary contains IDs the GPT-2 tokenizer cannot decode.
            probs = F.softmax(logits[:, -1, :enc.n_vocab], dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            x = torch.cat((x, torch.gather(topk_indices, -1, ix)), dim=1)
    for row in x.tolist():
        print('>', enc.decode(row))


def train(args, rank, world_size, device, ddp):
    master = rank == 0
    B, T = args.batch_size, args.seq_len
    if args.total_batch_size % (B * T * world_size):
        raise ValueError('total-batch-size must be divisible by B*T*world_size')
    accum_steps = args.total_batch_size // (B * T * world_size)
    output = Path(args.output_dir) / 'latest.pt'
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use --resume or a new --output-dir')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    loader = TokenLoader(B, T, rank, world_size, args.input_file, args.data_dir)
    model_config = GPTConfig(vocab_size=50304, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    raw_model = GPT(model_config).to(device)  # Keep the original uncompiled module for saving.
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, device.type, master)
    model = torch.compile(raw_model) if args.compile else raw_model
    if ddp:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    run_config = dict(model=asdict(model_config), batch_size=B, seq_len=T,
                      total_batch_size=args.total_batch_size, world_size=world_size,
                      max_steps=args.max_steps, warmup_steps=args.warmup_steps,
                      max_lr=args.max_lr, weight_decay=args.weight_decay, seed=args.seed,
                      device=device.type, compile=args.compile, torch_version=str(torch.__version__),
                      precision='bf16' if device.type == 'cuda' else 'fp32')
    start_step = 0
    if args.resume:
        # Restore after model initialization/wrapping, which may consume RNG.
        start_step = load_checkpoint(args.resume, raw_model, optimizer, loader, run_config, device)
    end_step = args.stop_after or args.max_steps
    if end_step < start_step:
        raise ValueError('stop-after precedes the checkpoint step')
    if master:
        print(f'parameters: {sum(p.numel() for p in raw_model.parameters())}; grad accum: {accum_steps}; steps: {start_step}..{end_step-1}', flush=True)
    model.train()
    for step in range(start_step, end_step):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(accum_steps):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = micro_step == accum_steps - 1
            context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with context:
                _, loss = model(x, y)
            loss = loss / accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= world_size
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, args.max_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if master:
            print(f'step {step}, loss: {loss_accum.item()}, lr: {lr:.4e} norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {args.total_batch_size/dt:.2f}', flush=True)
        # Every rank enters this call; only rank zero writes the shared checkpoint.
        # next_step points to the next update, and the loader already points to its data.
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == end_step:
            save_checkpoint(output, raw_model, optimizer, loader, step + 1, run_config, device)
            if master:
                print(f'checkpoint: {output} (next step {step+1})', flush=True)
    # Checkpoint precedes generation: sampling must not change saved training RNG.
    if master and args.generate:
        generate(raw_model, device)


def main():
    args = parse_args()
    ddp = int(os.environ.get('RANK', -1)) != -1
    rank = int(os.environ['RANK']) if ddp else 0
    local_rank = int(os.environ['LOCAL_RANK']) if ddp else 0
    world_size = int(os.environ['WORLD_SIZE']) if ddp else 1
    kind = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(f'cuda:{local_rank}' if kind == 'cuda' else 'cpu')
    if kind == 'cuda':
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA training requires BF16 support')
    try:
        if ddp:
            init_process_group(backend='nccl' if kind == 'cuda' else 'gloo')
        train(args, rank, world_size, device, ddp)
    finally:
        if dist.is_initialized():
            destroy_process_group()


if __name__ == '__main__':
    main()
