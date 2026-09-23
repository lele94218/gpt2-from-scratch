# GPT-2 from scratch: training lab

My hands-on implementation following Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) and [build-nanogpt](https://github.com/karpathy/build-nanogpt).

The training script is preserved as written during the exercise. This repository adds portable setup instructions and includes Tiny Shakespeare, so it can be cloned onto a Linux NVIDIA GPU machine without copying files from a private workspace.

## Quick start

Requirements: Git, internet access, Linux, an NVIDIA GPU supporting BF16 (such as RTX 3060 or A10), and [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is provisioned by uv. A working host NVIDIA driver is required; this script does not install or change drivers.

```bash
git clone https://github.com/lele94218/gpt2-from-scratch.git
cd gpt2-from-scratch

# Install uv if it is not already available:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Conservative wheel choice for older drivers, including the A10 host's 535 branch:
bash setup.sh cu118
source .venv/bin/activate
python train-gpt2.py
```

`setup.sh cu118` selects PyTorch 2.6.0 with CUDA 11.8 wheels. This is a compatibility profile, not a claim that this exact A10 host has been tested. Run the setup CUDA check and training before committing to a long rental.

For a newer driver, the original training machine used PyTorch 2.11.0 + CUDA 12.8:

```bash
bash setup.sh cu128
source .venv/bin/activate
python train-gpt2.py
```

These profiles follow the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/). The CUDA version displayed by `nvidia-smi` is not the installed PyTorch runtime; check `torch.version.cuda`. CUDA 12.8 wheels need a compatible driver, so do not select that profile solely because an A10 GPU is present.

The first run downloads GPT-2 tokenizer assets and compiles the model. Compilation can take minutes. Separate startup time from steady-state throughput. No separate `flash-attn` package is needed: attention uses PyTorch SDPA.

## Single GPU and DDP

Run from the repository root because the script reads `input.txt` relative to the working directory.

```bash
# Ordinary single-GPU execution
python train-gpt2.py

# Exercise the DDP path with one GPU
torchrun --standalone --nproc_per_node=1 train-gpt2.py

# Two visible GPUs on the same machine
torchrun --standalone --nproc_per_node=2 train-gpt2.py
```

Every rank has a model and optimizer. Data is sharded by rank; gradients synchronize on the final accumulation micro-step. Rank zero prints metrics and generates text after training.

## Current defaults

| Setting | Value |
|---|---:|
| Layers / heads / embedding width | 12 / 12 / 768 |
| Vocabulary size | 50,304 (padded) |
| Parameters | 124,475,904 |
| Micro-batch sequences / sequence length | 4 / 1,024 |
| Global tokens per optimizer step | 524,288 |
| Accumulation micro-steps on one GPU | 128 |
| Optimizer steps | 30 |
| Seed | 1337 |
| Precision / optimizer | BF16 autocast / fused AdamW on CUDA |

Settings are intentionally ordinary Python assignments in `train-gpt2.py`, not command-line flags. Edit `max_steps`, `B, T`, and `total_batch_size` directly for experiments. For a short check, set `max_steps = 5` before running either launch mode. With five steps and the current ten-step warmup, all five steps remain in warmup.

Keep the global batch divisible by `B * T * world_size`. Changing the micro-batch while holding the global batch fixed changes the accumulation count. On the original RTX 3060 run, steady-state throughput was approximately 23,000 tokens/s; 30 default steps took roughly 12 minutes plus startup/generation. This is historical context, not an A10 speed estimate.

## Scope and data

- `input.txt` is the public [Tiny Shakespeare dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), included for immediate execution.
- The current large global batch repeats this small dataset within an optimizer step. It exercises training infrastructure; it is not an optimal Shakespeare training recipe.
- This snapshot does **not** include FineWeb ingestion, validation/HellaSwag, checkpoint saving or resume. Model weights are not saved on exit. Do not use it for a day-long pretraining run expecting recovery or a saved checkpoint.
- High-performance GPU training on a larger pretraining dataset is a follow-up exercise.
- Linux NVIDIA CUDA is the documented target. CPU/MPS branches exist in the learning script but are not validated by this setup.
- Hugging Face `transformers` is only needed if you explicitly use the optional `GPT.from_pretrained()` helper; normal from-scratch training does not import it.

## Validation

The original implementation has run on RTX 3060 12GB with Python 3.12, PyTorch 2.11.0+cu128 and tiktoken 0.14.0, including single-rank DDP. A packaging smoke check on that GPU also passed: bundled data loading, full-size GPT forward/backward and fused AdamW update with B=1, T=32 (eager mode). This does not rerun the complete compiled/DDP training loop or validate a fresh dependency installation. Multi-GPU scaling and A10 runtime are not yet verified.

For direct-vs-DDP validation, use the same script, machine, environment, data and settings. Save both logs and compare step/loss fields, excluding timing. Confirm both commands exit successfully and contain the expected number of steps: two empty logs are not a passing comparison. Exact equality is a local regression target, not guaranteed across GPU types or PyTorch versions.

## Attribution

Based on Karpathy's teaching material and GPT-2 implementation patterns. The MIT license notice from [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/LICENSE) is retained in `LICENSE`. Tiny Shakespeare originates from the linked char-rnn dataset; this repository does not claim authorship of the dataset.
