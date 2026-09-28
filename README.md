# GPT-2 from scratch: training lab

A hands-on GPT-2 implementation following Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) and [build-nanogpt](https://github.com/karpathy/build-nanogpt).

Train on bundled Tiny Shakespeare or tokenized FineWeb(-Edu) shards, save checkpoints, and resume at the next optimizer step. The Transformer definition remains the original learning implementation. See [the implementation walkthrough](docs/fineweb-resume.md) for the new runtime changes.

## Install on a Linux NVIDIA GPU machine

Requirements: Git, internet access, a BF16-capable NVIDIA GPU and a compatible driver. The setup uses [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12; it never changes host drivers.

```bash
git clone https://github.com/lele94218/gpt2-from-scratch.git
cd gpt2-from-scratch
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash setup.sh cu118
source .venv/bin/activate
```

`cu118` installs PyTorch 2.6.0/CUDA 11.8 as a compatibility choice for older drivers, such as the A10 host's 535 branch. `bash setup.sh cu128` selects the original training machine's PyTorch 2.11.0/CUDA 12.8 profile, requiring a newer compatible driver. Follow the [official wheel matrix](https://pytorch.org/get-started/previous-versions/); the CUDA field in `nvidia-smi` is not `torch.version.cuda`. The exact A10 environment has not been tested.

Attention uses PyTorch SDPA; no separate flash-attn installation is required. First use downloads tokenizer assets and compiles kernels. Do not count compilation time as steady-state throughput.

## Quick run: bundled Shakespeare

```bash
python train-gpt2.py --max-steps 5 --output-dir checkpoints/shakespeare
```

This writes `checkpoints/shakespeare/latest.pt` after the final optimizer update, before generating text. By default, a longer run also saves every 100 updates. An existing checkpoint is never silently replaced by a fresh run: use `--resume` or another output directory.

```bash
# DDP path on one GPU
torchrun --standalone --nproc_per_node=1 train-gpt2.py \
  --max-steps 5 --output-dir checkpoints/ddp1

# Same-machine two-GPU training
torchrun --standalone --nproc_per_node=2 train-gpt2.py \
  --max-steps 5 --output-dir checkpoints/ddp2
```

## Prepare FineWeb-Edu

```bash
uv pip install --python .venv/bin/python -r requirements-data.txt

# Small pilot: 3 million GPT-2 tokens, including a 1-million-token validation shard.
python prepare_fineweb.py --output-dir data/fineweb-pilot \
  --shard-size 1000000 --max-tokens 3000000

python train-gpt2.py --data-dir data/fineweb-pilot \
  --max-steps 20 --checkpoint-every 5 --output-dir checkpoints/fineweb-pilot
```

The default source is [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), configuration `sample-10BT`, streamed without downloading the entire source first. For ordinary FineWeb, select `--dataset HuggingFaceFW/fineweb`. Use `--revision <dataset-commit>` to pin the input snapshot.

For the full source configuration, omit `--max-tokens`:

```bash
python prepare_fineweb.py --output-dir data/fineweb-edu --shard-size 100000000
```

Preparation uses one tokenizer process for clarity. Memory is bounded by one shard plus the current document; large-scale preprocessing will take time. GPT-2 tokenization may produce a different token count from the source dataset's advertised count. uint16 storage uses roughly two bytes per token, plus headers; reserve sufficient disk space before processing billions of tokens.

The first shard is held out as validation data. **This PR does not implement validation-loss evaluation or HellaSwag.** All training reads only `train` entries in `manifest.json`. An interrupted preparation has no completed manifest; retry into an empty directory. Preparation itself is not resumable.

Dataset attribution: FineWeb/FineWeb-Edu are published by HuggingFaceFW under ODC-By; see their dataset cards for attribution and use conditions. No FineWeb data or model checkpoints are committed here.

## Save and resume correctly

Keep the LR horizon fixed across interruption. `--max-steps` is the total intended training horizon, while `--stop-after` is an absolute completed-step count for a planned pause:

```bash
# Plan 100 steps, pause after completing step indices 0..9.
python train-gpt2.py --data-dir data/fineweb-pilot \
  --max-steps 100 --stop-after 10 --checkpoint-every 5 \
  --output-dir checkpoints/experiment

# Continue from step index 10 with the SAME training settings.
python train-gpt2.py --data-dir data/fineweb-pilot \
  --max-steps 100 --checkpoint-every 5 \
  --output-dir checkpoints/experiment --resume checkpoints/experiment/latest.pt
```

For DDP, launch resume with the same `torchrun --nproc_per_node=N` and settings. All ranks participate in checkpoint coordination; rank zero atomically replaces `latest.pt`. Every rank must be able to read the same checkpoint and data. Multi-node orchestration is outside the tested scope.

A checkpoint contains model weights, AdamW state, the next step, per-rank data cursors and Python/NumPy/PyTorch CPU/CUDA RNG states. The loader verifies shard content hashes on startup (a full sequential disk scan) and validates the dataset fingerprint on restore. A relocated copy with identical filenames/content is allowed.

Resume rejects changes to world size, batch shape, model shape, LR horizon, seed, optimizer settings, precision, compile mode or PyTorch version. Dataset content must match. This is continuation, not elastic rescaling or fine-tuning. Exact floating-point equality across different hardware/software is not promised.

Checkpoints are saved only at completed optimizer steps. Abrupt termination loses work after the most recent successful save; restart with `--resume`. Only `latest.pt` is retained; copy it elsewhere if you need history. Each GPT-2/AdamW checkpoint can occupy roughly 1.5 GB, and atomic replacement temporarily needs space for both the old and new file. Checkpoint time is excluded from the printed training-step throughput.

Load only your own trusted checkpoints: `torch.load(..., weights_only=False)` is used to restore Python/NumPy RNG objects.

## Configuration

Run `python train-gpt2.py --help` for all flags.

| Setting | Default |
|---|---:|
| Layers / heads / embedding width | 12 / 12 / 768 |
| Vocabulary / parameters | 50,304 / 124,475,904 |
| `--batch-size` / `--seq-len` | 4 / 1,024 |
| `--total-batch-size` (global tokens/update) | 524,288 |
| `--max-steps` / `--warmup-steps` | 30 / 10 |
| `--checkpoint-every` | 100, plus normal exit |
| `--compile` / `--generate` | enabled |
| CUDA precision | BF16 autocast |

Global batch must be divisible by batch-size × seq-len × world-size. The original single-GPU defaults accumulate 128 micro-batches. Small Shakespeare data repeats within an update: these defaults demonstrate training infrastructure, not an optimal small-data recipe.

`--no-compile` avoids compilation for debugging. `--no-generate` skips post-training sampling. Explicit `--device cpu` uses FP32 and supports lightweight Gloo tests; the documented performance target remains NVIDIA CUDA. Model dimensions can be reduced with `--n-layer`, `--n-head`, `--n-embd` for tests. MPS is not a supported training target in this runtime.

## Tests

For full GPT-2 124M acceptance on a rented two-GPU host, follow the [GPU test manual](docs/gpu-test-manual.md): single-rank entry-point comparison, real FineWeb shards, and two-rank uninterrupted versus resumed training with exact checkpoint checks.

Offline tests use synthetic documents, tiny model dimensions and no Hugging Face downloads beyond the GPT-2 tokenizer's first-use assets:

```bash
python -m unittest discover -s tests -v
```

They check documents spanning multiple shards, token caps, DDP rank boundaries, data corruption, RNG restoration, and uninterrupted versus interrupted/resumed training. CPU tests run both single process and two-rank Gloo and compare loss/LR sequences, final parameters, optimizer state and loader/RNG state exactly.

Optional CUDA/NCCL regression (one GPU):

```bash
TEST_DEVICE=cuda TEST_WORLDS=1 TEST_TORCHRUN=1 \
  python -m unittest discover -s tests -v
```

Set `TEST_COMPILE=1` to exercise the compiled training path. Tests use disposable directories and do not overwrite real checkpoints. Runtime validation results and limitations are recorded in the PR.

## Attribution

The bundled [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) comes from char-rnn. Based on Karpathy's teaching material and GPT-2 implementation patterns; the MIT notice from [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/LICENSE) is retained in `LICENSE`.
