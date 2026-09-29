# From a base checkpoint to a small chat model

This experiment fine-tunes our GPT-2 124M checkpoint on short English conversations.
It keeps the original Transformer, tokenizer, and vocabulary. The first version is
single-process, full-parameter SFT on one GPU. It is intended to make the training
mechanics easy to inspect; a 124M model is not expected to become a reliable general
assistant from a short fine-tuning run.

## Read the code in this order

1. [`chat_data.py`](../chat_data.py): `render`, `encode_conversation`, then `collate`.
2. [`prepare_chat.py`](../prepare_chat.py): filter complete conversations and publish a manifest.
3. [`train_sft.py`](../train_sft.py): `loss_sum`, `backward_update`, then the main loop.
4. [`checkpoint.py`](../checkpoint.py): the existing atomic checkpoint and resume helpers.
5. [`chat.py`](../chat.py): reuse the same role template at inference time.

[`model_io.py`](../model_io.py) imports the GPT class from `train-gpt2.py` without
moving or editing that learning script. No Hugging Face model conversion is involved.

## What changes compared with pretraining?

| Part | Pretraining | This SFT experiment |
| --- | --- | --- |
| Starting point | Random initialization | Existing base model weights |
| Example | A window of continuous text | One complete conversation |
| Input | All text tokens | System, user, and assistant text |
| Targets | Every next token | Assistant content and its end token only |
| Optimizer | Pretraining AdamW state | Fresh AdamW, smaller learning rate, no weight decay |
| Batch objective | Mean over fixed-size text targets | Mean over actual assistant targets |
| End of epoch | Token stream can wrap | Keep the final partial batch/update |

All model parameters still receive gradients. Masking the prompt's *loss* does not
mask the prompt from *attention*: the answer is learned in the context of the question.

## Follow a single example

The serialized conversation looks like this (EOT denotes the existing GPT-2 token 50256):

```text
User: Hi
Assistant: Hello!<EOT>
```

For illustration, suppose its tokens were:

```text
text:        User:    Hi    Assistant:    Hello    !    <EOT>
ids:           10    20        30          40    50      60
answer mask:    0     0         0           1     1       1

input_ids:     10    20        30          40    50
labels:      -100  -100        40          50    60
```

These numbers are explanatory, not real tokenizer IDs. Actual headers and words
can occupy multiple tokens. The implementation tokenizes each header, body, and
separator separately; inference uses exactly the same procedure.

`ids[:-1]` is the input. `ids[1:]` is the next-token target. Shift the answer mask
by one position too, then replace non-answer targets with `-100`. The existing
GPT forward does not do another shift. In this trainer we request logits and use
`cross_entropy(ignore_index=-100, reduction='sum')` explicitly.

Assistant headers are supplied by the application and have no loss. Assistant
content **and EOT** have loss. User/system text and right-padding have no loss.
For multiple turns, every assistant answer is supervised. One conversation stays
in one example, so unrelated conversations cannot attend to each other.

Padding uses EOT as an input value, but its labels are `-100`; real answer-ending
EOT targets remain supervised. Because padding is on the right and attention is
causal, real tokens cannot attend to padding. No new attention mask is required.

## Why gradient accumulation needs a different average

Suppose one micro-batch has 10 answer tokens and another has 100. Averaging their
two mean losses gives the short micro-batch as much weight as the long one.

Instead, for every update we count all valid assistant targets first:

$$
L = \frac{\sum_m \sum_{t \in A_m} -\log p(y_t\mid x_{\le t})}{\sum_m |A_m|}
$$

Each micro-batch backpropagates its summed loss divided by that same denominator.
We then clip the accumulated gradient and call `optimizer.step()` once. This also
handles an epoch's final partial update. The test suite compares these gradients
with those from a single combined batch with unequal answer lengths.

`input_slots` in the log counts padded input positions processed. `answer_tokens`
counts supervised targets. These are different quantities. Masking a target does
not eliminate the computation of its input representation.

## Data preparation

Use your existing environment. On the Bazzite machine this is `~/train-venv`;
there is no need to reinstall its working CUDA PyTorch wheel.

```bash
source ~/train-venv/bin/activate
python -m pip install -r requirements-data.txt

python prepare_chat.py \
  --train-examples 10000 --val-examples 500 --seq-len 512 \
  --output-dir data/chat-10k-512
```

The source is [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, marked Apache-2.0. Preserve this
attribution and source licensing information if redistributing derived data.
Training comes from the official `train` split; our validation comes from its
official `test` split. We use this as a development validation set, not a pristine
final benchmark after repeated tuning.

Preparation uses a seeded streaming shuffle with a 10,000-row buffer, not a uniform
shuffle of the whole corpus. It keeps complete conversations whose shifted input
fits the requested length; it never truncates an answer. It rejects empty/malformed
messages, invalid role order, and exact duplicate conversations both within and
across the selected splits. This is not semantic deduplication or a guarantee of
answer quality. Review example conversations when choosing your training mix.

The manifest records the source revision, template version, selection statistics,
source mixture, and SHA-256 hashes. Training verifies both data files. Data and
checkpoints are ignored by Git. Preparation uses CPU, not GPU, and may download
substantial source data even for a small selection. It needs an empty output
directory; an interrupted preparation can reuse the Hugging Face cache but must
write to a fresh output directory. Do not modify data during a run.

## Stage 1: a short lifecycle smoke test

Set `BASE_CKPT` to a **trusted** local copy of your final pretraining checkpoint.
This loader uses Python pickle-compatible checkpoint loading. The checkpoint must
contain this repository's `format_version=1`, model config, and model state dict.

```bash
BASE_CKPT=/absolute/path/to/base/latest.pt

python train_sft.py \
  --init-from "$BASE_CKPT" \
  --data-dir data/chat-10k-512 \
  --output-dir checkpoints/sft-smoke \
  --device cuda --batch-size 4 --seq-len 512 --accum-steps 8 \
  --epochs 1 --lr 3e-5 --stop-after 3 --eval-every 0
```

Expected: three JSON update lines with finite loss and gradient norm, then a
checkpoint with `next_step=3`. `--stop-after` keeps the full training horizon;
it is an interruption test, not a three-step learning-rate schedule.

Defaults are intentionally uncompiled for debugging. Add `--compile` after the
smoke test if desired; compilation must remain the same when resuming. This first
SFT implementation rejects `torchrun`, including one rank. Pretraining still
supports DDP as before.

## Stage 2: first learning run on RTX 3060 12GB

The initial configuration is BF16, batch size 4, sequence length 512, accumulation
8, one epoch, peak learning rate 3e-5, and no weight decay. With 10,000 conversations
there are 313 optimizer updates: 312 updates of 32 conversations and one of 16.
The learning rate warms up for 3% of updates and then decays to 10% of its peak.
These are starting settings, not a claim of optimal hyperparameters.

```bash
python train_sft.py \
  --init-from "$BASE_CKPT" \
  --data-dir data/chat-10k-512 \
  --output-dir checkpoints/sft-10k \
  --device cuda --batch-size 4 --seq-len 512 --accum-steps 8 \
  --epochs 1 --lr 3e-5 --checkpoint-every 100 --eval-every 100
```

If memory is tight, reduce batch size to 2 and raise accumulation to 16 **before
starting a new run**. Avoid playing a GPU-heavy game concurrently. Time a few
steady-state updates and extrapolate; include validation and saving time. Do not
assume the old pretraining throughput predicts SFT runtime exactly.

To resume that same SFT run, repeat its configuration and replace `--init-from`:

```bash
python train_sft.py \
  --resume checkpoints/sft-10k/latest.pt \
  --data-dir data/chat-10k-512 \
  --output-dir checkpoints/sft-10k \
  --device cuda --batch-size 4 --seq-len 512 --accum-steps 8 \
  --epochs 1 --lr 3e-5 --checkpoint-every 100 --eval-every 100
```

`--init-from` copies **weights only** and starts optimizer/step/cursor from scratch.
The new checkpoint records the SHA-256 of the initialization checkpoint for provenance.
`--resume` restores SFT weights, optimizer, shuffled epoch/cursor, and RNG; it rejects
changed data, learning-rate horizon, batch settings, precision, or PyTorch version.
Checkpoints are saved atomically at update boundaries. Evaluation preserves RNG
and training mode. Interval changes do not alter the training trajectory.

The original base checkpoint must stay separate. An existing output checkpoint
cannot be replaced by a new `--init-from` run. Loading an SFT checkpoint through
`--init-from` deliberately starts another new fine-tuning stage.

## Chat and evaluate behavior

```bash
python chat.py --checkpoint checkpoints/sft-10k/latest.pt --device cuda

# Controlled before/after comparison: identical role format and greedy decoding.
python chat.py --checkpoint "$BASE_CKPT" --allow-base --device cuda \
  --prompt 'Explain what a cat is in one sentence.' --temperature 0
python chat.py --checkpoint checkpoints/sft-10k/latest.pt --device cuda \
  --prompt 'Explain what a cat is in one sentence.' --temperature 0
```

Interactive commands: `/reset` clears history and `/quit` exits. Only fully ended
nonempty answers are retained in history. Generation stops on EOT or a token/context
limit. A limit is reported rather than silently pretending the model learned to
stop. If history fills the context window, reset or shorten it. There is no KV cache
yet: generation recomputes the prefix to keep this version simple. Invalid padded
vocabulary IDs above 50256 are excluded from sampling.

Use fixed held-out prompts covering greetings, short explanations, rewriting,
summarization, and follow-up questions. Inspect relevance, repetition, role switching,
and stopping behavior as well as validation loss. Assistant-only SFT loss is not
directly comparable to the old full-text FineWeb loss. The model may still invent
facts and fail simple tasks. A lower loss alone does not establish chat quality.

## Acceptance checks

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The automated tests cover shifted assistant/EOT labels, prompt/template agreement,
padding, long-example filtering, exact deduplication, checksum rejection, unequal
answer-token gradient weighting, partial epochs, evaluation RNG isolation, exact
CPU uninterrupted/resumed checkpoint equality, unsafe overwrite/config rejection,
EOT stopping and padded-vocabulary exclusion, and a real tiny-GPT chat smoke test. Tests use local fixtures; tokenizer assets
may be fetched on first use. Dataset downloading and 124M GPU performance are
separate manual checks; passing tiny CPU tests does not establish chat quality.

Before trusting a longer run, also overfit a small set of examples and verify that
the model can produce their answers and EOT. For example, prepare 32 train/8 val
examples in a separate directory and try multiple epochs; do not expect its held-out
quality to improve just because it memorized those 32 training answers.

## What is borrowed from nanochat?

Karpathy's [SFT trainer](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/chat_sft.py)
and [conversation renderer](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py)
are the reference for assistant-only supervision and answer-end targets. Its
SmolTalk task uses the same smol-smoltalk dataset.

We do not copy its architecture, tokenizer, special role tokens, packed batches,
distributed trainer, math-task mixture, or optimizer warm-start. Our first experiment
uses ordinary GPT-2 role text, existing EOT, independent padded conversations,
single-GPU AdamW initialized afresh, and a short-conversation subset. RL is not
needed to implement this supervised learning stage.
