# Recorded training results

These records describe experiments performed with this repository. They establish
training lifecycle correctness and measured results on specific samples, not
general chat reliability or equivalence to another published GPT-2 model.

## Pretraining

The model has 12 layers, 12 heads, width 768, a 1024-token context, and a padded
50,304-token vocabulary. Tokenization uses GPT-2's 50,257-token encoding. Sampling
excludes the extra padded vocabulary entries.

Runs used one NVIDIA L20 with CUDA BF16 and torch.compile. The measured runtime
used Python 3.11.10 and a PyTorch 2.7 development build with CUDA runtime 12.8;
these measurements do not certify every wheel/profile in `setup.sh`.

| Phase | Processed tokens | Updates | Tokens/update | Peak LR | Warmup | Wall time |
|---|---:|---:|---:|---:|---:|---|
| Initial | 4,000,055,296 | 30,518 | 131,072 | 3e-4 | 500 | 11h 57m 55s |
| Continuation | 6,000,082,944 additional | 45,777 | 131,072 | 1e-4 | 500 | 17h 55m 37s |

Both used micro-batch 8, sequence length 1024 and gradient accumulation 16.
The continuation loaded weights from the first phase and reset AdamW, the data
cursor and LR schedule. Its input was training shards 41–100, following shards
1–40 in the first phase. This differs from a single uninterrupted 10B schedule.

Source: [Karpathy's GPT-2 FineWeb-Edu token shards](https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards),
revision `a33f75d78c7f74236fb03754ec1eb3cc77507d64`.
Attribution: HuggingFaceFW/FineWeb-Edu and Karpathy; source corpus license ODC-By.

| Checkpoint | Same 2,097,152-token validation loss | Perplexity |
|---|---:|---:|
| 4B best/final | 3.200817 | 24.552577 |
| Continuation best, phase step 45,500 | 3.107913 | 22.374310 |
| Continuation final, phase step 45,777 | 3.108000 | 22.376252 |

The best continuation checkpoint has processed 9,963,831,296 cumulative tokens.
The full two-phase run completed 10,000,138,240. Best and final are close but are
not identical files. Model parameters in the selected checkpoint were checked
for finiteness, and its optimizer step counters matched its saved step.

The 4B run's raw source SHA-256 is
`adc8a6bf84bc6b0fa1985a3fd403932d462dedf3b41d4f080adcad0863430be9`.
The selected continuation checkpoint's raw source SHA-256 is
`ec6e068c5adf4a5ec9e3b1d35a98064d761a330adc5392db34fc8e03f0d34fd8`.
Exported release files have different hashes; use the release's `SHA256SUMS` file
to verify downloads.

## SFT and diagnostics

SFT uses fresh AdamW state, assistant-only next-token supervision including EOT,
the same GPT-2 tokenizer, and the repository's plain-text role template. Masked
prompt/padding targets remain visible as causal context. SFT loss is not directly
comparable to full-text FineWeb loss.

The first SFT run used 10,000 training conversations, 500 development conversations,
context 512, micro-batch 8, accumulation 4, one epoch and peak LR 3e-5. It completed
313 updates. On the same development conversations, assistant-only loss fell
from 2.6985 to 1.5801. Stopping behavior improved, but many answers remained wrong.

A 32-example memorization diagnostic reached 32/32 on its own training prompts.
That demonstrated small-sample fitting, not generalization. A four-run short-task
SFT grid then compared LR 1e-5/3e-5 and one/two epochs on 4,871 examples. The
development-selected candidate was LR 3e-5, one epoch; it was not universally
better at open-ended conversation.

The expanded experiment used 20,000 examples:

- 8,000 filtered natural-language first turns, including QA, rewriting,
  summarization and format constraints.
- 8,000 arithmetic examples, half answer-only and half short worked answers.
- 4,000 comparison, sorting, copying, case-conversion and extraction examples.

The 12,000 procedural examples represent 3,000 task instances with four variants
each, rather than 12,000 distinct knowledge facts. Arithmetic contributed 20.95%
of supervised tokens, including scaffolding text and EOT. The run used context
256, micro-batch 8, accumulation 4, LR 3e-5, and one epoch: 625 updates.

Natural data source: [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0. Filtering included
length, role structure, normalized-prompt deduplication and obvious repetition.
It was not exhaustive human factual review or full semantic deduplication.

The expanded experiment froze 256 procedural development tasks and 256 final
test tasks, 32 per category. Variants of a procedural task stayed in one split;
known historical procedural tasks were excluded from the new evaluation. Another
200 natural examples were held out. Final tests were run only after checkpoint
selection. Models shared greedy decoding and the same scoring rules.

| Final-test category | Earlier short-task SFT | Expanded 20k SFT |
|---|---:|---:|
| Addition | 0/32 | 0/32 |
| Subtraction | 1/32 | 1/32 |
| Comparison | 9/32 | 25/32 |
| Sorting | 0/32 | 3/32 |
| Copying | 30/32 | 30/32 |
| Extraction | 13/32 | 32/32 |
| Lowercase | 12/32 | 26/32 |
| Uppercase | 2/32 | 22/32 |

Strict success requires the requested concise answer format as well as content.
A zero score does not establish absence of all relevant latent knowledge.
The natural-test numbers are reference-answer likelihoods, not correctness rates.
The latest model's worked-answer arithmetic mode produced only 5/64 correct final
answers. A later diagnostic on 64 arithmetic prompts from its own training data
found only 8/64 correct despite correct integer formatting and stopping.

## Limits and artifact use

Open-ended samples still show repetition, incorrect facts, missed constraints and
irrelevant answers. The experiments do not establish reliable mathematical
reasoning, general assistant quality, or parity with Karpathy's model. No matched
HellaSwag comparison was completed. Data mixture, wording, amount and optimization
all changed between experiments, so their effects are not individually isolated.

Full training checkpoints contain optimizer, RNG and loader state and are kept
outside Git. `export_weights.py` creates smaller `.pt` artifacts for generation
and initializing a new training phase. They preserve source-checkpoint provenance
but intentionally omit resumable training state. Only artifacts actually listed
on the [Releases page](https://github.com/lele94218/gpt2-from-scratch/releases)
should be treated as published downloads.
