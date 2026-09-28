# Reading the FineWeb + checkpoint diff

Read the change in this order. The attention, MLP, Block and GPT definitions are unchanged.

## 1. Offline preparation: prepare_fineweb.py

A source document becomes `[endoftext] + GPT-2 tokens`. The inner `while` loop handles even a document larger than multiple shards. Tokens are written as uint16 `.npy` arrays. The first shard is val; subsequent shards are train. The final manifest records filenames, counts, source metadata and SHA-256 hashes. Writing it last distinguishes a complete preparation from partial output.

Questions: why tokenize once before training? Why can uint16 store this vocabulary? What happens when one document spans three shards?

## 2. Data ownership: data.py

The loader memory-maps prepared shards rather than allocating a long tensor for the entire corpus. It copies only the current batch into int64, the dtype expected by embedding lookup.

All ranks share one logical cursor: `(shard, global_round_start)`. Rank r adds `B*T*r`; advancing one round consumes `B*T*world_size` input tokens. Every rank switches shards using the same global boundary test. The extra token supplies the shifted target.

We drop a shard's tail when it cannot supply a complete global round. Short train shards are skipped with a message. We never stitch across shard boundaries or consume the validation shard. The stream cycles in fixed order; there is no epoch shuffle in this version.

Questions: why must rank zero and rank one switch shards together? Why is the cursor alone insufficient if the underlying data changes?

## 3. Checkpoint ownership: checkpoint.py

A checkpoint is taken after optimizer.step(), so no partial accumulated gradients need saving. It stores `next_step`, not the just-completed step, and captures the loader's already-advanced position.

Every rank contributes its own data/RNG state. Rank zero writes model and optimizer state once because ordinary DDP replicates them. This assumption would need changing for FSDP/ZeRO or a sharded optimizer.

The file is written to `latest.pt.tmp`, flushed, then atomically renamed to `latest.pt` on the same filesystem. A checkpoint-write failure is broadcast so other ranks do not proceed as if saving succeeded. This is local/shared-filesystem atomic replacement, not an object-store transaction.

Questions: what diverges if AdamW momentum is omitted? Why is putting the entire save call inside `if rank == 0` wrong? How much work can a sudden GPU eviction lose?

## 4. Runtime: train-gpt2.py

The model remains separate from its execution wrappers: `raw_model` is the original GPT; `model` may be compiled and DDP-wrapped. Saving `raw_model.state_dict()` avoids wrapper prefixes.

Initialization and wrapping happen before loading training RNG. The checkpoint then restores weights, optimizer state, cursor and RNG. Learning rate is computed from the restored global step and unchanged max_steps; no separate scheduler object is needed.

`--stop-after` deliberately does not alter max_steps. This lets a test compare four continuous steps against two steps plus restart without accidentally changing the cosine schedule.

Post-training generation happens after checkpointing, uses raw_model, and masks padded vocabulary IDs before decoding. This is the only sampling correction; model/training math is preserved apart from making runtime settings explicit.

## 5. Evidence: tests/test_training.py

The key regression compares complete checkpoint trees, not just rounded printed loss. It also checks rank alignment across shard switches, RNG round trips, corrupt data, and rejection of incompatible continuation settings.

These tests do not claim a full FineWeb training run, two-GPU speedup, bitwise reproducibility on different hardware, or recovery after losing unsaved optimizer steps.

## Validation performed for this change

- CPU eager: uninterrupted/restarted training matched exactly in single-process and two-rank Gloo runs.
- RTX 3060, Python 3.12, PyTorch 2.11.0+cu128: single-rank NCCL restart matched exactly in both eager and torch.compile modes.
- Final compiled CUDA test run: all five regression tests passed, including the Shakespeare generation path on CPU.
- Real FineWeb-Edu `sample-10BT` streaming: prepared 4,096 tokens as four 1,024-token shards; trained and resumed a tiny CPU model on the three train shards.
- GPU tests use a one-layer, width-16 model to test lifecycle behavior quickly. Full 124M checkpoint memory/throughput, two physical GPUs, a full FineWeb pass, and the A10 host remain untested.
