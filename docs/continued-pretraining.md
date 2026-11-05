# Continue pretraining on additional data

Use `--init-from` when an earlier training phase is complete and the next phase
uses a new dataset, learning-rate horizon or batch configuration. It loads model
weights only, then starts a fresh AdamW optimizer, schedule, seeded RNG and data
cursor at phase step zero. The model dimensions must match the source checkpoint.
This command accepts this project's pretraining checkpoints, not SFT or arbitrary
Hugging Face checkpoint formats.

Use `--resume` to recover an interrupted phase with the same settings and data.
The two flags are mutually exclusive. Do not change `--max-steps` on resume.

## Prepare the full dataset and the additional-data selection

After a first phase on training shards 1–40 (4B tokens), prepare shards 1–100
as the full 10B dataset. For the next phase, select only shards 41–100 (6B tokens):

```bash
python prepare_fineweb_tokens.py --train-shards 100 \
  --output-dir data/fineweb-pretokenized-10B

python prepare_fineweb_tokens.py --train-start 41 --train-shards 60 \
  --output-dir data/fineweb-continuation-6B
```

Both selections retain validation shard 0. They share the default BIN download
cache; NPY files live in their respective output directories. Each selection pins
the same dataset revision and verifies headers, token IDs and SHA-256 checksums.
Use a fresh output directory when changing a selection. Do not modify a dataset
directory while a training process is reading it.

If files were downloaded on another machine, use `--source-manifest sources.json`
with `--cache-dir` pointing to the transferred BIN files. This mode performs no
network calls and verifies every requested file before conversion. The source
manifest may contain more files than the requested selection and has this shape:

```json
{
  "dataset": "karpathy/fineweb-edu-100B-gpt2-token-shards",
  "revision": "a33f75d78c7f74236fb03754ec1eb3cc77507d64",
  "files": [
    {
      "file": "edu_fineweb_val_000000.bin",
      "bytes": 200001024,
      "sha256": "<SHA-256 of the downloaded file>"
    }
  ]
}
```

Include one entry per required file, including the validation file. Record hashes
on the downloading machine and verify them after transfer; do not calculate the
expected hashes from unverified destination files.

## Start a new 6B-token phase

Example for the 124M model on one NVIDIA GPU:

```bash
OMP_NUM_THREADS=1 python train-gpt2.py \
  --init-from checkpoints/phase1/latest.pt \
  --data-dir data/fineweb-continuation-6B \
  --device cuda --compile --no-generate \
  --batch-size 8 --seq-len 1024 --total-batch-size 131072 \
  --max-steps 45777 --warmup-steps 500 --max-lr 1e-4 \
  --checkpoint-every 250 --eval-every 250 --eval-batches 20 \
  --output-dir checkpoints/phase2
```

This processes approximately 6B additional tokens. The learning rate is a
conservative continuation starting point, not a claim of optimal tuning or exact
reproduction of a single uninterrupted 10B run. Evaluate it on the same held-out
sample used for phase 1. The script saves `latest.pt`; retaining a validation-best
checkpoint is the responsibility of the run launcher or the operator.

To recover phase 2, run the same command with
`--resume checkpoints/phase2/latest.pt` in place of `--init-from ...`.
The original phase-1 checkpoint is not needed for this resume.

The new checkpoint's `config.initialization` records the source checkpoint's
SHA-256, source next-step index and previously processed token count. Chained
phases accumulate that count. It counts optimizer-consumed tokens, not unique
dataset tokens; current-phase steps and optimizer state remain phase-local.

Back up the original checkpoint before starting a new phase. Load only trusted
checkpoints: full training-state deserialization uses `weights_only=False`.
