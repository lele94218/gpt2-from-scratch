# GPU acceptance manual: one host, two RTX 4090 GPUs

Run this manual on the `feat/fineweb-checkpoint-resume` PR branch. It validates the full default GPT-2 124M model, FineWeb input, the single-rank DDP entry point, and two-rank checkpoint recovery. Commands are for **Bash**, from the repository root, in the same shell. These are acceptance instructions, not a claim that the two-GPU run has already passed.

## 1. Prepare the machine and keep evidence

Use one instance with two visible NVIDIA GPUs. Allow 50–100 GB disk for the environment, compilation caches, small dataset, and several checkpoints. Keep at least 8 GB host RAM free for the final comparison, which loads two approximately 1.5 GB checkpoints plus Python/PyTorch overhead.

```bash
git clone --branch feat/fineweb-checkpoint-resume https://github.com/lele94218/gpt2-from-scratch.git
cd gpt2-from-scratch
# If already cloned, check out the PR branch instead of cloning again.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# Choose one profile compatible with the host driver; retain it for all runs.
bash setup.sh cu128
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements-data.txt

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p checkpoints
RUN_ROOT=$(mktemp -d "$PWD/checkpoints/acceptance.XXXXXX")
export RUN_ROOT
printf 'Evidence directory: %s\n' "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/commit.txt"
nvidia-smi > "$RUN_ROOT/nvidia-smi.txt"
nvidia-smi topo -m > "$RUN_ROOT/topology.txt"
df -h . /dev/shm > "$RUN_ROOT/storage.txt"
uv pip freeze --python .venv/bin/python > "$RUN_ROOT/packages.txt"
python - <<'PY' | tee "$RUN_ROOT/environment.txt"
import torch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2, 'Expose exactly two GPUs for this manual'
for i in range(2):
    with torch.cuda.device(i):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
        assert torch.cuda.is_bf16_supported()
PY
```

If `cu128` is incompatible with the driver, use the README's `cu118` profile before starting any tests. Do not change the environment between reference and resumed runs. Setup checks CUDA/BF16 access; it does not install host drivers. Inspect the topology rather than assuming the GPUs have a fast peer link.

`pipefail` matters: a failed training process must not look successful just because `tee` saved its output. After reconnecting, activate the venv and restore `RUN_ROOT` to the printed path; do not recreate the reference run or overwrite its logs.

## 2. Run the inexpensive lifecycle tests first

```bash
CUDA_VISIBLE_DEVICES=0,1 TEST_DEVICE=cuda TEST_WORLDS=1,2 TEST_TORCHRUN=1 \
  python -m unittest discover -s tests -v 2>&1 | tee "$RUN_ROOT/regression.log"
```

Pass: all five tests report `OK`, including restarts at world sizes one and two. These use a tiny model. They also verify rank-local input slices across shard boundaries, corruption/config rejection, and RNG restoration. They are not a full-model memory or throughput test. First tokenizer use requires network access.

## 3. Prepare a small real FineWeb dataset

```bash
python prepare_fineweb.py --output-dir "$RUN_ROOT/data" \
  --shard-size 1000000 --max-tokens 3000000 \
  2>&1 | tee "$RUN_ROOT/prepare.log"
cp "$RUN_ROOT/data/manifest.json" "$RUN_ROOT/data-manifest.json"
```

Pass: preparation completes with a manifest, one validation shard and two training shards. Use this exact data for every run below. Three million tokens are intentional: shard transitions and wraparound occur during the test. Repeated training on this small dataset says nothing about final model quality. If preparation is interrupted, retry in a new empty directory and update the path below.

## 4. Define the full-model configuration

```bash
COMMON=(
  --device cuda --compile --no-generate
  --data-dir "$RUN_ROOT/data"
  --batch-size 4 --seq-len 1024 --total-batch-size 524288
  --max-steps 20 --warmup-steps 10 --checkpoint-every 5
  --seed 1337
)
```

No model-size overrides: logs must report **124475904 parameters**. Gradient accumulation must be 128 on one GPU and 64 on two. `--no-generate` keeps this focused on training/recovery; generation is covered separately by the regression suite.

All runs retain `--max-steps 20`. `--stop-after 10` is an absolute completed-step count, not a new LR horizon. Output directories below must be fresh, except for the intentional resume.

## 5. Compare Python and single-rank torchrun

```bash
CUDA_VISIBLE_DEVICES=0 python train-gpt2.py "${COMMON[@]}" --stop-after 5 \
  --output-dir "$RUN_ROOT/python1" 2>&1 | tee "$RUN_ROOT/python1.log"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train-gpt2.py "${COMMON[@]}" --stop-after 5 \
  --output-dir "$RUN_ROOT/ddp1" 2>&1 | tee "$RUN_ROOT/ddp1.log"
```

Pass: both complete steps 0–4, save `next_step=5`, and pass the exact loss/LR and checkpoint comparison in section 7. Use the same physical GPU. Do not compare entire log files: timings, throughput, paths and launcher messages naturally differ.

## 6. Compare continuous and resumed two-GPU training

```bash
# Reference: complete steps 0–19 without restarting.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train-gpt2.py "${COMMON[@]}" --output-dir "$RUN_ROOT/ddp2-full" \
  2>&1 | tee "$RUN_ROOT/ddp2-full.log"

# Pause after completing steps 0–9.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train-gpt2.py "${COMMON[@]}" --stop-after 10 \
  --output-dir "$RUN_ROOT/ddp2-resume" \
  2>&1 | tee "$RUN_ROOT/ddp2-part.log"

# Inspect the saved boundary before the resumed run replaces latest.pt.
python - <<'PY' | tee "$RUN_ROOT/pause-state.txt"
import os
from pathlib import Path
import torch
p = torch.load(Path(os.environ['RUN_ROOT'])/'ddp2-resume/latest.pt',
               map_location='cpu', weights_only=False)
assert p['next_step'] == 10 and len(p['ranks']) == 2
assert p['ranks'][0]['loader'] == p['ranks'][1]['loader']
print('next_step:', p['next_step'], 'ranks:', len(p['ranks']))
for rank, state in enumerate(p['ranks']):
    print('rank:', rank, 'loader:', state['loader'])
PY

# Continue with the same two GPUs and unchanged training settings.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train-gpt2.py "${COMMON[@]}" --output-dir "$RUN_ROOT/ddp2-resume" \
  --resume "$RUN_ROOT/ddp2-resume/latest.pt" \
  2>&1 | tee "$RUN_ROOT/ddp2-resumed.log"
```

Pass: the resumed log starts at step 10, completes step 19 and saves `next_step=20`. Both GPUs do work and the jobs exit without a collective hang. The identical loader states are expected: they store a shared global cursor; each rank adds its own input offset. They do not mean both ranks read the same input slice.

This tests a clean process restart at a saved optimizer boundary. It does not test sudden termination during a write or recovery of unsaved steps. Do not resume a one-GPU checkpoint on two GPUs: this implementation rejects world-size changes.

## 7. Verify loss sequences and complete checkpoints

Run this from the repository root. It reuses the existing test comparator; no training code changes are needed. Load only checkpoints produced by your own runs.

```bash
python - <<'PY' | tee "$RUN_ROOT/comparison.log"
import os
from pathlib import Path
import re
import sys
import unittest
import torch
sys.path.insert(0, str(Path('tests').resolve()))
from test_training import assert_nested_equal

root = Path(os.environ['RUN_ROOT'])
test = unittest.TestCase()
pattern = r'^step (\d+), loss: ([^,\n]+), lr: ([^ ]+)'
def rows(*names):
    return [row for name in names for row in
            re.findall(pattern, (root/name).read_text(), re.MULTILINE)]
def compare(label, left_logs, right_logs, left_dir, right_dir, steps, world):
    a, b = rows(*left_logs), rows(*right_logs)
    test.assertEqual([int(row[0]) for row in a], list(range(steps)))
    test.assertEqual(a, b, f'{label}: loss/LR differs')
    left = torch.load(root/left_dir/'latest.pt', map_location='cpu', weights_only=False)
    right = torch.load(root/right_dir/'latest.pt', map_location='cpu', weights_only=False)
    test.assertEqual(left['next_step'], steps)
    test.assertEqual(len(left['ranks']), world)
    assert_nested_equal(test, left, right)
    print(label, 'PASS: exact loss/LR, model, optimizer, config, step, loader and RNG')

compare('Python vs DDP1', ['python1.log'], ['ddp1.log'],
        'python1', 'ddp1', 5, 1)
compare('DDP2 continuous vs restart', ['ddp2-full.log'],
        ['ddp2-part.log', 'ddp2-resumed.log'], 'ddp2-full', 'ddp2-resume', 20, 2)
PY
```

Pass: two `PASS` lines and exit code zero. A rounded loss match alone is insufficient. Do not compare checkpoint file hashes: serialization bytes are not the semantic state comparison.

Exact equality is the acceptance target for these matched runs, not a guarantee across hardware, PyTorch versions, or world sizes. If it fails, preserve the logs and checkpoints, then repeat the affected pair with `--no-compile` in new output directories. Investigate the first divergence; do not silently loosen tolerances or mark the original test passed. Fresh independent runs can also expose nondeterministic kernels, so a mismatch needs diagnosis before attributing it to resume logic.

**Do not demand bitwise equality between one-GPU and two-GPU training.** Gradient reduction order changes, and shard-tail handling can change the consumed stream. Compare each run only to its matching reference above.

## 8. Record performance and close out

During a run, use a second SSH session to sample utilization/memory:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv -l 1
```

Save that output separately if desired; stop the monitor with Ctrl-C. Sampling estimates peak memory, not an exact allocator peak. For a comparable throughput measurement, run a fresh one-GPU job with `COMMON` for all 20 steps (omit `--stop-after`); compare median tokens/sec over steps 5–19 with `ddp2-full`. Exclude compilation/recompilation steps. The five-step entry-point check alone is too short for a reliable speedup claim. Printed step time excludes checkpoint writes; record total wall time separately for cost estimates. No fixed speedup threshold is required for correctness, and a 4090 pair does not predict eight A100 SXM performance.

Record actual observations before checking any boxes:

| Check | Result / evidence |
|---|---|
| Commit, GPU topology, driver, torch/CUDA versions | |
| Tiny-model regression: single/two-rank | |
| Full 124M: Python vs single-rank DDP exact comparison | |
| Full 124M: two-rank continuous vs resume exact comparison | |
| Shard transitions / final rank cursors | |
| Stable single/two-GPU tokens/sec and sampled peak memory | |
| Checkpoint size, total job wall time, rental cost | |

Back up logs, manifest, environment records and any checkpoint you want to keep **off the rented instance** before destroying it. This manual establishes training lifecycle correctness; it does not establish validation quality, a full FineWeb run, or eight-GPU scaling.
