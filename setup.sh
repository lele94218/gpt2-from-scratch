#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
profile="${1:-cu118}"
case "$profile" in
  cu118) torch_version=2.6.0 ;;
  cu128) torch_version=2.11.0 ;;
  *) echo "Usage: bash setup.sh [cu118|cu128]" >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Linux ]]; then
  echo "This setup targets Linux with an NVIDIA GPU." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
# Never replace an existing environment automatically.
if [[ ! -e .venv ]]; then
  uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12 in .venv"'
uv pip install --python .venv/bin/python "torch==$torch_version" --index-url "https://download.pytorch.org/whl/$profile"
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python - <<'PY'
import torch
print('PyTorch:', torch.__version__, '| CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check the host driver and GPU/container access'
print('GPU:', torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), 'This training script uses BF16; a BF16-capable GPU is required'
a = torch.randn(64, 64, device='cuda')
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    b = a @ a
assert torch.isfinite(b).all()
torch.cuda.synchronize()
print('CUDA/BF16 check passed. Run: source .venv/bin/activate && python train-gpt2.py')
PY
