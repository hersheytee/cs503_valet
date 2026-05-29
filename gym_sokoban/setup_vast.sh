#!/usr/bin/env bash
# Prepare a Vast.ai PyTorch container for gym_sokoban training.
#
# Usage:
#   bash gym_sokoban/setup_vast.sh
#   bash gym_sokoban/setup_vast.sh --pull
#
# Notes:
# - Assumes you launched a PyTorch/CUDA image where torch is already installed.
# - Installs OpenCV headless instead of GUI OpenCV to avoid libxcb/libGL issues.
# - If WANDB_API_KEY is set, logs in non-interactively.

set -euo pipefail

DO_PULL=0
if [[ "${1:-}" == "--pull" ]]; then
    DO_PULL=1
elif [[ "${1:-}" != "" ]]; then
    echo "Unknown argument: $1"
    echo "Usage: bash gym_sokoban/setup_vast.sh [--pull]"
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== Repo =="
echo "  $REPO_ROOT"

if [[ "$DO_PULL" == "1" ]]; then
    echo "== Updating repo =="
    git pull
fi

echo "== Python =="
python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

echo "== CUDA / PyTorch =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
PY

echo "== Installing Python dependencies =="
REQ_TMP="$(mktemp)"
grep -v -E '^(torch|opencv-python|opencv-contrib-python)([=<>!~ ].*)?$' requirements.txt > "$REQ_TMP"
python -m pip install --upgrade pip
python -m pip install -r "$REQ_TMP"
python -m pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
python -m pip install opencv-python-headless PyYAML wandb
rm -f "$REQ_TMP"

echo "== Import check =="
python - <<'PY'
import cv2
import gym
import gymnasium
import numpy
import torch
import wandb
import yaml
print("imports: ok")
PY

echo "== Directories =="
mkdir -p runs launcher_logs logs figures checkpoints

echo "== W&B =="
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set."
    echo "Run 'wandb login' manually before online tracked runs."
fi

echo "== Done =="
echo "Recommended smoke test:"
cat <<'EOF'
CUDA_VISIBLE_DEVICES=0 python -u gym_sokoban/train.py \
  --env-id Sokoban-small-v0 \
  --seed 1 \
  --exp-name smoke_random_oracle \
  --total-timesteps 16384 \
  --save-model \
  --oracle-cost 0.0 \
  --oracle-accuracy 0.5 \
  --n-envs 64 \
  --n-steps 256 \
  --n-minibatches 8 \
  --no-track
EOF
