# Sokoban Vast.ai Commands

This file is for launching the next Sokoban run set with one 2-GPU Vast.ai
instance plus the laptop RTX 3060. These commands assume we are doing one seed
first, then adding more seeds later if the curves are worth keeping.

## Vast.ai Template Recommendation

Use a PyTorch template with SSH access.

Recommended template shape:

```text
Template family: PyTorch
Launch mode: SSH, or Jupyter + SSH
CUDA: 12.x
Python: 3.10 or 3.11
Disk: 60 GB minimum, 100 GB nicer
GPU count: 2 per Vast instance
GPU type: 2x RTX 4000 / RTX A4000 class is preferred for the paid grid
```

Avoid inference-specific templates such as vLLM/SGLang for PPO training. We
only need PyTorch, CUDA, Python, git, and enough CPU/RAM for parallel envs.

For offers, prefer:

```text
Reliability: high
CUDA works: verified
Direct SSH: yes
Internet/disk speed: reasonable
Price: sort by $/hr after filtering for 2 GPUs
```

## Instance Setup

Run this after SSHing into the Vast.ai instance. Use the existing setup script;
it preserves the PyTorch/CUDA install from the Vast template and installs
headless OpenCV to avoid GUI library issues.

```bash
# Get the repo.
git clone https://github.com/<YOUR_USER_OR_ORG>/cs503_project.git
cd cs503_project

# Set this first if you want non-interactive W&B login.
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

# Install dependencies and check CUDA/imports.
bash gym_sokoban/setup_vast.sh

# If the repo already existed and you want to update first:
bash gym_sokoban/setup_vast.sh --pull
```

If `WANDB_API_KEY` was not set, login manually:

```bash
wandb login
```

Quick check after setup:

```bash
# Quick code sanity check.
python -m py_compile gym_sokoban/train.py gym_sokoban/env_wrapper.py gym_sokoban/model.py
```

Important: only run the final matrix after `gym_sokoban/model.py` is updated to
the MiniGrid-equivalent CNN. Old WorldCoder-CNN runs should not be mixed with
new MiniGrid-CNN runs in final plots.

## Smoke Test

Use this before spending real money on 500k runs.

```bash
CUDA_VISIBLE_DEVICES=0 python -u gym_sokoban/train.py \
  --env-id Sokoban-small-v0 \
  --seed 1 \
  --exp-name smoke_minigridcnn_baseline \
  --total-timesteps 4096 \
  --n-envs 8 \
  --n-steps 64 \
  --n-minibatches 4 \
  --no-oracle \
  --wandb-project cs503-sokoban \
  --wandb-group sokoban_minigridcnn_smoke
```

## Overnight 23-Job Run, Seed 1: Laptop + One 2-GPU Vast Instance

Use this if you want to launch the whole one-seed overnight matrix and walk
away. This assumes:

```text
laptop: 1 GPU
Vast: 2 GPUs
```

Job count:

```text
baseline                              = 1
free perfect oracle                   = 1
paid costs 0.1,0.3,0.5,0.8,1.0
  x accuracies 0.25,0.5,0.75,1.0      = 20
linear cost schedule 0.0 -> 1.0       = 1
total                                 = 23
```

The laptop queue intentionally runs only three reference jobs:

```text
baseline
linear cost schedule
free perfect oracle
```

in that order. The Vast box runs the whole paid cost-by-accuracy grid.

### Laptop Queue

Run this locally from the repo root in PowerShell. It uses your laptop GPU for
only the baseline, linear schedule, and free perfect-oracle reference.

First verify PyTorch sees CUDA:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda)"
```

If this prints `False` for CUDA, install a CUDA-enabled PyTorch wheel. Check the
official PyTorch selector if this command goes stale:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda)"
```

```powershell
.\venv\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path laptop_logs | Out-Null

$env:WANDB_PROJECT = "cs503-sokoban"
$GROUP = "sokoban_minigridcnn_seed1_overnight_500k"

function Run-SokoJob {
  param(
    [string]$ExpName,
    [string]$ExtraArgs
  )

  Write-Host "[$(Get-Date)] start $ExpName"
  $argList = @(
    "-u", "gym_sokoban\train.py",
    "--env-id", "Sokoban-small-v0",
    "--seed", "1",
    "--exp-name", $ExpName,
    "--total-timesteps", "500000",
    "--n-envs", "64",
    "--n-steps", "256",
    "--n-minibatches", "8",
    "--save-model",
    "--wandb-project", "cs503-sokoban",
    "--wandb-group", $GROUP
  ) + ($ExtraArgs -split " ")

  & python @argList *> "laptop_logs\$ExpName`_s1.log"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[$(Get-Date)] FAILED $ExpName"
    Get-Content "laptop_logs\$ExpName`_s1.log" -Tail 80
    throw "$ExpName failed with exit code $LASTEXITCODE"
  }
  Write-Host "[$(Get-Date)] done  $ExpName"
}

# 1. Baseline first.
Run-SokoJob "baseline_minigridcnn_500k" "--no-oracle"

# 2. Linear cost schedule second.
Run-SokoJob "oracle_acc100_cost_linear_0to1_500k" "--oracle-accuracy 1.0 --oracle-cost 0.0 --oracle-cost-final 1.0"

# 3. Free perfect-oracle reference third.
Run-SokoJob "oracle_acc100_cost0_500k" "--oracle-accuracy 1.0 --oracle-cost 0.0"
```

If the laptop struggles with CPU env stepping, change only the laptop queue to:

```text
--n-envs 32 --n-steps 256 --n-minibatches 8
```

Keep `64` if it is stable enough; exact comparability is nicer.

### Vast Queue

Run this on the 2-GPU RTX4000 Vast instance. It covers all paid-cost conditions:
costs `0.1, 0.3, 0.5, 0.8, 1.0` crossed with accuracies
`0.25, 0.5, 0.75, 1.0`.

```bash
mkdir -p vast_logs

COMMON_ARGS="--env-id Sokoban-small-v0 \
  --total-timesteps 500000 \
  --n-envs 64 \
  --n-steps 256 \
  --n-minibatches 8 \
  --save-model \
  --wandb-project cs503-sokoban \
  --wandb-group sokoban_minigridcnn_seed1_overnight_500k"

run_job() {
  local gpu="$1"
  local exp_name="$2"
  local extra_args="$3"
  local log_path="vast_logs/${exp_name}_s1.log"

  echo "[$(date)] gpu=${gpu} start ${exp_name}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u gym_sokoban/train.py \
    $COMMON_ARGS \
    --seed 1 \
    --exp-name "$exp_name" \
    $extra_args \
    > "$log_path" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[$(date)] gpu=${gpu} FAILED ${exp_name}"
    tail -80 "$log_path"
    exit "$status"
  fi
  echo "[$(date)] gpu=${gpu} done ${exp_name}"
}

# GPU 0: costs 0.1 and 0.3, plus first half of cost 0.5.
(
  run_job 0 "oracle_acc025_cost01_500k" "--oracle-accuracy 0.25 --oracle-cost 0.1"
  run_job 0 "oracle_acc050_cost01_500k" "--oracle-accuracy 0.5 --oracle-cost 0.1"
  run_job 0 "oracle_acc075_cost01_500k" "--oracle-accuracy 0.75 --oracle-cost 0.1"
  run_job 0 "oracle_acc100_cost01_500k" "--oracle-accuracy 1.0 --oracle-cost 0.1"
  run_job 0 "oracle_acc025_cost03_500k" "--oracle-accuracy 0.25 --oracle-cost 0.3"
  run_job 0 "oracle_acc050_cost03_500k" "--oracle-accuracy 0.5 --oracle-cost 0.3"
  run_job 0 "oracle_acc075_cost03_500k" "--oracle-accuracy 0.75 --oracle-cost 0.3"
  run_job 0 "oracle_acc100_cost03_500k" "--oracle-accuracy 1.0 --oracle-cost 0.3"
  run_job 0 "oracle_acc025_cost05_500k" "--oracle-accuracy 0.25 --oracle-cost 0.5"
  run_job 0 "oracle_acc050_cost05_500k" "--oracle-accuracy 0.5 --oracle-cost 0.5"
) > vast_logs/gpu0_queue.log 2>&1 &

# GPU 1: second half of cost 0.5, then costs 0.8 and 1.0.
(
  run_job 1 "oracle_acc075_cost05_500k" "--oracle-accuracy 0.75 --oracle-cost 0.5"
  run_job 1 "oracle_acc100_cost05_500k" "--oracle-accuracy 1.0 --oracle-cost 0.5"
  run_job 1 "oracle_acc025_cost08_500k" "--oracle-accuracy 0.25 --oracle-cost 0.8"
  run_job 1 "oracle_acc050_cost08_500k" "--oracle-accuracy 0.5 --oracle-cost 0.8"
  run_job 1 "oracle_acc075_cost08_500k" "--oracle-accuracy 0.75 --oracle-cost 0.8"
  run_job 1 "oracle_acc100_cost08_500k" "--oracle-accuracy 1.0 --oracle-cost 0.8"
  run_job 1 "oracle_acc025_cost1_500k" "--oracle-accuracy 0.25 --oracle-cost 1.0"
  run_job 1 "oracle_acc050_cost1_500k" "--oracle-accuracy 0.5 --oracle-cost 1.0"
  run_job 1 "oracle_acc075_cost1_500k" "--oracle-accuracy 0.75 --oracle-cost 1.0"
  run_job 1 "oracle_acc100_cost1_500k" "--oracle-accuracy 1.0 --oracle-cost 1.0"
) > vast_logs/gpu1_queue.log 2>&1 &

wait
echo "Vast paid-grid jobs complete"
```

Check progress:

```bash
tail -f vast_logs/gpu*_queue.log
watch -n 2 nvidia-smi
```

If one queue finishes early, that is okay. The split keeps the laptop on the
three most useful reference runs and puts the heavier paid grid on Vast.

## Monitor Runs

```bash
# Watch logs.
tail -f vast_logs/*.log

# Check GPU use.
watch -n 2 nvidia-smi

# Check local artifacts.
find runs -maxdepth 2 -type f | sort | tail -50
```

## Download Results Back To Local Machine

Run this from your local machine, replacing host/port/path with the Vast.ai SSH
details.

```bash
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/workspace/cs503_project/runs/ gym_sokoban/runs/
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/workspace/cs503_project/vast_logs/ gym_sokoban/vast_logs/
```
