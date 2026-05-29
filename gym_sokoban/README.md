# Sokoban Oracle PPO

PPO agent on `gym-sokoban` with an optional learnable oracle-query action.
The agent can ask a BFS oracle for the next move; oracle quality (accuracy) and
cost (reward penalty) are configurable. The experiment studies how query
behavior responds to expert cost and reliability.

---

## Project overview

| Concept | Detail |
|---|---|
| Environment | `Sokoban-small-v0` — 7×7 board, 2 boxes, 56×56 RGB |
| Baseline | Pure PPO, `--no-oracle` |
| Oracle | Extra action `query_oracle` executes a BFS-optimal move and subtracts `--oracle-cost` from reward |
| Oracle accuracy | `--oracle-accuracy P` — probability oracle returns the BFS action; rest is a random non-optimal action |
| Linear cost ramp | `--oracle-cost 0.0 --oracle-cost-final C` schedules cost linearly over training |
| Budget oracle | `--max-oracle-queries N` caps queries per episode; remaining budget encoded as a 4th image channel |
| Transfer | Zero-shot evaluation on `FixedTarget-Sokoban-v2` after training on `Sokoban-small-v0` |

---

## Repository layout

```
gym_sokoban/
├── train.py               # PPO training loop (single-file CleanRL style)
├── env_wrapper.py         # SokobanOracleWrapper — oracle action, budget channel, gym compat
├── model.py               # CNNPolicy — 3- or 4-channel, auto-detected from checkpoint
├── bfs_oracle.py          # BFS oracle (standard + FixedTarget env variants)
├── eval_gif.py            # Render evaluation episodes to an annotated GIF
├── eval_transfer.py       # Zero-shot transfer eval, single checkpoint
├── eval_transfer_all.py   # Batch transfer eval across all checkpoints (parallel)
├── plot_results.py        # Generate final figures from run artifacts
├── benchmark_oracle.py    # Measure BFS solve rate and speed on sampled puzzles
├── test_oracle.py         # Visual smoke test: let the raw BFS oracle play
├── test_wrapper.py        # Smoke test for SokobanOracleWrapper
├── test_model.py          # Smoke test for CNNPolicy
├── job.sh                 # SLURM single-run wrapper (Izar)
├── setup_vast.sh          # One-shot Vast.ai instance setup script
├── COMMANDS.md            # Full Vast.ai launch commands and job scripts
└── figures/final/         # Output directory for figures and GIFs
    └── transfer_results.csv
```

**Root-level scripts (repo root, not `gym_sokoban/`):**
- `gen_eval_gifs.ps1` — batch-generate all 6 condition GIFs (PowerShell, Windows)

---

## Architecture

All runs use the MiniGrid partial-observation CNN on `56x56x3` Sokoban sprites:

```
Input: 56×56×3 uint8 → float, channel-first

Conv2d(3,  32, kernel=3, stride=2, padding=1)   → 32×28×28
Conv2d(32, 64, kernel=3, stride=1, padding=1)   → 64×28×28
Conv2d(64, 64, kernel=3, stride=1, padding=1)   → 64×28×28
AdaptiveAvgPool2d(8, 8)                          → 4096
Linear(4096, 256) → policy head + value head
```

Budget-aware runs (`--max-oracle-queries`) use 4 input channels instead of 3
(4th channel = remaining budget as a spatially-constant uint8). These checkpoints
are incompatible with standard 3-channel checkpoints.

---

## Environment setup

### Local (Windows, venv)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify CUDA:
python -c "import torch; print(torch.cuda.is_available())"
```

### Izar (EPFL cluster, conda)

```bash
conda activate cs503_proj
pip install wandb
wandb login
```

### Vast.ai

```bash
git clone https://github.com/hersheytee/cs503_project.git
cd cs503_project

export WANDB_API_KEY="<your-key>"
bash gym_sokoban/setup_vast.sh        # fresh setup
bash gym_sokoban/setup_vast.sh --pull # update existing repo

# V100 only — reinstall PyTorch with cu121 (sm_70 support):
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Smoke test

Run before any real training to confirm the stack works:

```bash
python -m py_compile gym_sokoban/train.py gym_sokoban/env_wrapper.py gym_sokoban/model.py

python gym_sokoban/train.py \
  --env-id Sokoban-small-v0 \
  --seed 1 \
  --exp-name smoke_test \
  --total-timesteps 4096 \
  --n-envs 8 \
  --n-steps 64 \
  --n-minibatches 4 \
  --no-oracle \
  --no-track
```

---

## Training runs

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--env-id` | `Sokoban-small-v0` | Gym environment |
| `--seed` | `1` | Random seed |
| `--exp-name` | required | Name used for the output directory |
| `--total-timesteps` | `500000` | Total env frames |
| `--n-envs` | `8` | Parallel environments (recommend 64) |
| `--n-steps` | `256` | Steps per env before PPO update |
| `--n-minibatches` | `4` | Minibatches per update (use 8 with 64 envs) |
| `--no-oracle` | off | Baseline PPO — removes oracle action entirely |
| `--oracle-cost` | `0.0` | Reward penalty per oracle query |
| `--oracle-cost-final` | none | If set, linearly ramps cost from `--oracle-cost` to this value |
| `--oracle-cost-anneal-steps` | total steps | Steps to complete the linear ramp |
| `--oracle-accuracy` | `1.0` | Probability oracle returns BFS-optimal action |
| `--max-oracle-queries` | none | Hard per-episode query budget (enables 4-channel obs) |
| `--save-model` | off | Save `checkpoints/final.pt` at end of run |
| `--wandb-project` | none | W&B project name |
| `--wandb-group` | none | W&B run group |
| `--no-track` | off | Disable W&B entirely |
| `--wandb-mode offline` | — | Buffer W&B offline; sync later with `wandb sync wandb/offline-run-*` |

### Recommended batch settings

```
n-envs=64, n-steps=256, n-minibatches=8
rollout batch = 64 × 256 = 16,384
optimizer minibatch = 16,384 / 8 = 2,048
```

Expected throughput: ~330 SPS on a V100, implying ~25 min for 500k steps.

### Run artifact layout

Each run writes to a self-contained directory:

```
runs/YYYYMMDD_HHMMSS__<exp-name>__<env-id>__seed<N>/
├── config.yaml           # all CLI args + derived PPO sizes, git commit, device
├── logs/stdout.log
├── data/metrics.csv      # per-episode metrics (canonical analysis source)
├── figures/training_metrics.png
└── checkpoints/final.pt  # saved only with --save-model
```

---

## Option A: Izar (EPFL SLURM cluster)

### Submit a single run

```bash
sbatch \
  --job-name="soko_baseline_s1" \
  --export=ALL,ENV_ID=Sokoban-small-v0,EXP_NAME=baseline_500k,SEED=1,TOTAL_TIMESTEPS=500000,EXTRA_ARGS="--no-oracle --n-envs 64 --n-steps 256 --n-minibatches 8 --wandb-project cs503-sokoban --wandb-group sokoban_baseline" \
  gym_sokoban/job.sh
```

`job.sh` activates the `cs503_proj` conda env and calls `train.py` with the
exported variables. Default wall time is 12 h; adjust `#SBATCH --time` as needed.

### Monitor SLURM jobs

```bash
squeue -u $USER

sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,MaxRSS

tail -f logs/slurm_<JOBID>.out
tail -f logs/slurm_<JOBID>.err

grep "sps=" logs/slurm_*.out | tail -40

scancel <JOBID>
scancel -u $USER     # cancel all your jobs
```

### Offline W&B on Izar

If Izar networking is unreliable:

```bash
python gym_sokoban/train.py ... --wandb-mode offline

# Sync later from a machine with internet access:
wandb sync wandb/offline-run-*
```

---

## Option B: Vast.ai

See [COMMANDS.md](COMMANDS.md) for the full set of ready-to-run scripts.
Summary below.

### Recommended instance

```
Template: PyTorch (not vLLM/SGLang)
Launch mode: SSH or Jupyter + SSH
CUDA: 12.x
Python: 3.10 / 3.11
Disk: 60–100 GB
GPUs: 2× RTX 4000 / A4000 class preferred
```

After SSH:

```bash
# Install deps (preserves Vast PyTorch/CUDA install):
bash gym_sokoban/setup_vast.sh

# Sanity check:
python -m py_compile gym_sokoban/train.py gym_sokoban/env_wrapper.py gym_sokoban/model.py
```

### Launch a run on Vast

See [COMMANDS.md](COMMANDS.md) for ready-to-run job scripts covering individual
conditions and multi-job grids across two GPUs.

### Start a tmux session (Vast)

```bash
tmux new-session -d -s soko './run_vast_paid_grid.sh'
tmux attach -t soko_paid_grid

# Detach without killing jobs:
# Ctrl-b d

# Monitor:
tail -f vast_logs/gpu*_queue.log
watch -n 2 nvidia-smi
```

### Download results from Vast

Run from your local machine, filling in the SSH details from the Vast console:

```bash
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/workspace/cs503_project/runs/ gym_sokoban/sokoban_results/runs/
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/workspace/cs503_project/vast_logs/ gym_sokoban/vast_logs/
```

---

## Checkpoint management

Checkpoints are saved as `checkpoints/final.pt` inside the run directory when
`--save-model` is passed. The run's `config.yaml` records all training
parameters needed to reproduce the model.

`gym_sokoban/sokoban_results/` is excluded from git entirely via `.gitignore`
due to size. It is not included in the repo — you need to produce it by running
the training jobs (see [Training runs](#training-runs) and [COMMANDS.md](COMMANDS.md))
and placing the resulting run directories under `gym_sokoban/sokoban_results/runs/`.

To archive a set of completed runs before destroying a Vast instance:

```bash
# On the Vast instance:
tar -czf sokoban_results.tar.gz runs/
# Then rsync the .tar.gz to your local machine.
```

**Important:** 3-channel (standard) and 4-channel (budget-oracle) checkpoints
are not interchangeable. The channel count is auto-detected from the weights
at eval time.

---

## Analyzing results

### Generate figures

```bash
python gym_sokoban/plot_results.py --out-dir gym_sokoban/figures/final
```

| Output file | Content |
|---|---|
| `cost_curves_acc100.png` | Success rate + oracle usage + queries/ep vs steps, all costs at acc=1.0 |
| `accuracy_lines.png` | Final success + usage + queries/ep vs oracle accuracy, one line per cost |
| `linear_cost_schedule.png` | 2×3 panel: scheduled cost, success, reward, usage, queries/ep, success/query |
| `transfer_results.png` | Zero-shot transfer to FixedTarget: success + usage + queries/ep vs accuracy |

Key flags for `plot_results.py`:

```bash
--no-shade          # disable ±1 std shading (on by default)
--window N          # rolling average window (default 50, over 500 uniform step points)
--min-steps N       # skip runs shorter than N steps (default 450k)
--out-dir PATH      # output directory (default gym_sokoban/figures/final)
```

### Zero-shot transfer evaluation

```bash
# Single checkpoint:
python gym_sokoban/eval_transfer.py \
  --checkpoint runs/.../checkpoints/final.pt \
  --env-id FixedTarget-Sokoban-v2 \
  --n-episodes 200

# All checkpoints at once (parallel, skips already-evaluated pairs):
python -u gym_sokoban/eval_transfer_all.py \
  --workers 8 \
  --n-episodes 200 \
  --min-steps 450000
# -> gym_sokoban/figures/final/transfer_results.csv
```

Delete `transfer_results.csv` to force a full re-evaluation.

### BFS oracle sanity checks

```bash
# Measure BFS solve rate and speed:
python gym_sokoban/benchmark_oracle.py

# Watch the raw BFS oracle play interactively:
python gym_sokoban/test_oracle.py
```

---

## Producing evaluation GIFs

### Single GIF

```bash
python gym_sokoban/eval_gif.py \
  --checkpoint runs/.../checkpoints/final.pt \
  --n-episodes 5 \
  --fps 4 \
  --scale 4 \
  --stochastic \
  --success-only \
  --out gym_sokoban/figures/final/gifs/baseline.gif \
  --no-oracle
```

Key flags:

| Flag | Description |
|---|---|
| `--no-oracle` | Baseline checkpoint (no oracle action) |
| `--stochastic` | Sample from the policy distribution (recommended; looks natural) |
| `--success-only` | Retry seeds until N successful episodes are collected |
| `--max-episode-steps` | Episode horizon (default 120; training uses 50) |
| `--oracle-accuracy` | Match the value used during training |
| `--max-oracle-queries N` | Required for 4-channel budget checkpoints |
| `--scale` | Upscale factor for the board pixels (default 4) |
| `--fps` | GIF frame rate (default 4) |
| `--env-id` | Override environment for transfer GIFs |

The rendered GIF shows: upscaled board, action probability bars below, a red
border + "ORACLE QUERY" banner on guided steps, and a HUD with step/action/reward/queries.

### Batch GIF generation (all 6 conditions)

Checkpoint paths are hard-coded in `gen_eval_gifs.ps1` (repo root). Edit them
if you re-run training, then:

```powershell
# From repo root:
.\gen_eval_gifs.ps1 -NEpisodes 5 -Fps 4 -Scale 4
```

Outputs to `gym_sokoban/figures/gifs/`. The 6 standard conditions are:
`baseline`, `oracle_free`, `oracle_cost01`, `oracle_cost05`, `oracle_cost08`,
`oracle_linear`.

The script also generates transfer GIFs for `FixedTarget-Sokoban-v2`.
Skips any GIF that already exists.

---

## Notes

**V100 PyTorch compatibility:** PyTorch 2.11+ nightly drops sm_70 support.
If you see `CUDA error: no kernel image`, reinstall with cu121:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4-channel / 3-channel checkpoint mismatch:** Budget-oracle runs
(`--max-oracle-queries`) produce 4-channel weights. Passing a 4-channel
checkpoint to `eval_gif.py` without `--max-oracle-queries` will raise an error.
