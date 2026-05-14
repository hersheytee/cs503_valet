# Oracle-Augmented PPO — Upper Bound Experiments

> CS503 project — EPFL  
> Research question: **Can a PPO agent learn to use a VLM as an optional action, and does it learn when it's worth the cost?**

The agent lives in a MiniGrid environment and has one extra action: `query_oracle`. When chosen, the oracle (a perfect BFS solver) executes the optimal action. Querying costs a configurable reward penalty. The agent must therefore learn both *how* to solve the task and *when* it is worth consulting the oracle.

---

## Repository structure

```
Upper_Bound/
├── train.py              # Main training loop (PPO)
├── env_wrapper.py        # Gym wrapper — adds oracle action to MiniGrid
├── oracle.py             # BFS oracle for DoorKey / Empty envs
├── oracle_transfer.py    # BFS oracle for transfer envs (Fetch, GoToDoor, GoToObject)
├── model.py              # CNN policy for 8x8 grids
├── model_large.py        # CNN policy for 16x16 grids (strided conv + AdaptiveAvgPool)
├── eval.py               # Inference script — runs a checkpoint and saves a GIF
├── compare_plot.py       # Multi-condition plot (one row per condition)
├── merged_plot.py        # Merged plot (all conditions overlaid, 3 vertical subplots)
├── plot.py               # Single-run debug plot
├── requirements.txt      # Python dependencies
├── submit_all.sh         # SLURM job — DoorKey-8x8, all oracle costs, 5 seeds
├── submit_16x16.sh       # SLURM job — DoorKey-16x16, all oracle costs, 5 seeds
├── logs/                 # CSV training logs (auto-created)
├── figures/              # Output plots and GIFs (auto-created)
└── checkpoints/          # Saved model weights (auto-created)
```

---

## Core concepts

| Concept | Description |
|---|---|
| `query_oracle` | Extra action (index `N`) added to the original action space. Executes the BFS-optimal action and marks the step as *guided*. |
| `oracle_cost` | Reward penalty subtracted each time the oracle is queried. At `0.0` the oracle is free (upper bound). At `0.05` the agent must earn back the cost by solving faster. |
| `reward_shaping` | Intermediate rewards for DoorKey: `+0.5` on key pickup, `+0.5` on door open. Helps the agent explore the task structure. |
| `success` | Episode terminates with `terminated=True` (goal reached), as opposed to `truncated=True` (timeout). |
| BFS oracle | Runs a full graph search on the true env state. Serves as a proxy for a perfect VLM with zero hallucination. |

---

## Files in detail

### `train.py` — PPO training

Standard CleanRL-style PPO with one extra BC (behavioural cloning) loss term on oracle-guided steps (disabled by default with `--bc-coef 0.0`).

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--env-id` | `MiniGrid-Empty-5x5-v0` | Gymnasium environment ID |
| `--env-type` | `empty` | Oracle type: `empty`, `doorkey`, `fetch`, `gotodoor`, `gotoobject` |
| `--oracle-cost` | `0.0` | Reward penalty per oracle query. `0.0` = free oracle (upper bound) |
| `--no-oracle` | `False` | Disable oracle action entirely — pure PPO baseline |
| `--reward-shaping` | `False` | Add intermediate rewards for key/door events (DoorKey only) |
| `--warmup-steps` | `0` | Force oracle queries for the first N steps (curriculum) |
| `--total-timesteps` | `500_000` | Total environment steps |
| `--n-envs` | `8` | Number of parallel environments |
| `--n-steps` | `128` | Rollout length per env before each PPO update |
| `--n-minibatches` | `4` | Number of minibatches per epoch |
| `--n-epochs` | `4` | PPO update epochs per rollout |
| `--lr` | `2.5e-4` | Learning rate (linearly annealed by default) |
| `--gamma` | `0.99` | Discount factor |
| `--clip-coef` | `0.2` | PPO clip coefficient |
| `--ent-coef` | `0.01` | Entropy coefficient |
| `--bc-coef` | `0.0` | Behavioural cloning loss weight on guided steps |
| `--hidden-dim` | `256` | FC layer size after CNN |
| `--seed` | `1` | Random seed |
| `--save-model` | `False` | Save checkpoint to `checkpoints/` at end of training |
| `--large-model` | `False` | Use `model_large.py` (required for 16x16 grids) |
| `--exp-name` | `oracle_ppo` | Prefix for log CSV and checkpoint filenames |

**Output:** CSV log at `logs/{exp_name}__{env_id}__seed{seed}__{timestamp}.csv`

---

### `env_wrapper.py` — Oracle environment wrapper

Wraps any MiniGrid environment with:
- `FullyObsWrapper` — full grid visibility
- `RGBImgObsWrapper` — RGB image observation (40×40 for 8×8 grid with `tile_size=8`)
- `+1` extra action: `query_oracle`
- Optional `RewardShaper` for intermediate rewards

The dispatcher routes oracle calls automatically:
- `env_type ∈ {empty, doorkey}` → `oracle.py`
- `env_type ∈ {fetch, gotodoor, gotoobject}` → `oracle_transfer.py`

---

### `oracle.py` — BFS oracle for DoorKey / Empty

Provides a BFS-optimal action for each env step.

- **`bfs_empty(env)`** — Navigate to the goal cell. State: `(x, y, dir)`.
- **`bfs_doorkey(env)`** — Pick up key, open door, reach goal. State: `(x, y, dir, has_key, door_open)`.
- **`get_oracle_action(env_unwrapped, env_type)`** — Entry point. Returns `6` (done) when the task is already complete.

---

### `oracle_transfer.py` — BFS oracle for transfer environments

Same BFS logic adapted to three new MiniGrid tasks. Mission text is parsed to identify the target object (color + type).

- **`bfs_fetch(env)`** — Navigate to the target object and pick it up. Supports any color/type combination parsed from `env.mission`.
- **`bfs_gotodoor(env)`** — Navigate to be within Chebyshev distance 1 of the target door.
- **`bfs_gotoobject(env)`** — Navigate to be within Chebyshev distance 1 of the target object.
- **`get_oracle_action(env_unwrapped, env_type)`** — Entry point for `env_type ∈ {fetch, gotodoor, gotoobject}`.

---

### `model.py` — CNN policy for 8×8 grids

Three conv layers (stride=1), flatten, FC(→256), policy head + value head. Designed for 40×40×3 RGB input. Weights are fixed at this architecture for checkpoint compatibility.

---

### `model_large.py` — CNN policy for 16×16 grids

Same interface as `model.py` but uses strided convolutions (stride=2, 2, 1) + `AdaptiveAvgPool2d(8, 8)` to keep the FC input at a fixed 4096 dimensions regardless of grid size. Required for 16×16 or larger environments.

---

### `eval.py` — Inference + GIF generation

Loads a saved checkpoint, runs N episodes, and saves a GIF.

**Parameters:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path or glob to `.pt` checkpoint file |
| `--env-id` | `MiniGrid-DoorKey-8x8-v0` | Environment to evaluate on |
| `--env-type` | `doorkey` | Oracle type (same values as `train.py`) |
| `--no-oracle` | `False` | Evaluate without oracle action (baseline checkpoint) |
| `--oracle-cost` | `0.0` | Oracle cost used during eval (for reward reporting) |
| `--reward-shaping` | `False` | Apply reward shaping during eval |
| `--n-episodes` | `3` | Number of episodes to run |
| `--hidden-dim` | `256` | Must match the training hidden dim |
| `--tile-size` | `8` | Must match the training tile size |
| `--fps` | `6` | GIF frame rate |
| `--out` | `figures/eval.gif` | Output GIF path |
| `--seed` | `42` | Starting seed (incremented per episode) |

---

### `compare_plot.py` — Multi-condition comparison plot

Produces a figure with one row per oracle condition (free / paid×5 / baseline) and four metric columns: episode return, success rate, oracle usage %, queries per episode.

Each metric is plotted as a function of total environment steps. Raw per-episode data is interpolated to a common step grid and smoothed with a rolling mean (window=50).

**Parameters:**

| Argument | Description |
|---|---|
| `--free-csv` | Glob(s) to oracle_free CSV logs |
| `--paid-001-csv` | Glob(s) to oracle_paid_001 CSV logs |
| `--paid-002-csv` | ... |
| `--paid-003-csv` | ... |
| `--paid-004-csv` | ... |
| `--paid-005-csv` | ... |
| `--base-csv` | Glob(s) to baseline CSV logs |
| `--out` | Output PNG path (default: `figures/overview.png`) |
| `--min-steps` | Ignore CSVs with fewer than this many steps |

When multiple CSVs exist for the same seed, only the one with the latest timestamp is used (avoids mixing old and new runs).

---

### `merged_plot.py` — Merged overlay plot

Same data as `compare_plot.py` but all conditions are overlaid on the same axes (3 vertical subplots: return, success rate, oracle usage %). Uses a viridis colormap for oracle conditions and black for baseline.

**Parameters:**

| Argument | Description |
|---|---|
| `--free-csv` | Glob(s) to oracle_free CSV logs |
| `--paid-low-csv` | Glob(s) to low-cost oracle logs |
| `--paid-mid-csv` | Glob(s) to mid-cost oracle logs |
| `--paid-high-csv` | Glob(s) to high-cost oracle logs |
| `--base-csv` | Glob(s) to baseline CSV logs |
| `--out` | Output PNG path (default: `figures/merged.png`) |
| `--min-steps` | Ignore CSVs with fewer than this many steps |

---

### `submit_all.sh` — SLURM job for DoorKey-8x8

Runs all 7 conditions × 5 seeds sequentially on a single GPU node (20h time limit).

Conditions: `oracle_free` (cost=0.0), `oracle_paid_001` through `oracle_paid_005`, `baseline`.  
Seeds: 4, 5, 6, 7, 8.

---

### `submit_16x16.sh` — SLURM job for DoorKey-16x16

Same structure as `submit_all.sh` but for `MiniGrid-DoorKey-16x16-v0` with:
- `--large-model` flag on all runs
- `--total-timesteps 2000000` (4× more than 8×8)
- 24h time limit

---

## Installation

```bash
conda activate nanofm   # or your env
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0.0
gymnasium>=0.29.0
minigrid>=3.0.0
numpy>=1.24.0
```

---

## Running experiments

### Quick local test

```bash
# Oracle free (upper bound)
python train.py \
    --env-id MiniGrid-DoorKey-8x8-v0 --env-type doorkey \
    --oracle-cost 0.0 --reward-shaping \
    --total-timesteps 100000 --seed 1 --exp-name test_free

# Paid oracle
python train.py \
    --env-id MiniGrid-DoorKey-8x8-v0 --env-type doorkey \
    --oracle-cost 0.02 --reward-shaping \
    --total-timesteps 100000 --seed 1 --exp-name test_paid

# Baseline (no oracle)
python train.py \
    --env-id MiniGrid-DoorKey-8x8-v0 --env-type doorkey \
    --no-oracle --reward-shaping \
    --total-timesteps 100000 --seed 1 --exp-name test_baseline
```

### On cluster (SLURM)

```bash
# From the Upper_Bound/ directory:
cd ~/Upper_Bound
sbatch submit_all.sh      # DoorKey-8x8, all conditions
sbatch submit_16x16.sh    # DoorKey-16x16, all conditions
```

### Transfer environment (Fetch-8x8)

```bash
python train.py \
    --env-id MiniGrid-Fetch-8x8-N3-v0 --env-type fetch \
    --oracle-cost 0.0 --reward-shaping \
    --total-timesteps 500000 --seed 1 --exp-name fetch_free --save-model
```

Supported transfer env_types:
- `fetch` → `MiniGrid-Fetch-8x8-N3-v0`
- `gotodoor` → `MiniGrid-GoToDoor-8x8-v0`
- `gotoobject` → `MiniGrid-GoToObject-8x8-N2-v0`

---

## Generating plots

### Multi-condition comparison (separate rows)

```bash
python compare_plot.py \
    --free-csv      logs/oracle_free__MiniGrid-DoorKey-8x8-v0__seed*.csv \
    --paid-001-csv  logs/oracle_paid_001__*.csv \
    --paid-002-csv  logs/oracle_paid_002__*.csv \
    --paid-003-csv  logs/oracle_paid_003__*.csv \
    --paid-004-csv  logs/oracle_paid_004__*.csv \
    --paid-005-csv  logs/oracle_paid_005__*.csv \
    --base-csv      logs/baseline__*.csv \
    --out figures/overview_doorkey.png
```

### Merged overlay plot

```bash
python merged_plot.py \
    --free-csv      logs/oracle_free__*.csv \
    --paid-low-csv  logs/oracle_paid_001__*.csv \
    --paid-mid-csv  logs/oracle_paid_003__*.csv \
    --paid-high-csv logs/oracle_paid_005__*.csv \
    --base-csv      logs/baseline__*.csv \
    --out figures/merged_doorkey.png
```

---

## Inference and visualization

```bash
# Evaluate a saved checkpoint and produce a GIF
python eval.py \
    --checkpoint "checkpoints/oracle_free__MiniGrid-DoorKey-8x8-v0__seed4__*.pt" \
    --env-id MiniGrid-DoorKey-8x8-v0 --env-type doorkey \
    --n-episodes 5 --out figures/eval_doorkey.gif

# Transfer: evaluate a DoorKey checkpoint on Fetch
python eval.py \
    --checkpoint "checkpoints/oracle_free__MiniGrid-DoorKey-8x8-v0__seed4__*.pt" \
    --env-id MiniGrid-Fetch-8x8-N3-v0 --env-type fetch \
    --n-episodes 5 --out figures/transfer_doorkey_to_fetch.gif
```

---

## Sending files to cluster

```bash
# Full sync
scp Upper_Bound/*.py Upper_Bound/*.sh Upper_Bound/requirements.txt \
    tjouven@izar.epfl.ch:~/Upper_Bound/

# Retrieve logs and figures
scp -r tjouven@izar.epfl.ch:~/Upper_Bound/logs ./
scp -r tjouven@izar.epfl.ch:~/Upper_Bound/figures ./
scp -r tjouven@izar.epfl.ch:~/Upper_Bound/checkpoints ./
```
