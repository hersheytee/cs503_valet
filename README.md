# VALET — Value-Aware Learning with Expert Tutoring

> CS503 — EPFL  
> **Research question:** Can a PPO agent learn to use an oracle (BFS or VLM) as an optional action, and does it learn *when* it is worth the cost?

The agent has one extra action: `query_oracle`. When chosen, the oracle executes the optimal action and the agent pays a configurable reward penalty. The agent must learn both *how* to solve the task and *when* consulting the oracle is worth it.

---

## Project structure

```
cs503_project/
├── minigrid/               # Training & evaluation code for MiniGrid environments
├── Upper_Bound/            # Analysis scripts — plots, transfer evaluation, per-cost stats
├── gym_sokoban/            # Sokoban experiments (PPO baseline + oracle)
├── Benchmark/              # VLM oracle action benchmark (dataset + evaluation)
├── Literature review/      # Related papers
├── VALET proposal.pdf      # Original project proposal
├── AGENTS.md               # High-level agent design notes
├── checkpoints/            # Sokoban checkpoints
├── logs/                   # Root-level logs
├── figures/                # Root-level figures
└── requirements.txt        # Root Python dependencies
```

---

## Components

### `minigrid/` — Core training code

PPO agent with oracle action in MiniGrid environments. All main experiments run from here.

**Key files:**

| File | Description |
|---|---|
| `train.py` | PPO training loop (CleanRL-style) |
| `env_wrapper.py` | Gym wrapper — adds `query_oracle` action, RGB obs |
| `oracle.py` | BFS oracle for DoorKey / Empty |
| `oracle_transfer.py` | BFS oracle for Fetch, GoToDoor, GoToObject, MultiRoom |
| `model.py` | CNN policy — 40×40×3 input, ~26M params (8×8 grids) |
| `model_large.py` | CNN policy — 128×128×3, ~1.1M params (16×16 full obs) |
| `model_partial.py` | CNN policy — 56×56×3, ~1.1M params (partial obs) |
| `eval.py` | Run checkpoint + save GIF |
| `compare_plot.py` | One row per condition, 4 metric columns |
| `merged_plot.py` | All conditions overlaid, plasma colorbar, horizontal layout |
| `vlm_oracle.py` | Replaces BFS oracle with a VLM (Qwen2-VL, InternVL, …) |
| `download_models.py` | Download VLMs from HuggingFace to cluster |
| `job_vlm.sh` | SLURM job for VLM training runs |

**Environments trained:**

| Environment | Obs | Steps | Notes |
|---|---|---|---|
| `MiniGrid-DoorKey-16x16-v0` | Full obs 128×128×3 | 2M | Full sweep of oracle costs |
| `MiniGrid-DoorKey-16x16-v0` | Partial obs 56×56×3 | 2M | Main experiments |
| `MiniGrid-DoorKey-8x8-v0` | Full obs 40×40×3 | 500k | Small-scale tests |

**Oracle costs tested (partial obs, 16×16):**

`0.000 (free)` · `0.007` · `0.008` · `0.009` · `0.010` · `0.011` · `0.012` · `0.015` · `0.018` · `0.020` · `0.030` · `0.040` · `0.050` · `baseline (no oracle)`

**Key finding:** Agent self-regulates oracle usage. Usage drops sharply between cost 0.010 and 0.012 — below this threshold the agent queries freely, above it the agent stops querying entirely.

---

### `Upper_Bound/` — Analysis & evaluation

Scripts for evaluating trained models and generating plots. Designed to run on the EPFL Izar cluster.

**Key files:**

| File | Description |
|---|---|
| `eval_transfer_stats.py` | Zero-shot transfer evaluation (100 episodes, full stats) |
| `per_cost_plot.py` | One 1×4 training curve plot per oracle cost vs baseline |
| `merged_plot.py` | Merged overlay plot |
| `submit_eval_transfer_stats.sh` | Transfer eval on Fetch-16x16 |
| `submit_eval_transfer_stats_multiroom.sh` | Transfer eval on MultiRoom-N6 |
| `submit_eval_all_partial_doorkey.sh` | GIFs for all partial-obs models |
| `job_noise.sh` | Noisy oracle sweep (oracle correct with prob p) |
| `noise_plot.py` | Plot for noisy oracle experiments |
| `vlm_plot.py` | Plot for VLM oracle experiments |

**Transfer experiments (zero-shot):**

Models trained on `DoorKey-16×16` evaluated on unseen tasks without finetuning:

| Target | Result |
|---|---|
| `MiniGrid-Fetch-16x16-N3-v0` | Oracle model transfers well; queries at key moments |
| `MiniGrid-MultiRoom-N6-v0` | Oracle model uses oracle at room transitions |

Trajectory efficiency = `BFS_opt_steps / agent_steps` (success episodes only).

---

### `gym_sokoban/` — Sokoban experiments

PPO baseline on `Sokoban-small-v0` (56×56×3 RGB). Recreates a WorldCoder-style baseline as a foundation for Sokoban-to-Sokoban transfer experiments.

---

### `Benchmark/` — VLM oracle benchmark

Dataset and evaluation pipeline for measuring how well VLMs predict the BFS-optimal action from a MiniGrid RGB observation.

| File | Description |
|---|---|
| `create_dataset.py` | Generate (observation, oracle_action) pairs |
| `create_history_dataset.py` | Dataset with action history context |
| `bench_job.sh` | SLURM job for benchmark runs |
| `analyse_results.ipynb` | Result analysis notebook |

---

## Training setup

- **Algorithm:** PPO with clipped surrogate objective (clip=0.2), GAE (λ=0.95), entropy bonus (coef=0.01)
- **Entropy:** Applied only to non-guided steps
- **Observation:** Raw RGB image — no extra flags or augmentation
- **Architecture:** Shared CNN encoder → FC(256) → policy head + value head
- **Oracle:** Available as action `N`. Executes BFS-optimal step, penalizes reward by `oracle_cost`

---

## Quick start

```bash
conda activate nanofm
pip install -r minigrid/requirements.txt

# Train — DoorKey-16x16, partial obs, oracle cost=0.01
python minigrid/train.py \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --oracle-cost 0.01 --reward-shaping --large-model --partial-obs \
    --total-timesteps 2000000 --seed 4 \
    --exp-name oracle_paid_001_16_partial --save-model

# Evaluate + GIF
python minigrid/eval.py \
    --checkpoint minigrid/checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --partial-obs --stochastic --n-episodes 3 \
    --out minigrid/all_gif/paid001.gif

# Zero-shot transfer
python Upper_Bound/eval_transfer_stats.py \
    --checkpoint minigrid/checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id MiniGrid-Fetch-16x16-N3-v0 --env-type fetch \
    --partial-obs --n-episodes 100 \
    --csv-out Upper_Bound/figures/fetch_comparison.csv \
    --out Upper_Bound/figures/fetch_comparison.png
```

## Cluster sync

```bash
# Push code
rsync -avz ~/Desktop/cs503_project/minigrid/ tjouven@izar.epfl.ch:~/Upper_Bound/

# Pull results
rsync -avz tjouven@izar.epfl.ch:~/Upper_Bound/logs/     ~/Desktop/cs503_project/minigrid/logs/
rsync -avz tjouven@izar.epfl.ch:~/Upper_Bound/figures/  ~/Desktop/cs503_project/Upper_Bound/figures/
rsync -avz "tjouven@izar.epfl.ch:~/Upper_Bound/checkpoints/best__*_partial__*.pt" \
    ~/Desktop/cs503_project/minigrid/checkpoints/
```

The agent lives in a MiniGrid environment and has one extra action: `query_oracle`. When chosen, the oracle (a perfect BFS solver) executes the optimal action. Querying costs a configurable reward penalty. The agent must therefore learn both *how* to solve the task and *when* it is worth consulting the oracle.

---

## Repository structure

```
Upper_Bound/
├── train.py                          # Main training loop (PPO)
├── env_wrapper.py                    # Gym wrapper — adds oracle action to MiniGrid
├── oracle.py                         # BFS oracle for DoorKey / Empty envs
├── oracle_transfer.py                # BFS oracle for transfer envs (Fetch, GoToDoor, GoToObject, MultiRoom)
├── model.py                          # CNN policy for 8x8 grids (full obs, 40x40x3, ~26M params)
├── model_large.py                    # CNN policy for 16x16 grids (full obs, 128x128x3, ~1.1M params)
├── model_partial.py                  # CNN policy for partial obs (56x56x3, ~1.1M params)
├── eval.py                           # Inference script — runs a checkpoint and saves a GIF
├── eval_transfer_stats.py            # Zero-shot transfer evaluation with detailed stats + plots
├── compare_plot.py                   # Multi-condition plot (one row per condition)
├── merged_plot.py                    # Merged plot (all conditions overlaid, horizontal layout, plasma colorbar)
├── per_cost_plot.py                  # One 1×4 plot per oracle cost vs baseline
├── plot.py                           # Single-run debug plot
├── requirements.txt                  # Python dependencies
├── submit_all.sh                     # SLURM job — DoorKey-8x8, all oracle costs, 5 seeds
├── submit_16x16.sh                   # SLURM job — DoorKey-16x16, full+partial obs, 5 seeds
├── submit_16x16_partial_fine.sh      # Fine-grained cost sweep (0.01–0.05) partial obs
├── submit_16x16_partial_vfine.sh     # Very fine-grained cost sweep (0.007–0.012) partial obs
├── submit_eval_transfer.sh           # SLURM — GIF generation for Fetch transfer
├── submit_eval_transfer_stats.sh     # SLURM — Transfer stats on Fetch-16x16 (100 eps)
├── submit_eval_transfer_stats_multiroom.sh  # SLURM — Transfer stats on MultiRoom-N6 (100 eps)
├── submit_eval_multiroom.sh          # SLURM — GIF for MultiRoom transfer
├── submit_eval_all_partial_doorkey.sh # SLURM — GIFs for all partial-obs models on DoorKey-16x16
├── logs/                             # CSV training logs (auto-created)
├── figures/                          # Output plots and GIFs (auto-created)
│   ├── per_cost/                     # Per-cost comparison plots
│   └── gifs/                         # GIF visualizations
├── all_gif/                          # All DoorKey-16x16 GIFs (one per model)
└── checkpoints/                      # Saved model weights
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
| `trajectory efficiency` | `opt_steps / agent_steps` — ratio of BFS-optimal path length to actual steps taken (only on successful episodes). |

---

## Model architectures

| Model | Input | Params | Notes |
|---|---|---|---|
| `model.py` | 40×40×3 | ~26.3M | No pooling — huge FC layer. For 8×8 grids. |
| `model_large.py` | 128×128×3 | ~1.1M | AdaptiveAvgPool(8,8). For 16×16 full obs. |
| `model_partial.py` | 56×56×3 | ~1.1M | AdaptiveAvgPool(8,8). For partial obs (any grid). |

All models: shared CNN backbone → FC(256) → policy head + value head. Weights are initialized with orthogonal init (std=√2 for conv/FC, 0.01 for policy head, 1.0 for value head).

---

## Training setup

PPO with:
- Clipped surrogate objective (`clip_coef=0.2`)
- Generalized Advantage Estimation (`gae_lambda=0.95`)
- Entropy bonus (`ent_coef=0.01`) applied only to **non-guided** steps
- Optional BC loss on oracle-guided steps (`bc_coef=0.0` by default)
- Linear learning rate annealing

The observation is the raw RGB image — no extra flags or augmentation. The agent has no explicit signal about whether it just queried the oracle beyond the reward it received.

---

## Files in detail

### `train.py` — PPO training

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--env-id` | `MiniGrid-Empty-5x5-v0` | Gymnasium environment ID |
| `--env-type` | `empty` | Oracle type: `empty`, `doorkey`, `fetch`, `gotodoor`, `gotoobject`, `multiroom` |
| `--oracle-cost` | `0.0` | Reward penalty per oracle query |
| `--no-oracle` | `False` | Disable oracle — pure PPO baseline |
| `--reward-shaping` | `False` | Intermediate rewards for key/door events (DoorKey only) |
| `--total-timesteps` | `500_000` | Total environment steps |
| `--n-envs` | `8` | Parallel environments |
| `--lr` | `2.5e-4` | Learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--clip-coef` | `0.2` | PPO clip coefficient |
| `--ent-coef` | `0.01` | Entropy coefficient |
| `--bc-coef` | `0.0` | Behavioural cloning loss weight on guided steps |
| `--hidden-dim` | `256` | FC layer size after CNN |
| `--large-model` | `False` | Use `model_large.py` (required for 16×16 full obs) |
| `--partial-obs` | `False` | Use partial obs (7×7 FOV, 56×56px) |
| `--save-model` | `False` | Save best checkpoint across seeds |
| `--exp-name` | `oracle_ppo` | Prefix for log CSV and checkpoint filenames |

**Output:** `logs/{exp_name}__{env_id}__seed{seed}__{timestamp}.csv`

---

### `env_wrapper.py` — Oracle environment wrapper

Wraps any MiniGrid environment with:
- `+1` extra action: `query_oracle`
- Full or partial observability (RGB image)
- Oracle dispatching: `{empty, doorkey}` → `oracle.py`, `{fetch, gotodoor, gotoobject, multiroom}` → `oracle_transfer.py`

**Observability modes:**
- `partial_obs=False`: `FullyObsWrapper` + `RGBImgObsWrapper` → full grid, shape `(H×tile, W×tile, 3)`
- `partial_obs=True`: `RGBImgPartialObsWrapper` → agent-centric 7×7 FOV, always `(56, 56, 3)`

---

### `oracle.py` — BFS oracle for DoorKey / Empty

- **`bfs_empty(env)`** — State: `(x, y, dir)`.
- **`bfs_doorkey(env)`** — State: `(x, y, dir, has_key, door_open)`.
- Returns `(action, path_length)` tuple. `path_length` is used to compute trajectory efficiency.

---

### `oracle_transfer.py` — BFS oracle for transfer environments

Adapted BFS for four MiniGrid tasks. Mission text is parsed to identify the target object.

- **`bfs_fetch(env)`** — Pick up the target object.
- **`bfs_gotodoor(env)`** — Reach target door (Chebyshev distance ≤ 1).
- **`bfs_gotoobject(env)`** — Reach target object (Chebyshev distance ≤ 1).
- **`bfs_multiroom(env)`** — Navigate through multiple rooms (state includes open doors as a frozenset).
- **`get_oracle_action(env, env_type)`** — Entry point for all transfer envs.
- **`get_optimal_steps(env, env_type)`** — Returns BFS path length (for trajectory efficiency).

---

### `eval.py` — Inference + GIF generation

Loads a checkpoint, runs N episodes, saves a GIF with action labels and guided/oracle annotations.

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` file |
| `--env-id` | `MiniGrid-DoorKey-8x8-v0` | Evaluation environment |
| `--env-type` | `doorkey` | Oracle type |
| `--no-oracle` | `False` | For baseline checkpoints |
| `--stochastic` | `False` | Sample from policy distribution (default: argmax) |
| `--partial-obs` | `False` | Must match training setting |
| `--n-episodes` | `3` | Number of episodes |
| `--out` | `figures/eval.gif` | Output path |

---

### `eval_transfer_stats.py` — Zero-shot transfer evaluation

Runs N episodes of a trained model on an unseen task (no finetuning) and reports:
- Success rate, mean return, mean steps
- Oracle query rate (% of steps)
- **Trajectory efficiency** = `opt_steps / agent_steps` (BFS-computed at episode start, success episodes only)

Supports multi-run comparison: append results to a shared CSV then plot a 2×2 comparison figure (success rate, return, oracle %, efficiency).

**Key parameters:**

| Argument | Description |
|---|---|
| `--checkpoint` | Model to evaluate |
| `--env-id` | Transfer target environment |
| `--env-type` | `fetch`, `gotodoor`, `gotoobject`, `multiroom` |
| `--partial-obs` | Partial obs model |
| `--n-episodes` | Number of episodes (default: 100) |
| `--csv-out` | Append to comparison CSV |
| `--out` | Output PNG path |

---

### `merged_plot.py` — Merged overlay plot

All conditions overlaid on the same axes. **Horizontal layout** (1×4 subplots): return, success rate, oracle usage %, trajectory efficiency. Uses a **plasma colormap** for oracle costs + dashed black line for baseline.

**Key parameters:**

| Argument | Description |
|---|---|
| `--tag` | Experiment tag, e.g. `16_partial` |
| `--out` | Output PNG path |
| `--min-steps` | Ignore incomplete runs |

---

### `per_cost_plot.py` — Per-cost training curves

Generates one 1×4 plot per oracle cost: return, success rate, oracle usage %, queries per episode. Baseline is overlaid as a dashed red line on each plot. Also generates a standalone baseline plot.

```bash
python per_cost_plot.py --log-dir logs --out-dir figures/per_cost
```

---

### `compare_plot.py` — Multi-condition comparison

One row per oracle condition, four metric columns. Loads CSVs by glob pattern, deduplicates by seed (latest timestamp wins).

---

## Experiments conducted

### DoorKey-16×16 training sweep

**Full observability** — 5 conditions × 5 seeds × 2M steps:
`oracle_free`, `oracle_paid_{001,002,003,004,005}`, `baseline`

**Partial observability** — same conditions + fine-grained sweep:
- Coarse: `oracle_paid_{001,002,003,004,005}`
- Fine: `oracle_paid_{007,008,009,011,012}` (costs 0.007–0.012)
- Extended: `oracle_paid_{015,018}`

**Key finding:** Oracle usage drops sharply between cost=0.010 and cost=0.012. Below this threshold the agent queries freely; above it, it stops querying entirely. The transition is unstable (seed-dependent) in the 0.007–0.010 range.

---

### Zero-shot transfer

DoorKey-16×16 partial obs models evaluated (100 episodes, no finetuning) on:

| Target env | Notes |
|---|---|
| `MiniGrid-Fetch-16x16-N3-v0` | Pick up target colored object |
| `MiniGrid-MultiRoom-N6-v0` | Navigate through 6 rooms |

Models tested: `oracle_paid_001_16_partial`, `oracle_free_16_partial`, `baseline_16_partial`.

**Fetch results:** Oracle-trained models transfer well. Success detection requires `terminated=True AND return > 0` (FetchEnv terminates on any pickup, not just target).

**MultiRoom results:** Oracle model shows meaningful oracle usage on transfer — queries at room transitions.

**Trajectory efficiency** = BFS-optimal / actual steps. Computed only on successful episodes.

---

## Running experiments

### Quick local test

```bash
python train.py \
    --env-id MiniGrid-DoorKey-8x8-v0 --env-type doorkey \
    --oracle-cost 0.01 --reward-shaping \
    --total-timesteps 100000 --seed 1 --exp-name test_paid
```

### DoorKey-16×16 partial obs

```bash
python train.py \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --oracle-cost 0.01 --reward-shaping --large-model --partial-obs \
    --total-timesteps 2000000 --seed 4 --exp-name oracle_paid_001_16_partial --save-model
```

### On cluster (SLURM)

```bash
cd ~/Upper_Bound
sbatch submit_16x16.sh                    # full sweep
sbatch submit_16x16_partial_vfine.sh      # fine-grained costs
```

### Zero-shot transfer evaluation

```bash
python eval_transfer_stats.py \
    --checkpoint checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id MiniGrid-Fetch-16x16-N3-v0 --env-type fetch \
    --partial-obs --n-episodes 100 --label "Oracle 0.01" \
    --csv-out figures/fetch_comparison.csv \
    --out figures/fetch_comparison.png
```

### GIF generation

```bash
# All partial-obs models on DoorKey-16x16
sbatch submit_eval_all_partial_doorkey.sh   # outputs to all_gif/

# Single model
python eval.py \
    --checkpoint checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --partial-obs --stochastic --n-episodes 3 --seed 42 \
    --out all_gif/doorkey16_oracle_paid001_partial.gif
```

### Per-cost plots

```bash
python per_cost_plot.py --log-dir logs --out-dir figures/per_cost
```

---

## Syncing with cluster

```bash
# Push scripts
rsync -avz ~/Desktop/cs503_project/Upper_Bound/*.py ~/Desktop/cs503_project/Upper_Bound/*.sh \
    tjouven@izar.epfl.ch:~/Upper_Bound/

# Pull logs
rsync -avz tjouven@izar.epfl.ch:~/Upper_Bound/logs/ ~/Desktop/cs503_project/minigrid/logs/

# Pull figures
rsync -avz tjouven@izar.epfl.ch:~/Upper_Bound/figures/ ~/Desktop/cs503_project/Upper_Bound/figures/

# Pull checkpoints
rsync -avz "tjouven@izar.epfl.ch:~/Upper_Bound/checkpoints/best__*_partial__*.pt" \
    ~/Desktop/cs503_project/minigrid/checkpoints/
```

---

## VLM oracle

### Files

```
vlm_oracle.py        # Replaces BFS with a VLM (Qwen2-VL, InternVL, SmolVLM, ...)
download_models.py   # Download VLMs from HuggingFace to cluster scratch
job_vlm.sh           # SLURM job for VLM training runs
submit_vlm_test.sh   # Quick test job
```

### Available models

| Key | Model |
|---|---|
| `qwen2vl` | Qwen2-VL 7B |
| `qwen3b` | Qwen2.5-VL 3B |
| `qwen7b` | Qwen2.5-VL 7B |
| `internvl` | InternVL2.5 4B |
| `internvl3` | InternVL3 8B |
| `wethink` | WeThink-Qwen2.5VL 7B |
| `smolvlm` | SmolVLM 2B |

### Usage

```bash
# Download models (login node, once)
python download_models.py --cache_dir /scratch/izar/tjouven/vlm_models

# Submit VLM training job
sbatch submit_vlm_test.sh
```

---

## Installation

```bash
conda activate nanofm
pip install -r requirements.txt
```
