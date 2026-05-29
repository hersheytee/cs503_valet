# Gym Sokoban Experiments

## Current Goal

Recreate a WorldCoder-style PPO baseline on `gym-sokoban`, then use that as the foundation for Sokoban-to-Sokoban transfer experiments.

We are not using behavior cloning. Oracle-query experiments may still exist later as an upper-bound comparison, but the WorldCoder baseline is plain PPO with `--no-oracle`.

## Baseline Setup

Environment:

```text
Sokoban-small-v0
```

Observation:

```text
56x56x3 RGB
```

This corresponds to a full 7x7 Sokoban board rendered at 8 pixels per square.

Training defaults in `train.py` are currently WorldCoder-style:

```text
obs-size: 56
max episode steps: 50
hidden dim: 64
lr: 3e-4
entropy coef: 0.0
n-steps: 256
n-epochs: 10
behavior cloning: disabled/removed
W&B tracking: enabled by default
```

The important command-line flag for the pure PPO baseline is:

```text
--no-oracle
```

For oracle-query experiments, oracle quality can be randomized:

```text
--oracle-accuracy 1.0   # perfect BFS oracle
--oracle-accuracy 0.5   # 50% BFS-optimal, 50% deliberately non-optimal native Sokoban action
```

This only affects steps where the agent chooses `query_oracle`. The queried
step is still logged as guided, and `oracle_correct_rate` records how often the
returned oracle action matched the BFS-optimal action.

Oracle query cost can also be linearly scheduled:

```text
--oracle-cost 0.0 --oracle-cost-final 1.0
```

This starts query cost at `0.0` and linearly increases it to `1.0` over
training. The active cost is logged as `oracle_cost` in the metrics CSV and in
W&B.

## Current CNN

The current `model.py` implements the fair WorldCoder full-sprite CNN:

```text
Input: 56x56x3 RGB

Conv2d(3, 16, kernel=2, stride=1)
ReLU
Conv2d(16, 32, kernel=2, stride=1)
ReLU
Conv2d(32, 64, kernel=2, stride=1)
ReLU
AdaptiveAvgPool2d(4, 4)
Flatten 1024
Linear(1024, 64)
ReLU
policy_head
value_head
```

WorldCoder used compact 7x7 grid observations. Applying the same conv stack directly to 56x56 sprites would create a huge feature map, so we pool back to the original 4x4 spatial footprint.

See `ARCHITECTURES.md` for architecture comparisons.

## PPO Batch Terms

`n-envs` is the number of parallel environments.

`n-steps` is how many steps each environment collects before PPO updates.

Rollout batch size:

```text
batch_size = n_envs * n_steps
```

PPO optimizer minibatch size:

```text
minibatch_size = batch_size / n_minibatches
```

For the current preferred Sokoban run:

```text
n-envs: 64
n-steps: 256
n-minibatches: 8

rollout batch: 64 * 256 = 16,384
optimizer minibatch: 16,384 / 8 = 2,048
```

This keeps the optimizer minibatch close to WorldCoder's reported PPO batch size.

## Izar Run Commands

Install/login W&B once:

```bash
conda activate cs503_proj
pip install wandb
wandb login
```

Run a quick syntax check:

```bash
python -m py_compile gym_sokoban/train.py gym_sokoban/env_wrapper.py gym_sokoban/model.py
```

Run a tiny smoke test:

```bash
python gym_sokoban/train.py \
  --no-oracle \
  --exp-name worldcoder_smoke \
  --total-timesteps 4096 \
  --n-envs 4 \
  --n-steps 64
```

Submit the current recommended 2M one-seed run:

```bash
sbatch \
  --job-name="soko_wc_2m_wandb_s1" \
  --export=ALL,ENV_ID=Sokoban-small-v0,EXP_NAME=worldcoder_baseline_2m_64env,SEED=1,TOTAL_TIMESTEPS=2000000,EXTRA_ARGS="--no-oracle --n-envs 64 --n-steps 256 --n-minibatches 8 --wandb-project cs503-sokoban --wandb-group worldcoder_2m" \
  gym_sokoban/job.sh
```

Disable W&B if needed:

```text
--no-track
```

Use offline W&B if Izar networking is unreliable:

```text
--wandb-mode offline
```

Sync offline runs later:

```bash
wandb sync wandb/offline-run-*
```

## Monitoring Jobs

Queue:

```bash
squeue -u $USER
```

Wall time and state:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,MaxRSS
```

Logs:

```bash
tail -f logs/slurm_<JOBID>.out
tail -f logs/slurm_<JOBID>.err
```

Search for SPS:

```bash
grep "sps=" logs/slurm_*.out | tail -40
```

Cancel a job:

```bash
scancel <JOBID>
```

Cancel all owned jobs:

```bash
scancel -u $USER
```

## Current Speed Notes

The 256-env attempt was too slow/noisy because `gym-sokoban` room generation repeatedly retries during reset. With `SyncVectorEnv`, many environments are not truly parallel in the useful sense.

The 64-env speed check gave around 147 SPS, which implies:

```text
2,000,000 / 147 ~= 3.8 hours
```

So 64 envs is currently a reasonable baseline setting.

## Why Not 256 Envs For Now

With 256 envs and 256 steps:

```text
rollout batch = 256 * 256 = 65,536
```

This delays the first progress print and makes reset generation painful. It also requires setting `--n-minibatches 32` to keep the optimizer minibatch at 2048.

For now, prefer:

```text
64 envs, 256 steps, 8 minibatches
```

## Run Artifact Layout

New `train.py` runs are self-contained under `runs/`.

Each run directory is named with the timestamp first:

```text
runs/YYYYMMDD_HHMMSS__<exp-name>__<env-id>__seed<seed>/
```

Expected contents:

```text
config.yaml
logs/stdout.log
data/metrics.csv
figures/training_metrics.png
checkpoints/final.pt
```

`config.yaml` records CLI args, derived PPO batch sizes, observation/action shapes, parameter count, device, git commit, and artifact paths. `data/metrics.csv` is the canonical file for analysis and figure regeneration.

or:

```text
32 envs, 256 steps, 4 minibatches
```

## Future Improvement: Pre-Generated Rooms

`gym-sokoban` spends a lot of wall-clock time generating random rooms, especially when many envs reset. A pre-generated room set could speed training and improve reproducibility.

Planned idea:

1. Generate a large set of valid `Sokoban-small-v0` rooms once.
2. Save `room_fixed` and `room_state` arrays to an `.npz`.
3. Add a wrapper that samples from this dataset at reset.
4. Use train/validation/test room splits.

This would support clean transfer experiments:

```text
train on small room set
evaluate on held-out small rooms
fine-tune/transfer to larger Sokoban boards
```

## Transfer Direction

The current preferred transfer story is Sokoban-to-Sokoban, not MiniGrid-to-Sokoban.

Reason:

- same action space
- same object semantics
- same reward structure
- same 8-pixels-per-cell visual interface

Later transfer experiments can compare:

```text
scratch WorldCoder CNN
transferred WorldCoder CNN trunk
frozen transferred trunk plus new heads
previous MiniGrid-style CNN on 56x56 sprites
```

For larger boards, use a 7x7 player-centered crop rendered at 8 pixels per cell so the input remains `56x56x3`.

## Budget-Limited Oracle with Budget-Aware Observation

**Note for report.**

We introduce a fixed per-episode oracle query budget (e.g. 1, 3, or 5 queries).
Rather than silently redirecting exhausted queries to a no-op — which would give the
agent no signal that its budget is gone — we encode the remaining budget directly
into the observation as a fourth image channel.

The 4th channel is a spatially-constant uint8 field:

```text
budget_channel[h, w] = int((queries_remaining / max_queries) * 255)
```

So the channel reads 255 (white) at the start of an episode and decays to 0 (black)
as the agent uses up its queries. After the budget is exhausted the channel stays at
0 for the rest of the episode.

This means the observation space becomes `56x56x4` instead of `56x56x3` for
budget-limited runs. The CNN first layer reads 4 input channels and the policy can
explicitly condition on remaining budget when deciding whether to query.

No architectural changes to `CNNPolicy` are needed: the model already
parameterises input channels from `obs_shape`, which flows through automatically
from `envs.single_observation_space.shape`. Budget and non-budget runs therefore
produce non-interchangeable checkpoints (3-channel vs 4-channel weights), which is
intentional — they are distinct experimental conditions.

Implementation:
- `--max-oracle-queries N` in `train.py`
- Budget tracking and 4th-channel encoding in `SokobanOracleWrapper._process_obs()`
- When budget is exhausted the `QUERY_ACTION` redirects to a no-op action (0),
  but the agent can anticipate this by reading the 4th channel before choosing.
