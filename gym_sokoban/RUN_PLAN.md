# Gym Sokoban Run Plan

This checklist captures the current experiment plan for recreating the
MiniGrid-style experiment suite in `gym_sokoban`.

Core intent:

- Recreate the WorldCoder-style PPO Sokoban baseline with full sprites.
- Keep PPO purely reinforcement learning; do not use behavior cloning as an
  experimental ingredient.
- Use oracle-query runs as an upper-bound / optional-help comparison, not as
  the main baseline.
- Prefer Sokoban-to-Sokoban transfer before MiniGrid-to-Sokoban transfer.
- Team experiment ladder:
  - small Sokoban
  - big Sokoban
  - Sokoban with VLM oracle
  - randomized-accuracy oracle
  - partial observability with oracle, VLM, and randomized-accuracy oracle
  - transfer to new Sokoban tasks

## Canonical Setup

- [ ] Environment: `Sokoban-small-v0`
- [ ] Observation: full 7x7 board rendered at 8 px/cell, shape `56x56x3`
- [ ] CNN: WorldCoder-style conv stack adapted to full sprites
  - [ ] `Conv2d(3, 16, kernel=2, stride=1)`
  - [ ] `Conv2d(16, 32, kernel=2, stride=1)`
  - [ ] `Conv2d(32, 64, kernel=2, stride=1)`
  - [ ] `AdaptiveAvgPool2d(4, 4)`
  - [ ] `Linear(1024, 64)`
- [ ] PPO defaults
  - [ ] `lr=3e-4`
  - [ ] `gamma=0.99`
  - [ ] `gae_lambda=0.95`
  - [ ] `clip_coef=0.2`
  - [ ] `ent_coef=0.0`
  - [ ] `vf_coef=0.5`
  - [ ] `n_epochs=10`
  - [ ] `max_episode_steps=50`
- [ ] Current preferred batch setup
  - [ ] `n_envs=64`
  - [ ] `n_steps=256`
  - [ ] `n_minibatches=8`
  - [ ] rollout batch `64 * 256 = 16,384`
  - [ ] optimizer minibatch `16,384 / 8 = 2,048`
- [ ] Reward shaping off for WorldCoder-style baseline unless explicitly
  running a shaping ablation.

## Phase 0: Reproducibility Infrastructure

- [ ] Each run writes to its own directory under `runs/`
- [ ] Run directory name starts with timestamp and includes run description
- [ ] Run directory contains:
  - [ ] `config.yaml`
  - [ ] `logs/stdout.log`
  - [ ] `data/metrics.csv`
  - [ ] `figures/training_metrics.png`
  - [ ] `checkpoints/final.pt` when `--save-model` is passed
- [ ] W&B logs smoothed episode metrics, not only raw per-episode spikes
- [ ] Generated PNG is uploaded to W&B at the end of the run
- [ ] Success metric is state-based: every box must be on a target
- [ ] Do not mix old logs whose filenames say `baseline` but have nonzero
  `guided_pct` or `queries_per_ep`

## Phase 1: PPO Baseline Reproduction

Goal: establish a clean no-oracle PPO baseline on `Sokoban-small-v0`.

- [ ] Smoke run
  - [ ] seed `1`
  - [ ] `--no-oracle`
  - [ ] `total_timesteps=4096`
- [ ] Speed sanity runs
  - [ ] seed `1`
  - [ ] `--no-oracle`
  - [ ] compare `n_envs in {8, 16, 32, 64}`
  - [ ] choose setting by SPS and stability
- [ ] Main baseline run
  - [ ] seed `1`
  - [ ] `--no-oracle`
  - [ ] `total_timesteps=2_000_000`
  - [ ] `n_envs=64`
  - [ ] `n_steps=256`
  - [ ] `n_minibatches=8`
- [ ] Multi-seed baseline
  - [ ] seeds `1, 2, 3`
  - [ ] same settings as main baseline
- [ ] Final baseline sweep
  - [ ] seeds `1, 2, 3, 4, 5`
  - [ ] same settings as main baseline

Fast 8-hour target:

- [ ] `Sokoban-small-v0`, baseline, seed `1`, `500k`
- [ ] `Sokoban-small-v0`, oracle-free, seed `1`, `500k`
- [ ] `Sokoban-small-v0`, randomized oracle, seed `1`, `500k`
- [ ] `Sokoban-small-v0`, partial-observation oracle-free, seed `1`, `500k`

Primary metrics:

- [ ] episodic return
- [ ] success rate
- [ ] episode length
- [ ] value loss / explained variance
- [ ] entropy
- [ ] SPS

## Phase 2: Oracle Upper-Bound Runs

Goal: recreate the MiniGrid oracle-cost sweep for Sokoban.

Conditions:

- [ ] `baseline`: `--no-oracle`
- [ ] `oracle_free`: `--oracle-cost 0.0`
- [ ] `oracle_paid_001`: `--oracle-cost 0.01`
- [ ] `oracle_paid_005`: `--oracle-cost 0.05`
- [ ] `oracle_paid_010`: `--oracle-cost 0.10`
- [ ] `oracle_paid_020`: `--oracle-cost 0.20`

Run order:

- [ ] Cheap triad first
  - [ ] `baseline`
  - [ ] `oracle_free`
  - [ ] `oracle_paid_005`
  - [ ] seeds `1, 2, 3`
  - [ ] `500k` or `1M` timesteps
- [ ] Full cost sweep
  - [ ] all conditions above
  - [ ] seeds `1, 2, 3`
  - [ ] `2M` timesteps
- [ ] Final cost sweep if needed
  - [ ] all conditions above
  - [ ] seeds `1, 2, 3, 4, 5`
  - [ ] `2M` timesteps

Important interpretation:

- [ ] Without behavior cloning, oracle runs test whether the agent learns that
  querying is useful.
- [ ] They do not necessarily test whether the agent learns to imitate the
  oracle's native Sokoban actions.
- [ ] Track `guided_pct` and `queries_per_ep` as first-class results.

## Phase 3: Randomized-Accuracy Oracle

Goal: simulate imperfect VLM-style advice without paying VLM cost or adding
model-serving instability.

Implementation target:

- [x] Add `--oracle-accuracy <float>` to Sokoban training/wrapper
- [x] `1.0` means perfect BFS oracle
- [x] `0.0` means always random native action when oracle is queried
- [x] Intermediate values return BFS action with probability `p`, otherwise a
  random valid native Sokoban action
- [x] Log `oracle_accuracy` in `config.yaml` and W&B config
- [x] Log `oracle_correct_rate` in CSV and W&B
- [x] Keep `guided=True` whenever the query action was used, even if the oracle
  returned a noisy action

Initial conditions:

- [ ] `oracle_acc_100`: accuracy `1.0`, cost `0.0`
- [ ] `oracle_acc_075`: accuracy `0.75`, cost `0.0`
- [ ] `oracle_acc_050`: accuracy `0.50`, cost `0.0`
- [ ] `oracle_acc_025`: accuracy `0.25`, cost `0.0`
- [ ] `oracle_acc_000`: accuracy `0.0`, cost `0.0`

Run order:

- [ ] First pass: accuracies `1.0`, `0.5`, `0.0`; seed `1`; `500k`
- [ ] Full pass: all accuracies above; seeds `1, 2, 3`; `500k`

## Phase 4: Big Sokoban

Goal: scale from small Sokoban to larger layouts after the small baseline and
oracle runs are interpretable.

Candidate environments:

- [ ] `Sokoban-v0`: 10x10, 3 boxes
- [ ] `Sokoban-large-v0`: 13x11, 3 boxes
- [ ] `Sokoban-huge-v0`: 13x13, 5 boxes

First-pass big Sokoban conditions:

- [ ] baseline, no oracle
- [ ] oracle-free
- [ ] randomized oracle accuracy `0.5`

Required setup before big runs:

- [ ] confirm observation interface for larger boards
- [ ] choose full-board observation vs 7x7 crop
- [ ] benchmark reset speed
- [ ] benchmark BFS oracle speed
- [ ] decide whether huge maps are feasible with BFS online

Recommended order:

- [ ] `Sokoban-v0` first
- [ ] `Sokoban-large-v0` second
- [ ] `Sokoban-huge-v0` only after BFS/reset speed is acceptable

## Phase 5: VLM Oracle

Goal: replace BFS oracle with a VLM-like action provider once BFS upper-bound
and randomized-accuracy oracle runs are working.

First VLM target:

- [ ] `Sokoban-small-v0`
- [ ] one VLM model only
- [ ] seed `1`
- [ ] `500k` max
- [ ] compare against BFS oracle and randomized-accuracy oracle

Conditions:

- [ ] VLM oracle, cost `0.0`
- [ ] VLM oracle, cost `0.1`
- [ ] BFS oracle, same costs
- [ ] randomized oracle, matched estimated VLM accuracy

Required setup:

- [ ] add VLM adapter for Sokoban observations
- [ ] define prompt/action parser for 9 native Sokoban actions
- [ ] log parse failures
- [ ] log action agreement with BFS when BFS is available
- [ ] support local/offline model serving
- [ ] include VLM model key, mode, server URL, and prompt template in
  `config.yaml`

## Phase 6: Partial Observability

Goal: make the harder/transfer setting use a fixed visual interface.

Canonical partial observation:

- [ ] 7x7 agent-centered crop
- [ ] 8 px/cell
- [ ] input shape `56x56x3`
- [ ] wall padding outside map bounds
- [ ] same CNN for small and big Sokoban

Partial-observation conditions:

- [ ] baseline partial obs
- [ ] BFS oracle partial obs
- [ ] randomized-accuracy oracle partial obs
- [ ] VLM oracle partial obs

Run order:

- [ ] `Sokoban-small-v0`, partial baseline, seed `1`, `500k`
- [ ] `Sokoban-small-v0`, partial oracle-free, seed `1`, `500k`
- [ ] `Sokoban-small-v0`, partial randomized oracle `0.5`, seed `1`, `500k`
- [ ] VLM partial only after full-observation VLM works

## Phase 7: Reward Shaping Ablation

Goal: determine whether performance comes from PPO/oracle help or dense shaping.

Conditions:

- [ ] baseline, unshaped
- [ ] baseline, shaped
- [ ] oracle_free, unshaped
- [ ] oracle_free, shaped
- [ ] oracle_paid_005, unshaped
- [ ] oracle_paid_005, shaped

Settings:

- [ ] seeds `1, 2, 3`
- [ ] `1M` or `2M` timesteps
- [ ] same PPO/batch settings as Phase 1

This is lower priority than the team ladder above unless performance is too
weak to interpret without shaping.

## Phase 8: Sokoban-to-Sokoban Transfer

Goal: transfer within the same domain before attempting MiniGrid-to-Sokoban.

Canonical visual interface:

- [ ] fixed 7x7 agent-centered crop
- [ ] 8 px/cell
- [ ] input shape `56x56x3`
- [ ] wall padding outside map bounds
- [ ] same action space across all tasks

Task ladder:

- [ ] `Sokoban-small-v0`: 7x7, 2 boxes
- [ ] `Sokoban-small-v1`: 7x7, 3 boxes
- [ ] `Sokoban-v0`: 10x10, 3 boxes
- [ ] `Sokoban-large-v0`: 13x11, 3 boxes
- [ ] `Sokoban-huge-v0`: 13x13, 5 boxes

Transfer comparisons:

- [ ] scratch PPO
- [ ] transferred CNN encoder
- [ ] frozen transferred CNN encoder
- [ ] transferred full policy/value network
- [ ] fine-tuned full network

Do this only after Phase 1/2 results are interpretable.

## Plotting / Reporting Checklist

- [ ] Do not include smoke runs in final plots
- [ ] Do not include runs below the planned timestep threshold
- [ ] Do not include old mislabeled baseline logs with nonzero oracle queries
- [ ] Plot mean and variation across seeds
- [ ] Report:
  - [ ] return vs environment steps
  - [ ] success rate vs environment steps
  - [ ] guided percentage vs environment steps
  - [ ] queries per episode vs environment steps
  - [ ] return vs cumulative oracle calls
- [ ] Keep unshaped and shaped runs in separate figures
