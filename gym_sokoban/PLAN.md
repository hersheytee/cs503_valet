# Sokoban Oracle/VLM PPO Plan

## Goal

Test whether optional expert help improves PPO on RGB Sokoban.

The experiment should be scoped to Sokoban first, without MiniGrid-to-Sokoban transfer. The agent and future VLM oracle should both see RGB observations. The RL policy should use the same CNN family as the 16x16 MiniGrid runs.

## Target Setup

- Environment: `Sokoban-small-v0`
- Observation: RGB frame resized to `128x128x3`
- Policy architecture: MiniGrid `model_large.py` style CNN
- Baseline action space: 9 native Sokoban actions
- Oracle action space: 10 actions, where action 9 is `query_oracle`
- Initial oracle backend: BFS oracle
- Later oracle backend: VLM oracle using the same RGB frame

## Model Choice

Use the MiniGrid large CNN architecture:

```text
Conv2d(C, 32, kernel=3, stride=2, padding=1)
ReLU
Conv2d(32, 64, kernel=3, stride=2, padding=1)
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)
ReLU
AdaptiveAvgPool2d((8, 8))
Flatten
Linear(4096, hidden_dim=256)
ReLU
policy_head
value_head
```

This matches the 16x16 MiniGrid visual policy family and handles `128x128` RGB Sokoban frames cleanly.

## Milestones

### 1. Clean Sokoban RGB PPO

- Change wrapper output from `84x84x3` to `128x128x3`.
- Replace the current Sokoban Nature CNN with the MiniGrid large CNN interface:
  - `CNNPolicy(obs_shape, n_actions, hidden_dim=256)`
- Update training/eval/tests to pass `obs_shape`.
- Make `hidden_dim=256` the default for consistency with MiniGrid.
- Add an explicit episode time limit/horizon for Sokoban, likely `50` or `120`.
- Log success from environment termination/final reward info, not only `ep_return > 0`.
- Make reward shaping an explicit experiment setting. Do not silently mix shaped and unshaped baselines.

### 2. Validate the BFS Oracle

- Run the raw BFS benchmark on `Sokoban-small-v0`.
- Run the wrapper test where the agent repeatedly chooses `query_oracle`.
- Confirm:
  - oracle solves most generated maps
  - `guided=True` appears only for oracle queries
  - `oracle_action` is in `0..8`
  - costs are subtracted only on guided steps

### 3. Run Core PPO Conditions

Start with a small smoke test, then scale to full runs.

Recommended initial conditions:

```text
baseline: no oracle
oracle_free: cost 0.00
oracle_paid_001: cost 0.01
oracle_paid_005: cost 0.05
oracle_paid_010: cost 0.10
oracle_paid_020: cost 0.20
```

Use at least 3 seeds for final plots. Use 1 seed for debugging.

### 4. Add VLM Oracle Backend

After BFS experiments are stable:

- Add `--oracle-kind bfs|vlm`.
- Keep `query_oracle` action unchanged.
- Feed the VLM the RGB frame from the same environment state.
- Parse VLM output into one of the 9 native Sokoban actions.
- Log VLM-specific metrics:
  - invalid responses
  - parse failures
  - latency
  - agreement with BFS when both are available

### 5. Plot and Compare

Use the same metrics as MiniGrid where possible:

- episode return
- success rate
- oracle guidance percentage
- queries per episode
- cumulative oracle calls
- return vs cumulative oracle calls
- agreement rate

For VLM runs, also plot invalid response rate and average query latency.

## Script Audit

### `train.py`

Status: usable core, needs changes.

What works:

- CleanRL-style PPO loop is already present.
- Supports vector envs, oracle action, BC loss, CSV logging, and model saving.
- Tracks useful metrics such as `guided_pct`, `queries_per_ep`, `cum_queries`, and `agreement_rate`.

Needed changes:

- Update docstring from `84x84` to `128x128`.
- Change model init to `CNNPolicy(obs_shape=obs_shape, n_actions=n_actions, hidden_dim=args.hidden_dim)`.
- Change default `hidden_dim` from `512` to `256`.
- Add a `--max-episode-steps` or wrapper-side time limit.
- Revisit `success = float(ret > 0)`. For shaped rewards this can be misleading.
- Add PPO diagnostics: approximate KL, clip fraction, explained variance.
- Decide whether BC should be enabled for oracle-guided steps; default is currently `0.0`.
- Avoid `os.system("python plot.py ...")` if running from repo root unless `plot.py` path is made explicit.

### `env_wrapper.py`

Status: usable foundation, needs important changes.

What works:

- Bridges old `gym` Sokoban to Gymnasium-style API.
- Adds `query_oracle`.
- Converts observations to RGB arrays.
- Applies oracle cost and optional reward shaping.

Needed changes:

- Resize observations to `128x128`, not `84x84`.
- Update observation-space comments and docstrings.
- Add max episode steps and return `truncated=True` when reached.
- Consider returning `success` in `info` when the puzzle is solved.
- Keep `render()` returning the full-resolution RGB frame for GIFs/VLM debugging.
- Later: add oracle backend selection for BFS vs VLM.

### `model.py`

Status: replace.

Current file is a Nature CNN for `84x84` RGB with `hidden_dim=512`.

Needed changes:

- Replace with the MiniGrid large CNN architecture.
- Use dynamic CNN output sizing with `obs_shape`.
- Default `hidden_dim=256`.
- Keep the same methods:
  - `forward`
  - `get_action_and_value`
  - `get_value`

### `bfs_oracle.py`

Status: usable BFS oracle, keep.

What works:

- Extracts Sokoban state from `room_state` and `room_fixed`.
- Computes BFS action sequence.
- Caches the planned path until the state deviates.
- Returns native Sokoban actions `0..8`.

Needed changes:

- Rename comments/docstrings that say A* if any caller mentions A*.
- Add stronger tests around action mapping and solved/deadlock behavior.
- For larger Sokoban later, expect BFS to become expensive.

### `benchmark_oracle.py`

Status: useful sanity benchmark, minor changes.

What works:

- Measures BFS solve time and failure count on generated maps.

Needed changes:

- Add CLI args for env id and number of episodes.
- Print median and p95 solve time, not only mean/max.
- Optionally save CSV results.

### `test_oracle.py`

Status: useful manual visual test, needs cleanup.

What works:

- Lets the raw BFS oracle play `Sokoban-small-v0`.
- Useful for eyeballing action quality.

Needed changes:

- Docstring says A* but implementation is BFS.
- Avoid mandatory live matplotlib window for headless machines.
- Add `--seed`, `--env-id`, and `--max-steps`.

### `test_wrapper.py`

Status: useful wrapper smoke test, needs update.

What works:

- Exercises `SokobanOracleWrapper` and repeated `query_oracle`.
- Saves a GIF.

Needed changes:

- Update comments from `84x84` to `128x128`.
- Assert observation shape is `(128, 128, 3)`.
- Assert action space is 10 with oracle and 9 without oracle.
- Add a no-oracle wrapper smoke test.

### `test_model.py`

Status: needs update after model replacement.

Current test assumes:

- `CNNPolicy(n_actions=...)`
- `84x84x3`
- old Nature CNN

Needed changes:

- Instantiate `CNNPolicy(obs_shape=(128, 128, 3), n_actions=10, hidden_dim=256)`.
- Test batched input only unless single-frame support is deliberately kept.
- Add a baseline-action test with `n_actions=9`.

### `eval_baseline.py`

Status: useful idea, needs rewrite to match new model.

What works:

- Loads a checkpoint.
- Runs deterministic greedy evaluation.
- Saves GIFs.

Needed changes:

- Rename to `eval.py` or support both baseline and oracle checkpoints.
- Instantiate model with `obs_shape`.
- Default hidden dim should match training.
- Use wrapper success/truncation info, not `total_reward > 0`.
- Support oracle action space if evaluating an oracle-trained checkpoint.
- Add `--oracle-enabled` or infer action count from checkpoint/args.

### `job.sh`

Status: usable SLURM single-run wrapper, needs parameter cleanup.

What works:

- Activates conda env.
- Runs `gym_sokoban/train.py`.
- Uses exported `EXP_NAME`, `SEED`, and `EXTRA_ARGS`.

Needed changes:

- Add `ENV_ID` as an exported parameter with default `Sokoban-small-v0`.
- Increase wall time for real runs.
- Ensure logs/figures/checkpoints dirs exist before SLURM writes logs.
- Consider adding `python -u` for unbuffered logs.

### `launch.sh`

Status: usable concept, costs/names need cleanup.

What works:

- Submits separate SLURM jobs for oracle costs and baseline.

Needed changes:

- Current names do not match costs cleanly: `oracle_paid_02` uses cost `0.2`, etc.
- Use consistent cost names:
  - `oracle_paid_001` for `0.01`
  - `oracle_paid_005` for `0.05`
  - `oracle_paid_010` for `0.10`
  - `oracle_paid_020` for `0.20`
- Add seeds array with final seeds, e.g. `1 2 3`.
- Pass `ENV_ID`.

### `run_all_sequential.sh`

Status: usable local/sequential runner, needs update.

What works:

- Runs baseline and several oracle costs sequentially.
- Points to `gym_sokoban/train.py`.

Needed changes:

- Add `oracle_paid_010` and `oracle_paid_020`.
- Make Python path portable instead of hardcoding `venv/Scripts/python.exe`.
- Add smoke-test mode with fewer timesteps.
- Add `--save-model` only for final runs or when needed.

### `launch_sequential`

Status: stale/wrong for this folder.

This file is a MiniGrid SLURM script inside `gym_sokoban`. It sets:

- `ENV_ID="MiniGrid-DoorKey-8x8-v0"`
- `ENV_TYPE="doorkey"`
- calls `python -u train.py` from the submit directory

Recommendation:

- Do not use it for Sokoban.
- Replace it with a proper Sokoban sequential SLURM script or delete it.

## Immediate Next Patch

Recommended first code patch:

1. Update `env_wrapper.py` to output `128x128`.
2. Replace `model.py` with MiniGrid-large CNN.
3. Update `train.py` model initialization and default hidden dim.
4. Update `test_model.py` and `test_wrapper.py`.
5. Run smoke tests:

```bash
python gym_sokoban/test_model.py
python gym_sokoban/test_wrapper.py
python gym_sokoban/benchmark_oracle.py
python gym_sokoban/train.py --env-id Sokoban-small-v0 --total-timesteps 2048 --n-envs 2 --n-steps 64 --exp-name smoke_oracle
python gym_sokoban/train.py --env-id Sokoban-small-v0 --no-oracle --total-timesteps 2048 --n-envs 2 --n-steps 64 --exp-name smoke_baseline
```
