# Result Analysis and Plotting Handoff

This handoff explains how to organize results for the final presentation and
report. The main project story is VLM-guided RL in MiniGrid. Sokoban is a bonus
stress test / harder-domain extension.

## Project Story

Main question:

```text
Can an RL agent learn when to query a visual expert?
```

Core setup:

- PPO agent receives normal environment actions plus one extra `query` action.
- Querying an expert returns an action and may incur a reward cost.
- Expert types:
  - BFS oracle: perfect controlled upper bound.
  - VLM oracle: intended real visual expert.
  - Randomized oracle: controlled proxy for imperfect VLM reliability.
- Main metrics:
  - episodic return
  - success rate
  - guided percentage / oracle usage
  - queries per episode
  - oracle correctness / VLM accuracy where available

## Data Sources

### MiniGrid

MiniGrid is the main result source. Look in:

```text
minigrid/logs/
minigrid/figures/
```

Useful plotting scripts:

```text
minigrid/compare_plot.py
minigrid/merged_plot.py
```

Primary MiniGrid experiments:

- DoorKey 8x8:
  - baseline PPO
  - BFS oracle free
  - BFS oracle costs
  - VLM oracle if available
- DoorKey 16x16 / partial-observation runs are secondary unless clearly
  complete.

### Sokoban

Sokoban is bonus / stress test. Local artifacts are under:

```text
gym_sokoban/logs/
logs/
runs/
```

New runs create self-contained directories:

```text
runs/YYYYMMDD_HHMMSS__<description>/
  config.yaml
  logs/stdout.log
  data/metrics.csv
  figures/training_metrics.png
  checkpoints/final.pt
```

Important caveat:

- Some older `baseline` Sokoban logs are not real baselines because they have
  nonzero `guided_pct` / `queries_per_ep`.
- Treat old logs carefully. A true no-oracle baseline should have:

```text
guided_pct = 0
queries_per_ep = 0
```

### W&B

Project:

```text
hersheytee/cs503-sokoban
```

Useful groups:

```text
worldcoder_2m
sokoban_randomized_oracle_500k
sokoban_randomized_oracle_extra_500k
sokoban_linear_cost_500k
```

W&B likely contains metrics even when local Vast artifacts were not downloaded.
Policy checkpoints are only on W&B if the run used:

```text
--upload-checkpoint
```

## W&B Export

For a quick run summary:

```python
import wandb
import pandas as pd

ENTITY = "hersheytee"
PROJECT = "cs503-sokoban"
GROUPS = [
    "worldcoder_2m",
    "sokoban_randomized_oracle_500k",
    "sokoban_randomized_oracle_extra_500k",
    "sokoban_linear_cost_500k",
]

api = wandb.Api()
rows = []

for group in GROUPS:
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})
    for run in runs:
        cfg = run.config
        summary = run.summary._json_dict
        rows.append({
            "run_name": run.name,
            "group": group,
            "state": run.state,
            "env_id": cfg.get("env_id"),
            "seed": cfg.get("seed"),
            "no_oracle": cfg.get("no_oracle"),
            "oracle_cost": cfg.get("oracle_cost"),
            "oracle_cost_final": cfg.get("oracle_cost_final"),
            "oracle_accuracy": cfg.get("oracle_accuracy"),
            "total_timesteps": cfg.get("total_timesteps"),
            "final_success_rate": summary.get("episode/success_rate"),
            "final_return": summary.get("episode/return"),
            "final_guided_pct": summary.get("episode/guided_pct"),
            "final_queries_per_ep": summary.get("episode/queries_per_ep"),
            "final_oracle_correct_rate": summary.get("episode/oracle_correct_rate"),
        })

df = pd.DataFrame(rows)
df.to_csv("wandb_sokoban_summary.csv", index=False)
print(df)
```

For full curves, use `run.scan_history()` and export keys such as:

```text
charts/global_step
episode/return
episode/success_rate
episode/guided_pct
episode/queries_per_ep
episode/oracle_correct_rate
episode/oracle_cost
losses/value_loss
losses/policy_loss
losses/explained_variance
losses/entropy
```

## Analysis Rules

Use consistent x-axis:

```text
global_step
```

Use smoothed episode metrics when possible:

```text
episode/success_rate
episode/return
episode/guided_pct
episode/queries_per_ep
```

If only raw CSVs are available, compute rolling means:

- success rate: rolling window 50 episodes
- return: rolling window 50 or 100 episodes
- guided percentage: rolling window 40 or 50 episodes
- queries per episode: rolling window 40 or 50 episodes

For final summary tables, use tail-window statistics:

```text
last 100 episodes
```

Recommended aggregate columns:

```text
condition
env
seed
timesteps
oracle_accuracy
oracle_cost
oracle_cost_final
tail_success_rate
tail_return
tail_guided_pct
tail_queries_per_ep
tail_oracle_correct_rate
```

## Main Figures to Make

### Figure 1: MiniGrid Main Result

Purpose:

```text
Show that optional expert access improves MiniGrid learning.
```

Plot:

- return vs steps
- success rate vs steps
- oracle usage vs steps
- queries per episode vs steps

Conditions:

- baseline PPO
- BFS oracle free
- selected oracle costs
- VLM oracle if available

Use this as the main presentation result.

### Figure 2: Query Cost Tradeoff

Purpose:

```text
Show that query cost changes behavior.
```

Plot:

- guided percentage vs cost
- final return/success vs cost

Best format:

- line plot if x-axis is cost
- compact table if few runs

### Figure 3: Imperfect Expert / Randomized Oracle

Purpose:

```text
Show what happens when the expert is unreliable.
```

Plot:

- oracle accuracy on x-axis
- final success/return on y-axis
- one line per cost level

Alternative:

- heatmap: accuracy x cost -> final success or return

This is especially useful for connecting BFS oracle experiments to VLM
limitations.

### Figure 4: Sokoban Stress Test

Purpose:

```text
Show preliminary transfer of the idea to a harder puzzle domain.
```

Keep this small. Suggested plot:

- baseline vs perfect oracle vs randomized oracle
- success rate or return vs steps
- guided percentage or queries per episode

Do not let Sokoban dominate the presentation unless its results are much
cleaner than MiniGrid/VLM results.

### Figure 5: Linear Cost Schedule

Purpose:

```text
Show whether the agent adapts as querying becomes more expensive.
```

Plot:

- oracle cost vs steps
- guided percentage vs steps
- success/return vs steps

Best message:

```text
As query cost rises, a good policy should reduce querying while preserving
performance.
```

## Presentation Priorities

For a 4-minute final presentation:

1. Motivation: RL is cheap but sample inefficient; VLMs are capable but costly.
2. Method: PPO with one extra query action.
3. MiniGrid result: expert access improves or stabilizes learning.
4. Cost / reliability result: querying should depend on cost and expert quality.
5. Sokoban bonus: early harder-domain stress test.
6. Limitations and future work.

Avoid:

- spending too much time on architecture details
- showing too many runs
- treating Sokoban as the main project
- claiming VLM success if only BFS/randomized-oracle results are clean

## Known Gotchas

- Old Sokoban `baseline` filenames can be misleading.
- Some Vast runs were W&B-only because local artifacts were not downloaded.
- Check whether a run used `--upload-checkpoint` before assuming policy weights
  exist.
- Redoing architecture means rerunning the core matrix.
- Single-seed results should be described as preliminary.
- Randomized oracle is a proxy for VLM reliability, not a replacement for real
  VLM evaluation.

## Suggested Final Takeaway

```text
Expert querying can make sparse RL tasks easier, but the agent must balance
task reward, query cost, and expert reliability. MiniGrid is the main evidence;
Sokoban shows the same framework beginning to extend to a harder visual puzzle.
```
