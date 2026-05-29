# Sokoban Result Analysis Handoff

This handoff captures the current Sokoban analysis state after the W&B CSV
exports and plotting cleanup. MiniGrid remains the main project evidence;
Sokoban should be presented as a harder-domain stress test / extension.

## Current Story

Question:

```text
Can an RL agent learn when to query an expert, and does query behavior respond
to expert cost and reliability?
```

Sokoban setup:

- PPO agent has normal Sokoban actions plus `query_oracle`.
- `query_oracle` executes an oracle-provided action and logs the step as guided.
- Oracle quality can be randomized with `--oracle-accuracy`.
- Oracle cost can be fixed with `--oracle-cost` or scheduled with
  `--oracle-cost-final`.
- The clean no-oracle PPO baseline uses `--no-oracle`.
- Behavior cloning is not part of the current Sokoban setup.

## Most Useful Current Figures

Presentation-ready figures are in:

```text
gym_sokoban/figures/
```

Use these two first:

```text
wandb_export_cost_usage_curves_acc100.png
wandb_export_accuracy_usage_curves_cost05.png
```

What they show:

- `wandb_export_cost_usage_curves_acc100.png`
  - Varying query cost at 100% oracle accuracy.
  - Oracle curves use only the newest W&B chart exports.
  - Current cost lines from the export: `0.3`, `0.5`, `1.0`.
  - Baseline is added in black from the true local no-oracle baseline.

- `wandb_export_accuracy_usage_curves_cost05.png`
  - Varying oracle accuracy at fixed cost `0.5`.
  - Oracle curves use the newest W&B chart exports.
  - Baseline is added in black on the success-rate panel.

Important plotting choices:

- The plots use line-only curves; no shaded confidence bands.
- The colormap is `plasma`.
- Legends are plain labels, with no `(n=...)` suffix.
- Success-rate curves use `success_rate_50ep` from W&B exports or a local
  50-episode rolling mean.
- Oracle-usage curves currently use the downloaded W&B `guided_pct_40ep`
  export. To make usage a true 50-episode average, export `guided_pct_50ep`
  or raw per-episode `guided_pct`.

Plotting script:

```text
gym_sokoban/plot_wandb_exports.py
```

Regenerate the focused figures with:

```bash
cd gym_sokoban
../venv/Scripts/python.exe plot_wandb_exports.py --only-requested
```

## Current CSV Inputs

Newest W&B chart exports:

```text
gym_sokoban/logs/wandb_export_2026-05-25T20_20_18.805+02_00.csv
gym_sokoban/logs/wandb_export_2026-05-25T20_20_24.201+02_00.csv
```

These contain:

- `charts/success_rate_50ep`
- `charts/guided_pct_40ep`
- `charts/global_step`

The quick W&B summary table is:

```text
gym_sokoban/wandb_sokoban_summary.csv
```

This is useful for checking which conditions exist, but it should not be used
as a learning curve source.

## Data Rules

Do not mix old local oracle runs into the main W&B cost/accuracy plots unless
the figure is explicitly labeled as historical/local.

Reason:

- Older local oracle logs contain free / `0.1` / `0.2` / `0.3` / `0.5` style
  cost sweeps, but they are not the same export set as the newest randomized
  oracle W&B grid.
- The current final cost plot intentionally uses only newest W&B oracle curves:
  `cost=0.3`, `0.5`, `1.0` at `oracle_accuracy=1.0`.
- The local true baseline is still okay to overlay in black because it is a
  clean no-oracle comparison.

Baseline rule:

```text
guided_pct = 0
queries_per_ep = 0
```

Some old files named `baseline__...csv` are not true baselines because they
have nonzero guided usage. The plotting script filters local baselines and only
uses baseline files with `guided_pct == 0`.

## Known Available Conditions

From `wandb_sokoban_summary.csv`, the randomized-oracle 500k grid includes:

- cost `0.0`: accuracies `0.0`, `0.25`, `0.5`, `0.75`
- cost `0.2`: accuracies `0.25`, `0.5`, `0.75`
- cost `0.3`: accuracy `1.0`
- cost `0.5`: accuracies `0.25`, `0.5`, `0.75`, `1.0`
- cost `1.0`: accuracies `0.25`, `0.5`, `0.75`, `1.0`

This is why the current accuracy-sweep figure uses `cost=0.5`: it has the
cleanest complete set of accuracies in the exported chart data.

## Interpretation Notes

Use careful language:

- These are mostly single-seed Sokoban results.
- Randomized oracle is a proxy for imperfect VLM reliability, not a VLM result.
- W&B summary values are final logged values, not necessarily tail averages.
- `oracle_accuracy` is the configured probability of returning the BFS-optimal
  action; `oracle_correct_rate` is the measured correctness over queried steps.
  They can differ when query counts are low or random actions accidentally match
  the BFS action.
- If an agent stops querying under high cost, measured oracle-correctness can
  become noisy because the denominator is small.

## Recommended Presentation Use

For Sokoban, keep the message compact:

```text
Sokoban is a harder visual puzzle stress test. The same query-action framework
extends to it, and the agent's query behavior changes with cost and oracle
reliability, but results are preliminary and single-seed.
```

Use two plots:

1. Varying query cost at perfect oracle accuracy.
2. Varying oracle accuracy at fixed cost `0.5`.

Do not make Sokoban the main evidence unless MiniGrid/VLM results are weak or
unavailable. The stronger claim should remain:

```text
Expert querying can help sparse RL, but useful querying depends on the reward
cost and reliability of the expert.
```

## Useful Commands

Quick W&B summary export:

```bash
cd gym_sokoban
../venv/Scripts/python.exe import_data.py
```

Full W&B history export, if needed:

```bash
cd gym_sokoban
../venv/Scripts/python.exe export_wandb_history.py
```

Plot from downloaded W&B chart CSVs:

```bash
cd gym_sokoban
../venv/Scripts/python.exe plot_wandb_exports.py --only-requested
```

## Next Best Improvements

If there is time:

- Export `guided_pct_50ep` from W&B so both panels use 50-episode smoothing.
- Export/download the baseline curve directly from W&B instead of relying on
  local baseline logs.
- Compute a final summary table using the last 100 episodes for:
  - success rate
  - return
  - guided percentage
  - queries per episode
  - oracle correctness
- Rerun multi-seed versions only if the presentation needs stronger Sokoban
  claims.
