# Sokoban Experiment Handover

Current state of the Sokoban codebase as of 2026-05-29. Everything needed to
reproduce results, generate figures, and extend the work is described here.

---

## What Was Built

### Core Training (`train.py`)

PPO agent on `Sokoban-small-v0` (7x7, 2 boxes, 56x56 RGB obs). Key flags:

| Flag | Purpose |
|---|---|
| `--no-oracle` | Baseline PPO, no oracle action |
| `--oracle-cost C` | Reward penalty per oracle query |
| `--oracle-cost-final C2` | Linear cost ramp from `--oracle-cost` to `C2` |
| `--oracle-cost-anneal-steps N` | Steps to complete ramp (default = total steps) |
| `--oracle-accuracy P` | Probability oracle returns BFS-optimal action |
| `--max-oracle-queries N` | Hard per-episode query budget (see Budget Oracle below) |

Uses `AsyncVectorEnv` (not `Sync`) — critical for throughput on CPU-bound Sokoban
env stepping. Expect ~330 SPS on a V100 with 64 envs.

**GPU compatibility note:** PyTorch 2.11+ nightly drops V100 (sm_70) support.
If you see `CUDA error: no kernel image`, reinstall:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Experiment Matrix

### Runs completed (in `sokoban_vast_results/runs/`)

| Condition | exp_name pattern | Notes |
|---|---|---|
| Baseline | `baseline_minigridcnn_500k` | No oracle, 500k steps |
| Fixed cost x accuracy grid | `oracle_acc{ACC}_cost{C}_500k` | Costs 0.1/0.3/0.5/0.8/1.0 x acc 0.25/0.5/0.75/1.0 |
| Free perfect oracle | `oracle_acc100_cost0_500k` | Cost=0, acc=1.0 |
| Linear cost schedule | `oracle_acc100_cost_linear_0to2_3M` | 0->2 over 2M steps, total 3M |

### Runs queued / in progress

| Condition | Script in COMMANDS.md | Notes |
|---|---|---|
| Budget oracle 1/3/5 | "Budget-Limited Oracle Runs" section | 1M steps each, 4-channel obs |

Cost=0 accuracy ablations (acc 0.25/0.5/0.75) were dropped — results were
confounded by an oracle bug present during those training runs (see Known Issues).

---

## Known Issues / Bugs Fixed

### BFS oracle env-type detection (FIXED)

`is_fixed_target_env` previously used `hasattr(env, 'box_mapping')` to distinguish
standard Sokoban from FixedTarget variants. Both env types have `box_mapping`, so
this always returned True for standard Sokoban, causing the fixed-target BFS to
run on standard puzzles. Fixed to check the class name hierarchy instead:

```python
def is_fixed_target_env(env_unwrapped) -> bool:
    return 'FixedTarget' in type(env_unwrapped).__name__ or \
           any('FixedTarget' in c.__name__ for c in type(env_unwrapped).__mro__)
```

**Impact on training:** All training runs on `Sokoban-small-v0` were affected.
The fixed-target BFS uses `box_mapping` to track box positions — in standard
Sokoban, `box_mapping` is set at reset and never updated as boxes move, so
the oracle was computing paths from stale initial box positions. The oracle
action was essentially noise for all standard Sokoban training runs.

**What this means for results:**
- High-cost runs / baseline: largely unaffected (agent learned to ignore oracle)
- Low-cost / free oracle runs: trained with random oracle guidance — interesting
  in itself but not what was intended
- FixedTarget transfer eval: unaffected, the fixed-target BFS was always correct
- All eval/GIF generation after the fix: correct oracle behaviour confirmed
  (20/20 success rate in direct oracle test)

### eval_transfer_all.py oracle_accuracy (FIXED)

Previously hardcoded `oracle_accuracy=1.0` during transfer evaluation regardless
of training config. Fixed to pass `rec["oracle_accuracy"]` so each policy is
evaluated with the same oracle quality it was trained with.

---

## Budget-Aware Oracle (Novel Contribution)

`--max-oracle-queries N` caps oracle queries per episode. The key design
decision: remaining budget is encoded as a **4th image channel** tiled
across the full 56x56 observation:

```
channel_value = int((queries_remaining / max_queries) * 255)
```

255 = full budget, 0 = exhausted. The CNN learns to condition on this.
Observation space becomes `56x56x4` — **incompatible with 3-channel
checkpoints**. Budget and non-budget runs produce separate model weights.

Implementation: `SokobanOracleWrapper` in `env_wrapper.py`.

---

## BFS Oracle

`bfs_oracle.py` contains two BFS implementations:

- **`bfs_sokoban`** — standard Sokoban, treats boxes as interchangeable (frozenset)
- **`bfs_sokoban_fixed_target`** — FixedTarget variants, tracks `box_mapping`
  (each box must reach its paired target, not any target)

`get_oracle_action` auto-detects env type via class name (see Known Issues above).

---

## Zero-Shot Transfer

Test a trained policy on `FixedTarget-Sokoban-v2` (same 7x7 board, different
puzzle variant) without fine-tuning. Evaluation uses the training oracle_accuracy
and oracle_cost=0.0 (cost only affects reward, not action quality).

```bash
# All checkpoints at once (parallel, skips already-evaluated pairs)
python -u gym_sokoban/eval_transfer_all.py

# Key flags
--workers 8          # parallel worker processes (default 8)
--n-episodes 200     # episodes per checkpoint
--min-steps 450000   # skip short/cancelled runs
```

Results saved to `gym_sokoban/figures/final/transfer_results.csv`.

---

## Plotting

```bash
python gym_sokoban/plot_results.py --out-dir gym_sokoban/figures/final
```

Outputs (all in `gym_sokoban/figures/final/`):

| File | Content |
|---|---|
| `cost_curves_acc100.png` | Success rate + oracle usage + queries/ep vs steps, varying cost (3 panels) |
| `accuracy_lines.png` | Final success + usage + queries/ep vs oracle accuracy, all costs (3 panels) |
| `linear_cost_schedule.png` | 2x3 panel — cost schedule, success, reward, usage, queries/ep, success/query |
| `transfer_results.png` | Zero-shot transfer to FixedTarget: success + usage + queries/ep vs accuracy (3 panels) |

Key flags:
- `--no-shade` — disable +/-1 std shading (on by default)
- `--window N` — rolling average window over 500 interpolated uniform step points (default 50)
- `--min-steps N` — skip runs shorter than N steps (default 450k)

Cost=0 runs are excluded from `accuracy_lines.png` and `transfer_results.png`
(results were confounded by the oracle bug).

---

## Visualisation GIFs

```bash
python gym_sokoban/eval_gif.py \
    --checkpoint runs/.../checkpoints/final.pt \
    --n-episodes 5 --fps 4 --scale 4 \
    --stochastic --success-only \
    --out gym_sokoban/figures/final/gifs/baseline.gif --no-oracle
```

Key flags:
- `--stochastic` — sample from policy distribution (recommended, looks more natural)
- `--success-only` — retry seeds until N successful episodes collected
- `--max-episode-steps` — defaults to 120 (oracle needs more headroom than training's 50)

Batch-generate all 6 conditions (paths already filled in):
```powershell
.\gen_eval_gifs.ps1
```

The 6 conditions are: Baseline / Free Oracle / Cost=0.1 / Cost=0.5 / Cost=0.8 / Linear.
GIF filenames must match exactly: `baseline`, `oracle_free`, `oracle_cost01`,
`oracle_cost05`, `oracle_cost08`, `oracle_linear`.

Export to website:
```powershell
.\export_to_website.ps1
```

---

## Website Integration

Website repo: `../CS503-VALET-Website` (separate GitHub repo).
Section: `#exp-sokoban` in `index.html`.

Interactive 6-button GIF picker expects GIFs at:
```
CS503-VALET-Website/static/gifs/sokoban/<condition>.gif
```

Plot figures go to:
```
CS503-VALET-Website/static/images/
```

Planning notes for the Sokoban website section: `gym_sokoban/WEBSITE.md`.

---

## File Map

```
gym_sokoban/
├── train.py                # PPO training loop
├── env_wrapper.py          # SokobanOracleWrapper (oracle, budget channel, gym compat)
├── model.py                # CNNPolicy (3- or 4-channel input, auto-detected)
├── bfs_oracle.py           # BFS oracle -- standard + fixed-target (env type via class name)
├── plot_results.py         # Figure generation (4 plots)
├── eval_transfer.py        # Single-checkpoint zero-shot transfer eval
├── eval_transfer_all.py    # Batch transfer eval (parallel, progress bars)
├── eval_gif.py             # GIF visualisation (stochastic + success-only modes)
├── test_oracle.py          # Oracle correctness smoke test (raw gym env)
├── ARCHITECTURES.md        # CNN architecture reference for all models
├── COMMANDS.md             # All Vast.ai launch commands
├── WEBSITE.md              # Website section planning notes
└── figures/final/          # Output directory for all figures + GIFs
    └── transfer_results.csv
```

---

## Immediate TODOs

1. **Run budget oracle jobs** — budgets 1/3/5, 1M steps each, see COMMANDS.md
2. **Download results** — `rsync` from Vast instance once done
3. **Generate figures** — `python gym_sokoban/plot_results.py`
4. **Re-run transfer eval** — delete `transfer_results.csv` and re-run
   `eval_transfer_all.py` (oracle_accuracy bug was fixed; existing CSV is stale)
5. **Generate GIFs** — `.\gen_eval_gifs.ps1` (checkpoint paths already filled in)
6. **Export to website** — `.\export_to_website.ps1`, then commit + push website repo
