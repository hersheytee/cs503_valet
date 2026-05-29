# Sokoban Experiment Handover

Current state of the Sokoban codebase as of 2026-05-29. Everything needed to
reproduce results, generate figures, and extend the work is described here.

---

## What Was Built

### Core Training (`train.py`)

PPO agent on `Sokoban-small-v0` (7×7, 2 boxes, 56×56 RGB obs). Key flags:

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
| Fixed cost × accuracy grid | `oracle_acc{ACC}_cost{C}_500k` | Costs 0.1/0.3/0.5/0.8/1.0 × acc 0.25/0.5/0.75/1.0 |
| Free perfect oracle | `oracle_acc100_cost0_500k` | Cost=0, acc=1.0 |
| Linear cost schedule | `oracle_acc100_cost_linear_0to2_3M` | 0→2 over 2M steps, total 3M |

### Runs queued / in progress

| Condition | Script in COMMANDS.md | Notes |
|---|---|---|
| Cost=0 accuracy ablation | "Cost=0 Accuracy Ablations" section | acc 0.25/0.5/0.75 at cost=0 |
| Budget oracle 1/3/5 | "Budget-Limited Oracle Runs" section | 1M steps each, 4-channel obs |

---

## Budget-Aware Oracle (Novel Contribution)

`--max-oracle-queries N` caps oracle queries per episode. The key design
decision: remaining budget is encoded as a **4th image channel** tiled
across the full 56×56 observation:

```
channel_value = int((queries_remaining / max_queries) * 255)
```

255 = full budget, 0 = exhausted. The CNN learns to condition on this.
Observation space becomes `56×56×4` — **incompatible with 3-channel
checkpoints**. Budget and non-budget runs produce separate model weights.

Implementation: `SokobanOracleWrapper` in `env_wrapper.py`.
See `README.md` § "Budget-Limited Oracle with Budget-Aware Observation" for
the full write-up (report-ready).

---

## BFS Oracle

`bfs_oracle.py` contains two BFS implementations:

- **`bfs_sokoban`** — standard Sokoban, treats boxes as interchangeable (frozenset)
- **`bfs_sokoban_fixed_target`** — FixedTarget variants, tracks `box_mapping`
  (each box must reach its paired target, not any target)

`get_oracle_action` auto-detects env type via `hasattr(env, 'box_mapping')`.

---

## Zero-Shot Transfer

Test a trained policy on `FixedTarget-Sokoban-v2` (same 7×7 board, different
puzzle variant) without fine-tuning.

```bash
# Single checkpoint
python gym_sokoban/eval_transfer.py --checkpoint runs/.../checkpoints/final.pt

# All checkpoints at once (skips already-evaluated pairs)
python gym_sokoban/eval_transfer_all.py --env-ids FixedTarget-Sokoban-v2 Sokoban-small-v0
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
| `cost_curves_acc100.png` | Success rate + oracle usage vs steps, varying cost |
| `accuracy_lines.png` | Final success + usage vs oracle accuracy, all costs |
| `linear_cost_schedule.png` | 2×3 panel — cost schedule, success, reward, usage, queries/ep, success/query |

Key flags:
- `--no-shade` — disable ±1 std shading (on by default)
- `--window N` — rolling average window over 500 interpolated uniform step points (default 50)
- `--min-steps N` — skip runs shorter than N steps (default 450k)

**Smoothing:** interpolates to 500 uniform step points first, then rolls.
Same approach as `minigrid/merged_plot.py` — much smoother than raw episode rolling.

---

## Visualisation GIFs

```bash
python gym_sokoban/eval_gif.py \
    --checkpoint runs/.../checkpoints/final.pt \
    --n-episodes 5 --fps 4 --scale 4 \
    --out gym_sokoban/figures/final/gifs/baseline.gif --no-oracle

# Budget-aware checkpoint (must specify matching budget):
python gym_sokoban/eval_gif.py \
    --checkpoint runs/.../checkpoints/final.pt \
    --max-oracle-queries 3
```

Renders: 4× upscaled board, action probability bars, oracle HUD, budget
remaining in HUD for budget runs.

Batch-generate all conditions:
```powershell
# Fill in checkpoint paths first, then:
.\gen_eval_gifs.ps1
```

Export to website:
```powershell
.\export_to_website.ps1
```

---

## Website Integration

Website repo: `../CS503-VALET-Website` (separate GitHub repo).

The Sokoban section already has an **interactive 6-button GIF picker**
(Baseline / Free Oracle / Cost=0.5 / Budget=1/3/5). GIFs are expected at:
```
CS503-VALET-Website/static/gifs/sokoban/<condition>.gif
```

The export script copies them there automatically.

---

## File Map

```
gym_sokoban/
├── train.py                # PPO training loop
├── env_wrapper.py          # SokobanOracleWrapper (oracle, budget channel, gym compat)
├── model.py                # CNNPolicy (3- or 4-channel input, auto-detected)
├── bfs_oracle.py           # BFS oracle — standard + fixed-target
├── plot_results.py         # Figure generation
├── eval_transfer.py        # Single-checkpoint zero-shot transfer eval
├── eval_transfer_all.py    # Batch transfer eval across all checkpoints
├── eval_gif.py             # GIF visualisation
├── COMMANDS.md             # All Vast.ai launch commands
├── README.md               # Architecture notes + budget oracle write-up
└── figures/final/          # Output directory for all figures + GIFs
    └── transfer_results.csv
```

---

## Immediate TODOs

1. **Wait for running jobs** — linear cost (3M) and reference jobs still training
2. **Run cost=0 accuracy ablations** — 3 jobs (acc 0.25/0.5/0.75), see COMMANDS.md
3. **Run budget oracle jobs** — budgets 1/3/5, 1M steps each, see COMMANDS.md
4. **Download results** — `rsync` from Vast instance once done
5. **Generate figures** — run `plot_results.py` with all results in `runs/`
6. **Generate GIFs** — fill checkpoint paths in `gen_eval_gifs.ps1` and run
7. **Export to website** — run `export_to_website.ps1`, commit + push website repo
