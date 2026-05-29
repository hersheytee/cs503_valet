# VALET — VLM-Assisted LEarning with Targeted queries

> CS503 — EPFL  
> **Basile Bultez, Tomas Jouven, Gaspard Lafont, Harsh Talathi**

VALET augments a PPO agent with an explicit, penalized `query_oracle` action to consult an expert (BFS oracle or VLM) on demand. The agent learns endogenously *when* guidance is worth its cost, rather than querying blindly or following a hand-crafted schedule.

---

## Abstract

We introduce VALET, a framework that augments a PPO agent with an explicit, penalized query action to consult an expert (BFS Oracle or VLM) on demand. The agent learns endogenously when guidance is worth its cost, rather than querying blindly or following a hand-crafted schedule. Experiments on MiniGrid-DoorKey and Sokoban show faster convergence, reduced variance, and self-regulated oracle usage. The querying behavior transfers zero-shot to unseen tasks, and the agent naturally stops querying when the oracle is not reliable enough. Furthermore, the framework enables training a single navigation model across multiple tasks without task-specific retraining.

---

## Project structure

```
cs503_project/
├── minigrid/               # Core training & evaluation code (MiniGrid)
├── Upper_Bound/            # Analysis scripts — plots, transfer eval, per-cost stats
├── gym_sokoban/            # Sokoban experiments (PPO + oracle)
├── Benchmark/              # VLM oracle benchmark (dataset + evaluation)
├── Literature review/      # Related papers
├── VALET proposal.pdf      # Original project proposal
├── checkpoints/            # Sokoban checkpoints
├── logs/                   # Root-level logs
└── figures/                # Root-level figures
```

---

## Method

The agent's action space is extended from A to A ∪ {query}. When the agent selects `query`, the oracle's suggested action is executed in the environment, and the agent incurs a fixed penalty `-c` subtracted from the reward. All other actions are executed normally with no penalty. No external scheduler decides when to query — the agent discovers the cost–benefit trade-off through the RL objective alone.

**Training:** PPO with clipped surrogate objective (clip=0.2), GAE (λ=0.95), entropy bonus (coef=0.01) on non-guided steps only.

**Observation:** Raw RGB image — no extra flags or augmentation.

**Architecture (partial obs):**
```
56×56×3 → Conv(3→32, k=3, s=2) → Conv(32→64, k=3, s=1) → Conv(64→64, k=3, s=1)
→ AdaptiveAvgPool(8×8) → Flatten (4096) → FC(256) → policy head + value head
~1.1M parameters
```

**Oracles:**
- **BFS oracle** — perfect optimal action from full env state (upper bound)
- **Noisy BFS** — optimal action with probability p, random otherwise
- **VLM oracle** — WeThink-Qwen2.5-VL 7B (selected via benchmarking)

---

## Experiments

### A. BFS Oracle — Upper Bound

Sweep of oracle costs on MiniGrid-DoorKey-8×8 and 16×16 (partial obs), 5 seeds each.

- With `c=0`, agent queries ~90% of steps and reaches max return immediately
- With `c=0.01`, usage stabilizes at ~40% — agent internalizes frequent sub-goals
- Key finding: oracle usage drops sharply between `c=0.010` and `c=0.012`
- Interpretability: converged agent (c=0.01) opens doors autonomously but queries for navigation

### B. VLM Benchmarking

300-sample MiniGrid evaluation set annotated by BFS oracle. Metrics: macro-F1 vs. latency.

- WeThink-Qwen2.5-VL 7B achieves highest macro-F1 (~51%)
- No model exceeds 52% accuracy
- Prompt phrasing alone shifts F1 by >20 points within a single model
- Temporal context (past frames/actions) yields no meaningful improvement

### C. VLM as Oracle

WeThink-Qwen2.5-VL 7B integrated as drop-in replacement for BFS. Results inconclusive: ~35% accuracy is insufficient. At `c=0` (free queries), VLM guidance actively hurts performance. At higher costs, agent queries less and approaches baseline.

### D. Noisy Oracle

Oracle returns optimal action with probability p, random otherwise. Clear accuracy threshold identified:
- Below threshold: agent stops querying entirely (self-regulates)
- Above threshold: success rate and oracle usage increase sharply
- **Key insight:** deploying VALET with a weak VLM carries no risk — the agent simply learns to ignore it when `c > 0`

### E. Zero-Shot Transfer

DoorKey-16×16 agent deployed on unseen tasks without any retraining:

| Task | PPO baseline | VALET argmax | VALET stochastic |
|------|-------------|--------------|-----------------|
| Fetch-16×16 | 6% | 69% | **100%** (efficiency 0.87) |
| MultiRoom-N6 | 0% | 97% | **100%** (efficiency 0.98) |

Stochastic sampling outperforms argmax: the agent retains a small probability of querying at every step, acting as an escape mechanism when stuck.

### F. Sokoban

VALET extended to Gym-Sokoban (irreversible actions, longer horizons). Same self-regulating threshold behavior observed. Zero-shot transfer to FixedTarget-Sokoban-v2 succeeds when trained with sufficient oracle accuracy.

Budget-limited variant: agent given at most K queries per episode, remaining budget encoded as a 4th input channel.

---

## `minigrid/` — Training code

| File | Description |
|---|---|
| `train.py` | PPO training loop |
| `env_wrapper.py` | Adds `query_oracle` action, RGB obs |
| `oracle.py` | BFS oracle for DoorKey / Empty |
| `oracle_transfer.py` | BFS oracle for Fetch, GoToDoor, GoToObject, MultiRoom |
| `model.py` | CNN — 40×40×3, ~26M params (8×8 full obs) |
| `model_large.py` | CNN — 128×128×3, ~1.1M params (16×16 full obs) |
| `model_partial.py` | CNN — 56×56×3, ~1.1M params (partial obs, any grid) |
| `eval.py` | Run checkpoint + save GIF |
| `vlm_oracle.py` | VLM drop-in replacement for BFS |
| `download_models.py` | Download VLMs from HuggingFace |
| `merged_plot.py` | All conditions overlaid, plasma colorbar |
| `compare_plot.py` | One row per condition, 4 metric columns |

---

## `Upper_Bound/` — Analysis & evaluation

| File | Description |
|---|---|
| `eval_transfer_stats.py` | Zero-shot transfer (100 eps, success/return/oracle%/efficiency) |
| `per_cost_plot.py` | One 1×4 training curve plot per oracle cost vs baseline |
| `submit_eval_transfer_stats.sh` | Transfer eval on Fetch-16×16 |
| `submit_eval_transfer_stats_multiroom.sh` | Transfer eval on MultiRoom-N6 |
| `submit_eval_all_partial_doorkey.sh` | GIFs for all partial-obs models |
| `job_noise.sh` | Noisy oracle sweep |
| `noise_plot.py` | Noisy oracle plots |
| `vlm_plot.py` | VLM oracle plots |

---

## `Benchmark/` — VLM benchmarking

| File | Description |
|---|---|
| `create_dataset.py` | Generate (observation, oracle_action) pairs |
| `create_history_dataset.py` | Dataset with action history context |
| `bench_job.sh` | SLURM benchmark job |
| `analyse_results.ipynb` | Result analysis |

---

## Quick start

```bash
conda activate nanofm
pip install -r minigrid/requirements.txt

# Train — DoorKey-16x16, partial obs, oracle cost=0.01
cd minigrid
python train.py \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --oracle-cost 0.01 --reward-shaping --large-model --partial-obs \
    --total-timesteps 2000000 --seed 4 \
    --exp-name oracle_paid_001_16_partial --save-model

# GIF
python eval.py \
    --checkpoint checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --partial-obs --stochastic --n-episodes 3 \
    --out all_gif/paid001.gif

# Zero-shot transfer stats
cd ../Upper_Bound
python eval_transfer_stats.py \
    --checkpoint ../minigrid/checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id MiniGrid-Fetch-16x16-N3-v0 --env-type fetch \
    --partial-obs --n-episodes 100 \
    --csv-out figures/fetch_comparison.csv \
    --out figures/fetch_comparison.png
```

## Cluster sync

```bash
# Push code
rsync -avz minigrid/ <user>@<cluster>:~/Upper_Bound/

# Pull results
rsync -avz <user>@<cluster>:~/Upper_Bound/logs/    minigrid/logs/
rsync -avz <user>@<cluster>:~/Upper_Bound/figures/ Upper_Bound/figures/
rsync -avz "<user>@<cluster>:~/Upper_Bound/checkpoints/best__*_partial__*.pt" \
    minigrid/checkpoints/
```
