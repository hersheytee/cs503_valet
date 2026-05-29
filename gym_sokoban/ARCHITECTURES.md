# Sokoban / MiniGrid Architecture Notes

This file records the visual policy variants we have discussed for Sokoban experiments. The current `gym_sokoban/model.py` implements the fair WorldCoder-style full-sprite variant.

## Current: Fair WorldCoder Full-Sprite CNN

Target use:

- Recreate the WorldCoder Sokoban PPO baseline while still using full sprites.
- `Sokoban-small-v0`, 7x7 board.
- 8 pixels per cell, so observations are `56x56x3`.
- Pure PPO baseline should be run with `--no-oracle`.

Architecture:

```text
Input: 56x56x3 RGB, channel-last in buffers
Preprocess: uint8 / 255, then channel-first

Conv2d(3, 16, kernel=2, stride=1)
ReLU
Conv2d(16, 32, kernel=2, stride=1)
ReLU
Conv2d(32, 64, kernel=2, stride=1)
ReLU
AdaptiveAvgPool2d(4, 4)
Flatten
Linear(1024, hidden_dim=64)
ReLU
policy_head
value_head
```

Why the adaptive pool exists:

WorldCoder's original input was a compact `3x7x7` grid-like representation. With three `2x2` stride-1 convolutions, the spatial size becomes:

```text
7x7 -> 6x6 -> 5x5 -> 4x4
```

That gives `64 * 4 * 4 = 1024` features before the policy/value heads. If we apply the same conv stack directly to `56x56` full sprites, the spatial size becomes:

```text
56x56 -> 55x55 -> 54x54 -> 53x53
```

That gives `64 * 53 * 53 = 179,776` features, which is not a fair adaptation of the original model. Pooling to `4x4` preserves the original feature footprint while allowing full 8-pixel sprites.

## Original WorldCoder-Style Tiny-Grid CNN

Target use:

- One RGB/tiny symbolic value per Sokoban cell.
- Observation is effectively `3x7x7`, not full sprites.

Architecture:

```text
Input: 3x7x7

Conv2d(3, 16, kernel=2, stride=1)
ReLU
Conv2d(16, 32, kernel=2, stride=1)
ReLU
Conv2d(32, 64, kernel=2, stride=1)
ReLU
Flatten 1024
policy/value heads
```

This is smaller and more local than our MiniGrid-style CNNs.

## Previous Sokoban / MiniGrid-Large CNN

Target use:

- Larger rendered observations such as `128x128x3`.
- Matched the MiniGrid 16x16 full-observation visual policy family.

Architecture:

```text
Input: 128x128x3 RGB, or 56x56x3 if resized

Conv2d(3, 32, kernel=3, stride=2, padding=1)
ReLU
Conv2d(32, 64, kernel=3, stride=2, padding=1)
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)
ReLU
AdaptiveAvgPool2d(8, 8)
Flatten 4096
Linear(4096, hidden_dim=256)
ReLU
policy_head
value_head
```

Compared with the WorldCoder-style CNN, this downsamples more aggressively and feeds a larger `4096`-dimensional feature vector into the shared head.

## MiniGrid Partial-Observation CNN

Target use:

- MiniGrid partial observations with a 7x7 field of view rendered at 8 pixels per cell.
- Observation is `56x56x3`.

Architecture:

```text
Input: 56x56x3 RGB

Conv2d(3, 32, kernel=3, stride=2, padding=1)
ReLU
Conv2d(32, 64, kernel=3, stride=1, padding=1)
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)
ReLU
AdaptiveAvgPool2d(8, 8)
Flatten 4096
Linear(4096, hidden_dim=256)
ReLU
policy_head
value_head
```

This has the same `56x56x3` input shape as the full-sprite WorldCoder adaptation, but it uses larger 3x3 filters, one stride-2 downsampling layer, and a larger pooled feature map.

## Recommended Ablations

For Sokoban-to-Sokoban transfer:

```text
scratch WorldCoder full-sprite CNN
transferred WorldCoder full-sprite CNN trunk
frozen transferred trunk + new heads
previous MiniGrid-large CNN on the same 56x56 sprites
MiniGrid partial CNN on the same 56x56 sprites
```

Keep action space and observation size fixed when comparing transfer results. For `Sokoban-small-v0`, use the full 7x7 board at `56x56x3`. For larger boards, use a 7x7 player-centered crop rendered at the same 8 pixels per cell.

## Current Concern

The active WorldCoder full-sprite adaptation may be underpowered for rendered
sprites. It is faithful to the original WorldCoder filter sizes, but the
original network operated on compact `3x7x7` cell-level input. On `56x56`
sprites, the adaptive pool back to `4x4` may discard too much spatial detail.

The most plausible stronger drop-in candidate is the MiniGrid partial-observation
CNN above:

```text
Input: 56x56x3
Conv2d(3, 32, kernel=3, stride=2, padding=1)
Conv2d(32, 64, kernel=3, stride=1, padding=1)
Conv2d(64, 64, kernel=3, stride=1, padding=1)
AdaptiveAvgPool2d(8, 8)
Linear(4096, 256)
```

If we switch to this architecture, all core runs must be rerun for clean
comparison:

- PPO baseline
- perfect oracle cost sweep
- randomized-accuracy oracle sweep
- linear oracle-cost schedule

---

## Final Architecture (all runs in sokoban_vast_results/)

The switch was made. All runs in the final experiment matrix use the
MiniGrid partial-observation CNN described above — not the WorldCoder variant.
The "Current Concern" above was resolved by adopting this architecture.

### Standard policy (3-channel, all non-budget runs)

```text
Input: 56x56x3 RGB, uint8, channel-last
Preprocess: / 255, permute to channel-first (3x56x56)

Conv2d(3,  32, kernel=3, stride=2, padding=1)   -> 32x28x28
ReLU
Conv2d(32, 64, kernel=3, stride=1, padding=1)   -> 64x28x28
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)   -> 64x28x28
ReLU
AdaptiveAvgPool2d(8, 8)                          -> 64x8x8 = 4096
Flatten
Linear(4096, 256)
ReLU
policy_head: Linear(256, n_actions)   init std=0.01
value_head:  Linear(256, 1)           init std=1.0
```

`n_actions = 10` (9 env actions + oracle query), or `9` for `--no-oracle` baseline.
All weights initialised with orthogonal init (std=sqrt(2)), bias=0.

### Budget-aware policy (4-channel, --max-oracle-queries runs)

Identical to above except the first conv takes 4 input channels:

```text
Input: 56x56x4  (RGB + budget channel)
Conv2d(4, 32, kernel=3, stride=2, padding=1)
... remainder identical ...
```

The 4th channel encodes remaining query budget as a spatially-constant
uint8 value: `int((queries_remaining / max_queries) * 255)`.
These checkpoints are incompatible with 3-channel checkpoints.

### Key hyperparameters (from config.yaml)

| Parameter | Value |
|---|---|
| hidden_dim | 256 |
| obs_size | 56 |
| max_episode_steps | 50 |
| n_envs | 64 (AsyncVectorEnv) |
| total_timesteps | 500k (standard) / 3M (linear schedule run) |
| optimizer | Adam, lr=3e-4, annealed |
| PPO clip | 0.2 |
| entropy coef | 0.0 |
| GAE lambda | 0.95 |

---

## MiniGrid Architectures (minigrid/)

Three model files exist for different MiniGrid observation sizes.

### model.py — DoorKey-8x8 full observation (40x40x3)

```text
Input: 40x40x3 RGB, uint8, channel-last
Preprocess: / 255, permute to channel-first (3x40x40)

Conv2d(3,  32, kernel=3, stride=1, padding=1)   -> 32x40x40
ReLU
Conv2d(32, 64, kernel=3, stride=1, padding=1)   -> 64x40x40
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)   -> 64x40x40
ReLU
Flatten                                          -> 102400
Linear(102400, 256)
ReLU
policy_head: Linear(256, n_actions)   init std=0.01
value_head:  Linear(256, 1)           init std=1.0
```

No pooling — the full spatial map is flattened directly.

### model_partial.py — DoorKey-16x16 partial obs (56x56x3)

Same architecture as the Sokoban model (`gym_sokoban/model.py`): one
stride-2 conv then AdaptiveAvgPool to 8x8.

```text
Input: 56x56x3 RGB, uint8, channel-last
Preprocess: / 255, permute to channel-first (3x56x56)

Conv2d(3,  32, kernel=3, stride=2, padding=1)   -> 32x28x28
ReLU
Conv2d(32, 64, kernel=3, stride=1, padding=1)   -> 64x28x28
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)   -> 64x28x28
ReLU
AdaptiveAvgPool2d(8, 8)                          -> 64x8x8 = 4096
Flatten
Linear(4096, 256)
ReLU
policy_head: Linear(256, n_actions)   init std=0.01
value_head:  Linear(256, 1)           init std=1.0
```

### model_large.py — DoorKey-16x16 full observation (128x128x3)

Two stride-2 convolutions for more aggressive downsampling of the
larger input.

```text
Input: 128x128x3 RGB, uint8, channel-last
Preprocess: / 255, permute to channel-first (3x128x128)

Conv2d(3,  32, kernel=3, stride=2, padding=1)   -> 32x64x64
ReLU
Conv2d(32, 64, kernel=3, stride=2, padding=1)   -> 64x32x32
ReLU
Conv2d(64, 64, kernel=3, stride=1, padding=1)   -> 64x32x32
ReLU
AdaptiveAvgPool2d(8, 8)                          -> 64x8x8 = 4096
Flatten
Linear(4096, 256)
ReLU
policy_head: Linear(256, n_actions)   init std=0.01
value_head:  Linear(256, 1)           init std=1.0
```

### Which model was used for which experiment

| Experiment | Model file | Obs size |
|---|---|---|
| DoorKey-8x8 BFS oracle sweep | `model.py` | 40x40x3 |
| DoorKey-16x16 partial obs | `model_partial.py` | 56x56x3 |
| DoorKey-16x16 full obs | `model_large.py` | 128x128x3 |
| Sokoban all runs | `gym_sokoban/model.py` | 56x56x3 (or x4 for budget) |
