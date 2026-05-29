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
