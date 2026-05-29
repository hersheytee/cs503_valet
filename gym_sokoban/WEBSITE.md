# Sokoban Website Section — Handover Notes

Target section: `#exp-sokoban` in `CS503-VALET-Website/index.html`
Already stubbed in the HTML — search for `id="exp-sokoban"` to find it.

---

## Assets

### Figures (copy to `static/images/`)
| File | Status | Description |
|---|---|---|
| `gym_sokoban/figures/final/cost_curves_acc100.png` | ✅ ready | Success rate + oracle usage + queries/ep vs steps, all costs at acc=1.0 (3-panel) |
| `gym_sokoban/figures/final/accuracy_lines.png` | ✅ ready | Success + oracle usage + queries/ep vs oracle accuracy, lines by cost (3-panel) |
| `gym_sokoban/figures/final/linear_cost_schedule.png` | ✅ ready | 2×3 panel: cost schedule, success, reward, oracle usage, queries/ep, success/query |
| `gym_sokoban/figures/final/transfer_results.png` | ⚠️ needs re-run | 3-panel: success + oracle usage + queries/ep vs accuracy, lines by cost, FixedTarget only |

To regenerate all: `python gym_sokoban/plot_results.py`
Transfer results also need: `python -u gym_sokoban/eval_transfer_all.py` first (takes ~30min on CPU).

### GIFs (copy to `static/gifs/sokoban/`)
The interactive 6-button picker in the HTML expects exactly these filenames:
| Filename | Button label | Status |
|---|---|---|
| `baseline.gif` | Baseline | ⏳ pending — `baseline_minigridcnn_500k` checkpoint, `--no-oracle` |
| `oracle_free.gif` | Free Oracle | ⏳ pending — `oracle_acc100_cost0_500k` |
| `oracle_cost01.gif` | Cost=0.1 | ⏳ pending — `oracle_acc100_cost01_500k` |
| `oracle_cost05.gif` | Cost=0.5 | ⏳ pending — `oracle_acc100_cost05_500k` |
| `oracle_cost08.gif` | Cost=0.8 | ⏳ pending — `oracle_acc100_cost08_500k` |
| `oracle_linear.gif` | Linear | ⏳ pending — `oracle_acc100_cost_linear_0to2_3M` |

Run: `.\gen_eval_gifs.ps1` (fill in checkpoint paths first — see `HANDOVER.md` § Visualisation GIFs).
Export to website: `.\export_to_website.ps1`

Budget GIFs use a 4-channel model checkpoint (incompatible with 3-channel ones).

---

## Proposed Section Structure

The existing stub is thin. Replace it with this structure:

### 1. Environment intro (2–3 sentences)
Sokoban is harder than MiniGrid: actions are irreversible, dead-ends are reachable after
a few bad pushes, and solving a puzzle requires multi-step planning. We built a BFS oracle
that finds the optimal action given full board state.

### 2. Interactive GIF picker  ← MOVE THIS FIRST (already in the HTML, just reorder)
The 6-button picker is already wired in JavaScript. Put it right after the intro paragraph,
before any training curves. CLIPasso3D style: show the agent behaving first, explain after.

Layout: `is-one-third` column for the picker+gif, `is-two-thirds` for cost_curves_acc100.png.
This is already in the HTML — just reorder so picker comes before the figures.

### 3. Cost ablation
Figure: `cost_curves_acc100.png` (full width or right-column next to the gif picker)
Caption: "Training curves at oracle accuracy=1.0 under varying query costs.
Oracle-assisted agents converge faster; usage self-regulates as cost increases."

### 4. Accuracy ablation
Figure: `accuracy_lines.png` (full width)
Caption: "Final metrics vs oracle accuracy, one line per cost. Below the accuracy threshold
the agent stops querying entirely — the same threshold behavior seen in the MiniGrid noisy
oracle ablation."
Explicitly cross-link to `#exp-noisy` section.

### 5. Linear cost schedule
Figure: `linear_cost_schedule.png` (full width, it's a rich 2×3 panel)
Caption: "Linearly increasing query cost (0→2 over 2M steps, 3M total). The agent bootstraps
on cheap guidance early and internalizes the policy as querying becomes expensive."

### 6. Zero-shot transfer
Figure: `transfer_results.png`
Caption: "Zero-shot transfer to FixedTarget-Sokoban-v2 (never seen during training).
Performance tracks oracle accuracy — only policies trained with sufficient guidance quality
transfer successfully."

### 7. Budget-limited oracle (novel contribution)
1–2 sentences explaining the 4th channel design: remaining budget encoded as a spatially-constant
uint8 value so the CNN can condition on it. Show budget GIFs via the picker (already wired).
Note: results are pending; if weak, frame as exploratory / future work.

### 8. VLM note (1 sentence)
"We did not evaluate a real VLM on Sokoban due to compute constraints; the noisy oracle
ablation suggests a VLM would need >75% action accuracy to reliably help the agent."

---

## What's already in the HTML (don't duplicate)

The current `#exp-sokoban` stub has:
- The 6-button gif picker + sokoban-demo img + sokoban-caption p
- `setSokobanGif()` JS function (at bottom of file, inside the closing script tag — search for it)
- Placeholders for cost_curves_acc100.png, accuracy_lines.png, linear_cost_schedule.png

The stub intro text ("To verify that VALET generalizes...") is fine, keep it.

---

## Key things NOT in the current stub that need adding

1. Transfer results section (new subsection after linear schedule)
2. Budget oracle subsection (new, after transfer)
3. VLM note (1 sentence, end of section)
4. Richer figure captions (current ones are 1 line, expand to 2–3)
5. Cross-link to `#exp-noisy` from the accuracy ablation paragraph

---

## Layout patterns to follow (matching rest of page)

```html
<!-- Full-width figure -->
<div class="has-text-centered mt-4 mb-2">
  <img src="./static/images/FILENAME.png" style="width: 85%; border-radius: 4px;">
  <p class="mt-2" style="font-size: 0.82em; color: #555;">
    <strong>Figure N.</strong> Caption here.
  </p>
</div>

<!-- Two-column: text left, figure right -->
<div class="columns is-vcentered mt-4">
  <div class="column is-half"> <p>...</p> </div>
  <div class="column is-half has-text-centered">
    <img src="..." style="width: 100%; border-radius: 4px;">
  </div>
</div>
```

Figure numbers in the Sokoban section start at 9 (Figures 1–8 are used by earlier sections).
