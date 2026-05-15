"""
Preview the exact prompt that would be sent to a model for a given sample.
No model inference — purely text output for debugging and prompt design.

Shows 4 sections:
  HISTORY            — all available history steps (which are included marked ►)
  IMAGE FILES        — file paths with existence check
  INTERLEAVED CONTENT — exact sequence sent to the model (text label → image → … → prompt)
  PROMPT TEXT        — the main text prompt with line numbers

Usage:
    # First sample, optimal variant, baseline mode, 3 history images (default)
    python preview_prompt.py --dataset ./history_dataset/dataset.json

    # Select sample by index or id
    python preview_prompt.py --idx 5
    python preview_prompt.py --id 00042

    # Select by phase or oracle action
    python preview_prompt.py --phase find_key
    python preview_prompt.py --action pickup
    python preview_prompt.py --action turn_right

    # Change variant, mode, number of history images
    python preview_prompt.py --idx 10 --variant optimal --mode thinking --history_images 5
    python preview_prompt.py --idx 10 --variant negative_rules --mode cot --history_images 0

    # Compare all variants for one sample
    python preview_prompt.py --idx 0 --all_variants

    # Compare all modes for one variant
    python preview_prompt.py --idx 0 --variant optimal --all_modes

    # Use InternVL image marker style (shows Image-i: <image> in INTERLEAVED CONTENT)
    python preview_prompt.py --idx 0 --model internvl8b_mpo
"""

import argparse
import json
import sys
from pathlib import Path

# Import prompt builders from run_benchmark (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark import (
    MODELS,
    PROMPT_VARIANTS,
    _MODE_CONFIGS,
    _DIR_WORD,
    _detect_phase,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(title: str = "", width: int = 72, char: str = "─"):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print(char * width)


def _sample_header(sample: dict, n_history: int, mode: str, view: str, variant: str):
    phase   = _detect_phase(sample)
    history = sample.get("history", [])
    n_hist  = min(n_history, len(history))

    _divider("SAMPLE", char="═")
    print(f"  id           : {sample.get('id', '?')}")
    print(f"  env          : {sample.get('env', '?')}")
    print(f"  complexity   : {sample.get('complexity', '?')}")
    print(f"  mission      : {sample.get('mission', '?')}")
    print(f"  agent_pos    : {sample.get('agent_pos')}  dir: {sample.get('agent_dir_str','?')}")
    print(f"  carrying     : {sample.get('agent_carrying') or 'nothing'}")
    print(f"  door_open    : {sample.get('door_open', 'N/A')}")
    print(f"  phase        : {phase}")
    print(f"  oracle action: {sample.get('optimal_actions')}  ({sample.get('action_names')})")
    print(f"  history_type : {sample.get('history_type', 'N/A')}")
    print(f"  action_seq   : {sample.get('action_sequence', [])}")
    print()
    print(f"  view         : {view}")
    print(f"  variant      : {variant}")
    print(f"  mode         : {mode}")
    print(f"  history_imgs : {n_hist} (of {len(history)} available)")
    _divider()


def _history_summary(sample: dict, n_history: int):
    history = sample.get("history", [])
    if not history:
        print("  (no history in this sample)")
        return
    recent = history[-n_history:] if n_history > 0 else []
    print(f"  Full history ({len(history)} steps available):")
    for h in history:
        marker = "  ►" if h in recent else "   "
        print(f"{marker}  step {h['step']:+d}  "
              f"pos={h['agent_pos']}  dir={h['agent_dir_str']}  "
              f"carry={h.get('agent_carrying') or '-'}  "
              f"action→ {h.get('action_name','?')}")
    if not recent:
        print("  (no history images requested)")
    else:
        print(f"\n  ► = included in prompt ({len(recent)} images)")


def _image_paths(sample: dict, n_history: int, view: str, dataset_root: Path):
    history = sample.get("history", [])
    recent  = history[-n_history:] if n_history > 0 else []
    imgs    = []
    for i, h in enumerate(recent):
        path = dataset_root / h[f"{view}_image"]
        imgs.append((f"history img {i+1} (step {h['step']:+d})", path))
    cur_path = dataset_root / sample[f"{view}_image"]
    imgs.append((f"current (img {len(recent)+1})", cur_path))
    return imgs


# ── Main display ──────────────────────────────────────────────────────────────

def show_prompt(sample: dict, dataset_root: Path,
                view: str, variant_name: str, mode: str,
                n_history: int, model_key: str = "qwen7b"):

    mode_cfg  = _MODE_CONFIGS[mode]
    img_style = MODELS[model_key].get("img_style", "qwen")
    variant   = next(v for v in PROMPT_VARIANTS if v["name"] == variant_name)

    n_hist_use = min(n_history, len(sample.get("history", [])))

    _sample_header(sample, n_history, mode, view, variant_name)

    _divider("HISTORY")
    _history_summary(sample, n_history)

    _divider("IMAGE FILES")
    for label, path in _image_paths(sample, n_hist_use, view, dataset_root):
        exists = "✓" if path.exists() else "✗ MISSING"
        print(f"  [{exists}]  {label:30s}  {path}")

    _divider("INTERLEAVED CONTENT (as sent to model)")
    history  = sample.get("history", [])
    recent   = history[-n_hist_use:] if n_hist_use > 0 else []
    for i, h in enumerate(recent):
        fw    = _DIR_WORD.get(h.get("agent_dir_str", "?"), h.get("agent_dir_str", "?"))
        carry = "key" if h.get("agent_carrying") == "key" else "nothing"
        act   = h.get("action_name", "?")
        marker = f"Image-{i+1}: <image>  " if img_style == "internvl" else ""
        print(f"  [TEXT ] {marker}[Step {h['step']:+d} | facing {fw} | carrying {carry} | action: {act}]")
        print(f"  [IMAGE] history img {i+1}")
    cur_n   = n_hist_use + 1
    cur_mrk = f"Image-{cur_n}: <image>  " if img_style == "internvl" else ""
    print(f"  [TEXT ] {cur_mrk}[CURRENT STATE — image {cur_n}]")
    print(f"  [IMAGE] current")
    print(f"  [TEXT ] (main prompt — see PROMPT TEXT below)")

    _divider("PROMPT TEXT")
    prompt_text = variant["fn"](
        sample, view, n_hist_use, img_style,
        mode_cfg["cot"], mode_cfg["thinking"]
    )
    # Print with line numbers for easier reading
    for i, line in enumerate(prompt_text.split("\n"), 1):
        print(f"  {i:3d} │ {line}")

    max_tok = mode_cfg["max_tokens_override"] or variant["base_max_out"]
    print(f"\n  max_tokens : {max_tok}")
    print(f"  ~chars     : {len(prompt_text)}")
    print(f"  ~words     : {len(prompt_text.split())}")
    _divider(char="═")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Preview a benchmark prompt without running inference")
    p.add_argument("--dataset",       type=str, default="./history_dataset/dataset.json")
    p.add_argument("--idx",           type=int, default=0,
                   help="Sample index in the dataset (default: 0)")
    p.add_argument("--id",            type=str, default=None,
                   help="Sample id string (overrides --idx)")
    p.add_argument("--phase",         type=str, default=None,
                   choices=["navigate", "find_key", "open_door"],
                   help="Pick the first sample matching this phase")
    p.add_argument("--action",        type=str, default=None,
                   help="Pick the first sample whose oracle action matches (e.g. pickup)")
    p.add_argument("--history_type",  type=str, default=None,
                   choices=["optimal", "suboptimal"],
                   help="Pick the first sample with this history_type")
    p.add_argument("--view",          type=str, default="global",
                   choices=["global", "partial"])
    p.add_argument("--variant",       type=str, default="optimal",
                   choices=[v["name"] for v in PROMPT_VARIANTS])
    p.add_argument("--mode",          type=str, default="baseline",
                   choices=list(_MODE_CONFIGS.keys()))
    p.add_argument("--history_images",type=int, default=3,
                   help="Number of history frames to include (default: 3)")
    p.add_argument("--model",         type=str, default="qwen7b",
                   choices=list(MODELS.keys()),
                   help="Model family — affects image marker style (qwen vs internvl)")
    p.add_argument("--all_variants",  action="store_true",
                   help="Show all 3 prompt variants for the selected sample")
    p.add_argument("--all_modes",     action="store_true",
                   help="Show all 3 reasoning modes for the selected variant")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = Path(args.dataset)
    dataset_root = dataset_path.parent

    with open(dataset_path) as f:
        samples = json.load(f)

    # ── Sample selection ──────────────────────────────────────────────────────
    sample = None

    if args.id:
        sample = next((s for s in samples if s.get("id") == args.id), None)
        if sample is None:
            print(f"  Error: no sample with id='{args.id}'")
            sys.exit(1)
    elif args.phase:
        sample = next((s for s in samples if _detect_phase(s) == args.phase), None)
        if sample is None:
            print(f"  Error: no sample with phase='{args.phase}'")
            sys.exit(1)
    elif args.action:
        sample = next(
            (s for s in samples if args.action in s.get("action_names", [])),
            None
        )
        if sample is None:
            print(f"  Error: no sample with action='{args.action}'")
            sys.exit(1)
    elif args.history_type:
        sample = next(
            (s for s in samples if s.get("history_type") == args.history_type),
            None
        )
        if sample is None:
            print(f"  Error: no sample with history_type='{args.history_type}'")
            sys.exit(1)
    else:
        if args.idx >= len(samples):
            print(f"  Error: --idx {args.idx} out of range (dataset has {len(samples)} samples)")
            sys.exit(1)
        sample = samples[args.idx]

    print(f"\n  Dataset : {dataset_path}  ({len(samples)} samples)")

    # ── Display ───────────────────────────────────────────────────────────────
    variants_to_show = [v["name"] for v in PROMPT_VARIANTS] if args.all_variants \
                       else [args.variant]
    modes_to_show    = list(_MODE_CONFIGS.keys()) if args.all_modes \
                       else [args.mode]

    for vname in variants_to_show:
        for mname in modes_to_show:
            show_prompt(
                sample=sample,
                dataset_root=dataset_root,
                view=args.view,
                variant_name=vname,
                mode=mname,
                n_history=args.history_images,
                model_key=args.model,
            )
