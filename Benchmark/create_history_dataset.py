"""
Generates a dataset where each sample includes the current state AND the N
preceding states, with a mix of optimal and suboptimal histories.
Rare-action samples (pickup / toggle) are constructed explicitly so their
count is fully controlled, exactly like in the original create_dataset.py.

Output layout:
  history_dataset/
    images/
      00000_global.png          <- current state
      00000_partial.png
      00000_h1_global.png       <- 1 step before current (most recent)
      00000_h1_partial.png
      ...
      00000_h5_global.png       <- 5 steps before current (oldest)
      00000_h5_partial.png
    dataset.json                <- includes action_sequence field
    dataset.parquet

Each JSON entry has:
  "action_sequence": [a_{-5}, a_{-4}, a_{-3}, a_{-2}, a_{-1}]
  where a_{-k} is the action taken FROM history step -k TO the next state.

Usage:
    # Default: 200 main + 50 pickup + 50 toggle, 50% suboptimal history
    python create_history_dataset.py

    # Custom
    python create_history_dataset.py --n 500 --n_rare 100 --history_len 3

    # Reproducible
    python create_history_dataset.py --seed_offset 42

    # Only optimal history
    python create_history_dataset.py --p_suboptimal 0.0
"""

import argparse

from utils.generate_history_dataset import generate_with_history


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a MiniGrid history dataset for VLM benchmarking"
    )
    p.add_argument("--n",            type=int,   default=200,
                   help="Number of main samples (default: 200)")
    p.add_argument("--n_rare",       type=int,   default=50,
                   help="Rare-action samples PER action (pickup + toggle) "
                        "(default: 50, set 0 to skip)")
    p.add_argument("--history_len",  type=int,   default=5,
                   help="Number of preceding states per sample (default: 5)")
    p.add_argument("--p_suboptimal", type=float, default=0.5,
                   help="Fraction of main samples with suboptimal history "
                        "(default: 0.5)")
    p.add_argument("--out",          type=str,   default="./history_dataset",
                   help="Output directory (default: ./history_dataset)")
    p.add_argument("--seed_offset",  type=int,   default=0,
                   help="Seed offset for reproducibility (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print(f"  History Dataset")
    print(f"  Main samples   : {args.n}")
    print(f"  Rare samples   : {args.n_rare} × pickup + {args.n_rare} × toggle")
    print(f"  History length : {args.history_len} steps")
    print(f"  Suboptimal mix : {args.p_suboptimal:.0%}")
    print("=" * 60)

    generate_with_history(
        n_samples=args.n,
        out_dir=args.out,
        seed_offset=args.seed_offset,
        history_len=args.history_len,
        p_suboptimal=args.p_suboptimal,
        n_rare=args.n_rare,
    )
