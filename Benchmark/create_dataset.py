"""
Generates the full dataset to evaluate vision-language models on the MiniGrid environment.
----------------
dataset/
  images/
    00000_global.png
    00000_partial.png
    ...
  dataset.json
  dataset.parquet    (if pandas+pyarrow installed)

Usage:
    # Full dataset: 200 main samples + 50 pickup + 50 toggle
    python create_dataset.py

    # Custom sizes
    python create_dataset.py --n 500 --n_rare 100 --out ./dataset

    # Reproducible with a specific seed
    python create_dataset.py --seed_offset 42
"""

import argparse

from utils.generate_dataset import generate
from utils.sample_rare_actions import generate_rare


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the MiniGrid VLM benchmark dataset"
    )
    p.add_argument("--n",           type=int, default=200,
                   help="Number of main samples (default: 200)")
    p.add_argument("--n_rare",      type=int, default=50,
                   help="Rare-action samples PER action (pickup + toggle) (default: 50)")
    p.add_argument("--out",         type=str, default="./dataset",
                   help="Output directory (default: ./dataset)")
    p.add_argument("--seed_offset", type=int, default=0,
                   help="Seed offset for main generation (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  Step 1 / 2 — Main dataset generation")
    print("=" * 60)
    generate(
        n_samples=args.n,
        out_dir=args.out,
        seed_offset=args.seed_offset,
    )

    print("=" * 60)
    print("  Step 2 / 2 — Rare actions (pickup & toggle)")
    print("=" * 60)
    generate_rare(
        n_per_action=args.n_rare,
        out_dir=args.out,
        seed_offset=args.seed_offset + 5000,
    )
