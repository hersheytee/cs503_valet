#!/bin/bash
# Generate all PPO result plots from training CSVs in logs/
# Usage: bash plot_all.sh [min_steps]

MIN=${1:-0}
mkdir -p figures

echo "=== 8x8 full obs ==="
python merged_plot.py --tag "" --log-dir logs \
    --min-steps $MIN \
    --title "PPO + Oracle — MiniGrid-DoorKey-8x8 (full obs)" \
    --out figures/merged_8x8.png

echo "=== 16x16 full obs ==="
python merged_plot.py --tag 16 --log-dir logs \
    --min-steps $MIN \
    --title "PPO + Oracle — MiniGrid-DoorKey-16x16 (full obs)" \
    --out figures/merged_16x16_full.png

echo "=== 16x16 partial obs ==="
python merged_plot.py --tag 16_partial --log-dir logs \
    --min-steps $MIN \
    --title "PPO + Oracle — MiniGrid-DoorKey-16x16 (partial obs)" \
    --out figures/merged_16x16_partial.png

echo ""
echo "Done. Figures saved to figures/"
