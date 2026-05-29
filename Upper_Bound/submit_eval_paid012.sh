#!/bin/bash
#SBATCH --job-name=eval_paid012
#SBATCH --output=logs/slurm_eval_paid012_%j.out
#SBATCH --error=logs/slurm_eval_paid012_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

set -e

source ~/.bashrc
conda activate nanofm

cd $SLURM_SUBMIT_DIR
mkdir -p figures/gifs

CKPT="checkpoints/best__oracle_paid_012_16_partial__MiniGrid-DoorKey-16x16-v0.pt"

echo "=== Eval oracle_paid_012 — argmax ==="
python eval.py \
    --checkpoint "$CKPT" \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --partial-obs --n-episodes 3 --seed 42 \
    --out figures/gifs/eval_paid012_16_partial_probs.gif

echo "=== Eval oracle_paid_012 — stochastic ==="
python eval.py \
    --checkpoint "$CKPT" \
    --env-id MiniGrid-DoorKey-16x16-v0 --env-type doorkey \
    --partial-obs --stochastic --n-episodes 3 --seed 42 \
    --out figures/gifs/eval_paid012_16_partial_probs_stoch.gif

echo "Done."
