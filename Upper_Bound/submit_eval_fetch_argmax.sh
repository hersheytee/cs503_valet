#!/bin/bash
#SBATCH --job-name=eval_fetch_argmax
#SBATCH --output=logs/slurm_eval_fetch_argmax_%j.out
#SBATCH --error=logs/slurm_eval_fetch_argmax_%j.err
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
mkdir -p figures/gifs logs

CKPT="checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt"

echo "=== Zero-shot transfer: DoorKey-16x16 partial → Fetch-16x16 (argmax) ==="
python eval.py \
    --checkpoint "$CKPT" \
    --env-id MiniGrid-Fetch-16x16-N3-v0 --env-type fetch \
    --partial-obs \
    --n-episodes 3 --seed 42 \
    --out figures/gifs/transfer_fetch_argmax.gif

echo "Done."
