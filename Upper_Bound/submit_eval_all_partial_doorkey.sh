#!/bin/bash
#SBATCH --job-name=eval_all_partial
#SBATCH --output=logs/slurm_eval_all_partial_%j.out
#SBATCH --error=logs/slurm_eval_all_partial_%j.err
#SBATCH --time=01:30:00
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
mkdir -p all_gif logs

ENV_ID="MiniGrid-DoorKey-16x16-v0"
ENV_TYPE="doorkey"
N_EPS=3
SEED=42

# baseline — no oracle action
echo "--- baseline_16_partial ---"
python eval.py \
    --checkpoint checkpoints/best__baseline_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id $ENV_ID --env-type $ENV_TYPE \
    --partial-obs --no-oracle \
    --n-episodes $N_EPS --seed $SEED \
    --out all_gif/doorkey16_baseline_partial.gif

# oracle_free
echo "--- oracle_free_16_partial ---"
python eval.py \
    --checkpoint checkpoints/best__oracle_free_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
    --env-id $ENV_ID --env-type $ENV_TYPE \
    --partial-obs \
    --n-episodes $N_EPS --seed $SEED \
    --out all_gif/doorkey16_oracle_free_partial.gif

# oracle_paid costs
for TAG in 001 002 003 004 005 007 008 009 011 012 015 018; do
    CKPT="checkpoints/best__oracle_paid_${TAG}_16_partial__MiniGrid-DoorKey-16x16-v0.pt"
    if [ ! -f "$CKPT" ]; then
        echo "[skip] $TAG — not found"
        continue
    fi
    echo "--- oracle_paid_${TAG}_16_partial ---"
    python eval.py \
        --checkpoint "$CKPT" \
        --env-id $ENV_ID --env-type $ENV_TYPE \
        --partial-obs \
        --n-episodes $N_EPS --seed $SEED \
        --out "all_gif/doorkey16_oracle_paid${TAG}_partial.gif"
done

echo "Done."
