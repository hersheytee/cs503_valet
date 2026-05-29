#!/bin/bash
#SBATCH --job-name=dk16_partial_fine
#SBATCH --output=logs/slurm_16x16_partial_fine_%j.out
#SBATCH --error=logs/slurm_16x16_partial_fine_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

set -e

source ~/.bashrc
conda activate nanofm
pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR
mkdir -p logs figures checkpoints

ENV_ID="MiniGrid-DoorKey-16x16-v0"
ENV_TYPE="doorkey"
STEPS=3000000

for SEED in 4 5 6 7 8; do

    echo "=== Oracle cost=0.012 | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.012 --reward-shaping --partial-obs \
        --exp-name oracle_paid_012_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle cost=0.015 | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.015 --reward-shaping --partial-obs \
        --exp-name oracle_paid_015_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle cost=0.018 | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.018 --reward-shaping --partial-obs \
        --exp-name oracle_paid_018_16_partial --total-timesteps $STEPS --save-model

done

echo "Runs fine-grained partial obs terminés."
