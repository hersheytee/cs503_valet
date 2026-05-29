#!/bin/bash
#SBATCH --job-name=dk16_partial
#SBATCH --output=logs/slurm_16x16_partial_%j.out
#SBATCH --error=logs/slurm_16x16_partial_%j.err
#SBATCH --time=72:00:00
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

    echo "=== Oracle gratuit        (cost=0.000) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.0 --reward-shaping --partial-obs \
        --exp-name oracle_free_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle payant         (cost=0.010) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.01 --reward-shaping --partial-obs \
        --exp-name oracle_paid_001_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle payant         (cost=0.020) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.02 --reward-shaping --partial-obs \
        --exp-name oracle_paid_002_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle payant         (cost=0.030) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.03 --reward-shaping --partial-obs \
        --exp-name oracle_paid_003_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle payant         (cost=0.040) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.04 --reward-shaping --partial-obs \
        --exp-name oracle_paid_004_16_partial --total-timesteps $STEPS --save-model

    echo "=== Oracle payant élevé   (cost=0.050) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.05 --reward-shaping --partial-obs \
        --exp-name oracle_paid_005_16_partial --total-timesteps $STEPS --save-model

    echo "=== Baseline PPO                       | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --no-oracle --reward-shaping --partial-obs \
        --exp-name baseline_16_partial --total-timesteps $STEPS --save-model

done

echo "Runs partial obs 16x16 terminés."
