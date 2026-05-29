#!/bin/bash
#SBATCH --job-name=fetch_partial
#SBATCH --output=logs/slurm_fetch_%j.out
#SBATCH --error=logs/slurm_fetch_%j.err
#SBATCH --time=24:00:00
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

ENV_ID="MiniGrid-Fetch-16x16-N3-v0"
ENV_TYPE="fetch"
STEPS=3000000

for SEED in 4 5 6; do

    echo "=== Oracle gratuit  (cost=0.000) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.0 --reward-shaping --partial-obs \
        --exp-name oracle_free_fetch --total-timesteps $STEPS --save-model

    echo "=== Oracle payant   (cost=0.010) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.01 --reward-shaping --partial-obs \
        --exp-name oracle_paid_001_fetch --total-timesteps $STEPS --save-model

    echo "=== Oracle payant   (cost=0.030) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.03 --reward-shaping --partial-obs \
        --exp-name oracle_paid_003_fetch --total-timesteps $STEPS --save-model

    echo "=== Oracle payant   (cost=0.050) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.05 --reward-shaping --partial-obs \
        --exp-name oracle_paid_005_fetch --total-timesteps $STEPS --save-model

    echo "=== Baseline PPO                 | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --no-oracle --reward-shaping --partial-obs \
        --exp-name baseline_fetch --total-timesteps $STEPS --save-model

done

echo "Runs Fetch terminés."
