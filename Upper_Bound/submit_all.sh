#!/bin/bash
#SBATCH --job-name=upperbound_all
#SBATCH --output=logs/slurm_all_%j.out
#SBATCH --error=logs/slurm_all_%j.err
#SBATCH --time=10:00:00
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
mkdir -p logs figures

ENV_ID="MiniGrid-DoorKey-8x8-v0"
ENV_TYPE="doorkey"

for SEED in 4 5 6 7 8; do

    echo "=== Oracle gratuit       (cost=0.000) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.0 \
        --exp-name oracle_free --total-timesteps 500000

    echo "=== Oracle payant faible (cost=0.005) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.005 \
        --exp-name oracle_paid_0005 --total-timesteps 500000

    echo "=== Oracle payant moyen  (cost=0.010) | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --oracle-cost 0.01 \
        --exp-name oracle_paid_001 --total-timesteps 500000

    echo "=== Baseline PPO                      | seed=$SEED ==="
    python -u train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --seed "$SEED" --no-oracle \
        --exp-name baseline --total-timesteps 500000

done

echo "Tous les runs terminés."
