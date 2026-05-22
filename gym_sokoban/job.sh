#!/bin/bash
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

set -euo pipefail

: "${ENV_ID:=Sokoban-small-v0}"
: "${EXP_NAME:=oracle_free}"
: "${SEED:=1}"
: "${TOTAL_TIMESTEPS:=500000}"
: "${EXTRA_ARGS:=--oracle-cost 0.0}"

echo "=== Job ${SLURM_JOB_ID:-local} | env=${ENV_ID} exp=${EXP_NAME} seed=${SEED} ==="
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"

source ~/.bashrc
conda activate cs503_proj

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs figures checkpoints

python -u gym_sokoban/train.py \
    --env-id "$ENV_ID" \
    --seed "$SEED" \
    --exp-name "$EXP_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --save-model \
    $EXTRA_ARGS

echo "End: $(date)"
