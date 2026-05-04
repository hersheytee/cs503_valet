#!/bin/bash
# Lance tous les runs en parallèle sur Izar
# 4 conditions × 3 seeds = 12 jobs SLURM

set -e

SEEDS=(1 2 3)

mkdir -p logs figures checkpoints

for SEED in "${SEEDS[@]}"; do

    # 1. Oracle Gratuit (cost = 0.0)
    echo "Submitting oracle_free seed${SEED} ..."
    sbatch \
        --job-name="soko_free_s${SEED}" \
        --export=EXP_NAME="oracle_free",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.0" \
        job.sh

    # 2. Oracle Payant (cost = 0.01)
    echo "Submitting oracle_paid_01 seed${SEED} ..."
    sbatch \
        --job-name="soko_paid01_s${SEED}" \
        --export=EXP_NAME="oracle_paid_01",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.01" \
        job.sh

    # 3. Oracle Payant (cost = 0.05)
    echo "Submitting oracle_paid_05 seed${SEED} ..."
    sbatch \
        --job-name="soko_paid05_s${SEED}" \
        --export=EXP_NAME="oracle_paid_05",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.05" \
        job.sh

    # 4. Baseline PPO pur (sans oracle)
    echo "Submitting baseline seed${SEED} ..."
    sbatch \
        --job-name="soko_base_s${SEED}" \
        --export=EXP_NAME="baseline",SEED="$SEED",EXTRA_ARGS="--no-oracle" \
        job.sh

done

echo ""
echo "12 jobs submitted to Izar! Monitor with: squeue -u $USER"