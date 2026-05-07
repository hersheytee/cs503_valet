#!/bin/bash
# Launches all runs in parallel on Izar
# 4 conditions × 3 seeds = 12 SLURM jobs

set -e

SEEDS=(1)

# Create necessary directories
mkdir -p logs figures checkpoints

for SEED in "${SEEDS[@]}"; do

    echo "Submitting oracle_free seed${SEED} ..."
    sbatch \
        --job-name="soko_free_s${SEED}" \
        --export=ALL,EXP_NAME="oracle_free",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.0" \
        job.sh

    echo "Submitting oracle_paid_02 seed${SEED} ..."
    sbatch \
        --job-name="soko_paid02_s${SEED}" \
        --export=ALL,EXP_NAME="oracle_paid_02",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.2" \
        job.sh

    echo "Submitting oracle_paid_03 seed${SEED} ..."
    sbatch \
        --job-name="soko_paid03_s${SEED}" \
        --export=ALL,EXP_NAME="oracle_paid_03",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.3" \
        job.sh

    echo "Submitting oracle_paid_05 seed${SEED} ..."
    sbatch \
        --job-name="soko_paid05_s${SEED}" \
        --export=ALL,EXP_NAME="oracle_paid_05",SEED="$SEED",EXTRA_ARGS="--oracle-cost 0.5" \
        job.sh

    echo "Submitting baseline seed${SEED} ..."
    sbatch \
        --job-name="soko_base_s${SEED}" \
        --export=ALL,EXP_NAME="baseline",SEED="$SEED",EXTRA_ARGS="--no-oracle" \
        job.sh

done

echo ""
echo "All jobs submitted to Izar! Monitor with: squeue -u $USER"