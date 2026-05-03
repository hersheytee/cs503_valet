#!/bin/bash
# Lance tous les runs : 2 envs × 3 conditions × 3 seeds = 18 jobs SLURM
#   - oracle gratuit  (--oracle-cost 0.0)
#   - oracle payant   (--oracle-cost 0.01)
#   - baseline PPO    (--no-oracle)

set -e

SEEDS=(1 2 3)
ENVS=(
    "MiniGrid-Empty-5x5-v0,empty"
    "MiniGrid-DoorKey-8x8-v0,doorkey"
)
ORACLE_COST=0.01

mkdir -p logs figures

for env_entry in "${ENVS[@]}"; do
    ENV_ID=$(echo $env_entry | cut -d',' -f1)
    ENV_TYPE=$(echo $env_entry | cut -d',' -f2)
    for SEED in "${SEEDS[@]}"; do

        # Oracle gratuit (upper bound)
        echo "Submitting oracle_free  ${ENV_TYPE} seed${SEED} ..."
        sbatch \
            --job-name="ub_free_${ENV_TYPE}_s${SEED}" \
            --export=ENV_ID="$ENV_ID",ENV_TYPE="$ENV_TYPE",SEED="$SEED",EXTRA_ARGS="" \
            job.sh

        # Oracle payant
        echo "Submitting oracle_paid  ${ENV_TYPE} seed${SEED} ..."
        sbatch \
            --job-name="ub_paid_${ENV_TYPE}_s${SEED}" \
            --export=ENV_ID="$ENV_ID",ENV_TYPE="$ENV_TYPE",SEED="$SEED",EXTRA_ARGS="--oracle-cost ${ORACLE_COST}" \
            job.sh

        # Baseline PPO pur
        echo "Submitting baseline     ${ENV_TYPE} seed${SEED} ..."
        sbatch \
            --job-name="base_${ENV_TYPE}_s${SEED}" \
            --export=ENV_ID="$ENV_ID",ENV_TYPE="$ENV_TYPE",SEED="$SEED",EXTRA_ARGS="--no-oracle" \
            job.sh

    done
done

echo ""
echo "18 jobs submitted. Monitor with: squeue -u $USER"
