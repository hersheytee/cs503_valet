#!/bin/bash
# Submit the Sokoban RGB PPO experiment matrix to SLURM.

set -euo pipefail

ENV_ID="${ENV_ID:-Sokoban-small-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SEEDS=(${SEEDS:-1 2 3})

mkdir -p logs figures checkpoints

submit_run() {
    local exp_name="$1"
    local seed="$2"
    local extra_args="$3"

    echo "Submitting ${exp_name} | seed=${seed} | ${extra_args}"
    sbatch \
        --job-name="soko_${exp_name}_s${seed}" \
        --export=ALL,ENV_ID="$ENV_ID",EXP_NAME="$exp_name",SEED="$seed",TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS",EXTRA_ARGS="$extra_args" \
        gym_sokoban/job.sh
}

for seed in "${SEEDS[@]}"; do
    submit_run "oracle_free" "$seed" "--oracle-cost 0.0"
    submit_run "oracle_paid_001" "$seed" "--oracle-cost 0.01"
    submit_run "oracle_paid_005" "$seed" "--oracle-cost 0.05"
    submit_run "oracle_paid_010" "$seed" "--oracle-cost 0.10"
    submit_run "oracle_paid_020" "$seed" "--oracle-cost 0.20"
    submit_run "baseline" "$seed" "--no-oracle"
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
