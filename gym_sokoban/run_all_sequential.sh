#!/bin/bash
# Run the Sokoban RGB PPO experiment matrix sequentially.

set -euo pipefail

ENV_ID="${ENV_ID:-Sokoban-small-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SEEDS=(${SEEDS:-1})
PYTHON_EXE="${PYTHON_EXE:-python}"
SAVE_MODEL="${SAVE_MODEL:-0}"

mkdir -p logs figures checkpoints

run_one() {
    local exp_name="$1"
    local seed="$2"
    local extra_args="$3"
    local save_args=""

    if [[ "$SAVE_MODEL" == "1" ]]; then
        save_args="--save-model"
    fi

    echo "=== ${exp_name} | ${ENV_ID} | seed=${seed} | ${extra_args} ==="
    $PYTHON_EXE gym_sokoban/train.py \
        --env-id "$ENV_ID" \
        --seed "$seed" \
        --exp-name "$exp_name" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        $save_args \
        $extra_args
}

for seed in "${SEEDS[@]}"; do
    run_one "oracle_free" "$seed" "--oracle-cost 0.0"
    run_one "oracle_paid_001" "$seed" "--oracle-cost 0.01"
    run_one "oracle_paid_005" "$seed" "--oracle-cost 0.05"
    run_one "oracle_paid_010" "$seed" "--oracle-cost 0.10"
    run_one "oracle_paid_020" "$seed" "--oracle-cost 0.20"
    run_one "baseline" "$seed" "--no-oracle"
done

echo "All sequential runs completed."
