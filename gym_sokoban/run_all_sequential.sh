#!/bin/bash
# Runs all experiments sequentially on a single GPU/CPU
# Optimized for 1 seed to reduce local training time

set -e

# Create directories in the root folder
mkdir -p logs figures checkpoints

# Reduced to 1 seed to save time on laptop execution[cite: 4]
SEEDS=(1)

# The environment we are testing
ENV_ID="Sokoban-small-v0"

# Path to your virtual environment's Python[cite: 4]
PYTHON_EXE="venv/Scripts/python.exe"

# Path to the training script
TRAIN_SCRIPT="gym_sokoban/train.py"

for SEED in "${SEEDS[@]}"; do

    echo "=== Free Oracle (cost=0.0) | $ENV_ID | seed=$SEED ==="
    $PYTHON_EXE $TRAIN_SCRIPT --env-id "$ENV_ID" \
        --seed "$SEED" --oracle-cost 0.0 \
        --exp-name oracle_free --total-timesteps 500000

    echo "=== Paid Oracle (cost=0.01) | $ENV_ID | seed=$SEED ==="
    $PYTHON_EXE $TRAIN_SCRIPT --env-id "$ENV_ID" \
        --seed "$SEED" --oracle-cost 0.01 \
        --exp-name oracle_paid_01 --total-timesteps 500000

    echo "=== Paid Oracle (cost=0.05) | $ENV_ID | seed=$SEED ==="
    $PYTHON_EXE $TRAIN_SCRIPT --env-id "$ENV_ID" \
        --seed "$SEED" --oracle-cost 0.05 \
        --exp-name oracle_paid_05 --total-timesteps 500000

    echo "=== Baseline PPO (no oracle) | $ENV_ID | seed=$SEED ==="
    $PYTHON_EXE $TRAIN_SCRIPT --env-id "$ENV_ID" \
        --seed "$SEED" --no-oracle \
        --exp-name baseline --total-timesteps 500000

done

echo "All runs completed successfully."