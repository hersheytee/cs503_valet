#!/bin/bash
# Lance tous les runs séquentiellement sur un seul GPU

set -e

mkdir -p logs figures

SEEDS=(1 2 3)

ENVS=(
    "MiniGrid-DoorKey-8x8-v0,doorkey"
)

for env_entry in "${ENVS[@]}"; do
    ENV_ID=$(echo $env_entry | cut -d',' -f1)
    ENV_TYPE=$(echo $env_entry | cut -d',' -f2)
    for SEED in "${SEEDS[@]}"; do

        echo "=== Oracle gratuit | $ENV_ID | seed=$SEED ==="
        python train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
            --seed "$SEED" --oracle-cost 0.0 \
            --exp-name oracle_free --total-timesteps 500000

        echo "=== Oracle payant  | $ENV_ID | seed=$SEED ==="
        python train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
            --seed "$SEED" --oracle-cost 0.01 \
            --exp-name oracle_paid --total-timesteps 500000

        echo "=== Baseline PPO   | $ENV_ID | seed=$SEED ==="
        python train.py --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
            --seed "$SEED" --no-oracle \
            --exp-name baseline --total-timesteps 500000

    done
done

echo "Tous les runs terminés."
