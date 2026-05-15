#!/bin/bash
# Lance tous les runs : 2 envs × 3 conditions × 3 seeds = 18 jobs SLURM
#   - oracle gratuit  (--oracle-cost 0.0)
#   - oracle payant   (--oracle-cost 0.01)
#   - baseline PPO    (--no-oracle)
#
# Usage:
#   ./launch_all.sh                  # BFS oracle (défaut)
#   ./launch_all.sh --vlm qwen3b     # VLM oracle (Qwen2.5-VL 3B)
#   ./launch_all.sh --vlm qwen7b     # VLM oracle (Qwen2.5-VL 7B)

set -e

# ── Paramètre VLM (optionnel) ─────────────────────────────────────────────────
VLM_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vlm) VLM_MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

VLM_ARG=""
if [[ -n "$VLM_MODEL" ]]; then
    VLM_ARG="--vlm-model $VLM_MODEL"
    echo "VLM oracle : $VLM_MODEL"
else
    echo "VLM oracle : none (BFS)"
fi

# ── Configuration ─────────────────────────────────────────────────────────────
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
            --export=ENV_ID="$ENV_ID",ENV_TYPE="$ENV_TYPE",SEED="$SEED",EXTRA_ARGS="$VLM_ARG" \
            job.sh

        # Oracle payant
        echo "Submitting oracle_paid  ${ENV_TYPE} seed${SEED} ..."
        sbatch \
            --job-name="ub_paid_${ENV_TYPE}_s${SEED}" \
            --export=ENV_ID="$ENV_ID",ENV_TYPE="$ENV_TYPE",SEED="$SEED",EXTRA_ARGS="--oracle-cost ${ORACLE_COST} $VLM_ARG" \
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
