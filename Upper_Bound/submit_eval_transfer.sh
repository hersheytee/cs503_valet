#!/bin/bash
#SBATCH --job-name=eval_transfer
#SBATCH --output=logs/slurm_eval_transfer_%j.out
#SBATCH --error=logs/slurm_eval_transfer_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

set -e

source ~/.bashrc
conda activate nanofm

cd $SLURM_SUBMIT_DIR
mkdir -p figures/gifs

ENV_ID="MiniGrid-Fetch-16x16-N3-v0"
ENV_TYPE="fetch"
N_EPS=2
SEED=101

echo "=== Zero-shot transfer: DoorKey-16x16 partial → Fetch 16x16 ==="

for CKPT_TAG in \
    "oracle_paid_001_16_partial" \
    "baseline_16_partial"
do
    CKPT="checkpoints/best__${CKPT_TAG}__MiniGrid-DoorKey-16x16-v0.pt"

    if [ ! -f "$CKPT" ]; then
        echo "[skip] $CKPT_TAG — checkpoint not found"
        continue
    fi

    echo "--- $CKPT_TAG ---"

    NO_ORACLE=""
    if [[ "$CKPT_TAG" == "baseline_16_partial" ]]; then
        NO_ORACLE="--no-oracle"
    fi

    python eval.py \
        --checkpoint "$CKPT" \
        --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --partial-obs $NO_ORACLE \
        --n-episodes $N_EPS \
        --seed $SEED \
        --out "figures/gifs/transfer_fetch16_argmax__${CKPT_TAG}.gif"

    python eval.py \
        --checkpoint "$CKPT" \
        --env-id "$ENV_ID" --env-type "$ENV_TYPE" \
        --partial-obs $NO_ORACLE --stochastic \
        --n-episodes $N_EPS \
        --seed $SEED \
        --out "figures/gifs/transfer_fetch16_stoch__${CKPT_TAG}.gif"

    echo ""
done

echo "Done."
