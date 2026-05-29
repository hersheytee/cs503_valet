#!/bin/bash
# Zero-shot transfer eval: DoorKey-16x16 models → MiniGrid-Fetch-8x8-N3-v0
# Usage: bash eval_transfer.sh

source ~/.bashrc
conda activate nanofm

cd $HOME/Upper_Bound
mkdir -p figures

ENV_ID="MiniGrid-Fetch-16x16-N3-v0"
ENV_TYPE="fetch"
N_EPS=3
SEED=100   # seeds jamais vus pendant le training (training = seeds 4-8)

echo "=== Zero-shot transfer: DoorKey-16x16 partial → Fetch 8x8 ==="
echo ""

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
        --out "figures/transfer_fetch__${CKPT_TAG}.gif"

    echo ""
done

echo "Done. GIFs saved to figures/transfer_fetch__*.gif"
