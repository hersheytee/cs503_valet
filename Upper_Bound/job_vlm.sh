#!/bin/bash
#SBATCH --job-name=vlm_sweep
#SBATCH --output=logs/slurm_vlm_%j.out
#SBATCH --error=logs/slurm_vlm_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=ee-452

# ── Paramètres ────────────────────────────────────────────────────────────────

ENV_ID="MiniGrid-DoorKey-8x8-v0"
ENV_TYPE="doorkey"             # empty | doorkey | fetch | gotodoor | gotoobject

# VLM_MODEL : clé du modèle (voir download_models.py)
#   qwen2vl | qwen3b | qwen7b | internvl | internvl3 | internvl8b_mpo | wethink | smolvlm
#   "" → BFS oracle (upper bound, aucun VLM)
VLM_MODEL="wethink"

# Mode de prompt VLM : baseline | cot | thinking
VLM_MODE="baseline"

# Sweep : coûts oracle et seeds à évaluer
ORACLE_COSTS=(0.01 0.02)
SEEDS=(4 5 6 7 8)

TOTAL_TIMESTEPS=500000
CACHE_DIR="/scratch/izar/bultez/vlm_models"  # adapter à ton username

# Flags supplémentaires passés à train.py (décommenter selon le besoin)
EXTRA=""
# EXTRA="--partial-obs --large-model"

# ── Setup ─────────────────────────────────────────────────────────────────────

set -e

N_COSTS=${#ORACLE_COSTS[@]}
N_SEEDS=${#SEEDS[@]}
N_RUNS=$((N_COSTS * N_SEEDS))

echo "================================================================"
echo " Job $SLURM_JOB_ID — VLM sweep"
echo "   model      : ${VLM_MODEL:-BFS}   mode: $VLM_MODE"
echo "   env        : $ENV_ID  (type=$ENV_TYPE)"
echo "   costs      : ${ORACLE_COSTS[*]}"
echo "   seeds      : ${SEEDS[*]}"
echo "   total runs : $N_RUNS   timesteps/run: $TOTAL_TIMESTEPS"
echo "   node       : $SLURMD_NODENAME"
echo "   start      : $(date)"
echo "================================================================"

source ~/.bashrc
conda activate bench_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR
mkdir -p logs figures checkpoints

# ── Gestion du serveur VLM ────────────────────────────────────────────────────

SERVER_PID=""
MODEL_REPO=""
MODEL_DTYPE=""
MODEL_GMEM=""

_stop_vlm_server() {
    [ -z "$SERVER_PID" ] && return
    echo ">>> Stopping vLLM server (PID $SERVER_PID) ..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for i in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo ">>> Server stopped."; SERVER_PID=""; return; }
        sleep 1
    done
    echo ">>> Graceful shutdown timed out — force killing ..."
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo ">>> Server killed."
    SERVER_PID=""
}

_start_vlm_server() {
    _stop_vlm_server

    echo ">>> Starting vLLM server for $VLM_MODEL ..."
    LOG_FILE="logs/vllm_server_${SLURM_JOB_ID}.log"
    HF_HOME="$CACHE_DIR" HUGGINGFACE_HUB_CACHE="$CACHE_DIR" \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python -m vllm.entrypoints.openai.api_server \
        --model                  "$MODEL_REPO"  \
        --dtype                  "$MODEL_DTYPE" \
        --gpu-memory-utilization "$MODEL_GMEM"  \
        --host 127.0.0.1 --port 8000            \
        --trust-remote-code                     \
        --max-model-len 4096                    \
        --enforce-eager                         \
        --download-dir "$CACHE_DIR"             \
        >> "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    echo ">>> vLLM server PID: $SERVER_PID — waiting for /health (up to 15 min) ..."
    python -c "
from vlm_oracle import wait_for_server
import sys
ok = wait_for_server(timeout=900)
sys.exit(0 if ok else 1)
" && echo ">>> vLLM server ready." || { echo "!!! Server failed to start."; return 1; }
}

_check_or_restart_server() {
    # Vérifie si le serveur répond ; le redémarre si non
    if python -c "
from vlm_oracle import wait_for_server
import sys
sys.exit(0 if wait_for_server(timeout=15) else 1)
" 2>/dev/null; then
        return 0
    fi
    echo "!!! vLLM server unreachable — restarting (attempt $1 of 3) ..."
    _start_vlm_server
}

if [ -n "$VLM_MODEL" ]; then
    MODEL_REPO=$(python -c "from download_models import MODELS; print(MODELS['$VLM_MODEL']['repo_id'])")
    MODEL_DTYPE=$(python -c "from download_models import MODELS; print(MODELS['$VLM_MODEL']['dtype'])")
    MODEL_GMEM=$(python -c "from download_models import MODELS; print(MODELS['$VLM_MODEL']['gpu_mem'])")
    _start_vlm_server
fi

# ── Baseline (--no-oracle) — une run par seed, avant le sweep VLM ─────────────

RUN=0
FAILED=0

# echo ""
# echo "════════════════════════════════════════════════════════════"
# echo " BASELINE RUNS (pure PPO, no oracle)"
# echo "════════════════════════════════════════════════════════════"

# for SEED in "${SEEDS[@]}"; do
#     RUN=$((RUN + 1))
#     EXP_NAME="baseline_seed${SEED}"

#     echo ""
#     echo "────────────────────────────────────────────────────────────"
#     echo " Baseline $SEED / ${#SEEDS[@]} — seed=$SEED   $(date)"
#     echo "────────────────────────────────────────────────────────────"

#     python train.py \
#         --env-id          "$ENV_ID"          \
#         --env-type        "$ENV_TYPE"        \
#         --seed            "$SEED"            \
#         --total-timesteps "$TOTAL_TIMESTEPS" \
#         --n-envs          8                  \
#         --no-oracle                          \
#         --exp-name        "$EXP_NAME"        \
#         $EXTRA \
#     || { echo "!!! Baseline seed=$SEED FAILED"; FAILED=$((FAILED + 1)); }
# done

# ── Sweep costs × seeds ───────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════"
echo " ORACLE SWEEP (${VLM_MODEL:-BFS})"
echo "════════════════════════════════════════════════════════════"

for ORACLE_COST in "${ORACLE_COSTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        COST_TAG="${ORACLE_COST/./}"   # 0.02 → 002

        if [ -z "$VLM_MODEL" ]; then
            EXP_NAME="bfs_cost${COST_TAG}_seed${SEED}"
        else
            EXP_NAME="vlm_${VLM_MODEL}_${VLM_MODE}_cost${COST_TAG}_seed${SEED}"
        fi

        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo " Run $RUN / $N_RUNS — cost=$ORACLE_COST  seed=$SEED"
        echo " exp: $EXP_NAME   $(date)"
        echo "────────────────────────────────────────────────────────────"

        if [ -z "$VLM_MODEL" ]; then
            python train.py \
                --env-id          "$ENV_ID"          \
                --env-type        "$ENV_TYPE"        \
                --seed            "$SEED"            \
                --total-timesteps "$TOTAL_TIMESTEPS" \
                --n-envs          8                  \
                --oracle-cost     "$ORACLE_COST"     \
                --exp-name        "$EXP_NAME"        \
                $EXTRA \
            || { echo "!!! Run $RUN FAILED"; FAILED=$((FAILED + 1)); }
        else
            # Health check : redémarre le serveur s'il a crashé
            RESTART_OK=true
            for attempt in 1 2 3; do
                _check_or_restart_server $attempt && break
                if [ "$attempt" -eq 3 ]; then
                    echo "!!! Server could not be restarted after 3 attempts — skipping run $RUN"
                    RESTART_OK=false
                fi
            done

            if $RESTART_OK; then
                python train.py \
                    --env-id          "$ENV_ID"          \
                    --env-type        "$ENV_TYPE"        \
                    --seed            "$SEED"            \
                    --total-timesteps "$TOTAL_TIMESTEPS" \
                    --n-envs          8                  \
                    --oracle-cost     "$ORACLE_COST"     \
                    --vlm-model       "$VLM_MODEL"       \
                    --vlm-mode        "$VLM_MODE"        \
                    --cache-dir       "$CACHE_DIR"       \
                    --vlm-no-start                       \
                    --exp-name        "$EXP_NAME"        \
                    $EXTRA \
                || { echo "!!! Run $RUN FAILED"; FAILED=$((FAILED + 1)); }
            else
                FAILED=$((FAILED + 1))
            fi
        fi

        echo " Run $RUN done — $(date)"
    done
done

# ── Arrêt du serveur VLM ──────────────────────────────────────────────────────

if [ -n "$SERVER_PID" ]; then
    echo ""
    _stop_vlm_server
fi

# ── Résumé ────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
N_TOTAL=$((N_RUNS + ${#SEEDS[@]}))
echo " Sweep complete : $((RUN - FAILED)) / $N_TOTAL runs succeeded  (${#SEEDS[@]} baseline + $N_RUNS oracle)"
if [ "$FAILED" -gt 0 ]; then
    echo " WARNING: $FAILED run(s) failed — check logs above"
fi
echo " End: $(date)"
echo "================================================================"
