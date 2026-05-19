#!/bin/bash
#SBATCH --job-name=vlm_oracle_test
#SBATCH --output=logs/slurm_vlm_%j.out
#SBATCH --error=logs/slurm_vlm_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

# ── Paramètres ────────────────────────────────────────────────────────────────

# Environnement MiniGrid à utiliser
ENV_ID="MiniGrid-DoorKey-8x8-v0"
ENV_TYPE="doorkey"

# Graine aléatoire
SEED=1

#choisis le vlm que tu veux --> "" pour utiliser le BFS 
VLM_MODEL="qwen2vl" 

# Dossier où les modèles HuggingFace sont stockés
CACHE_DIR="/scratch/izar/lafont/vlm_models"

TOTAL_TIMESTEPS=1000

set -e

echo "=== Job $SLURM_JOB_ID — VLM oracle: $VLM_MODEL ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

source ~/.bashrc

#change si ton env à un autre nom
conda activate bench_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR

mkdir -p logs figures

python train.py \
    --env-id          "$ENV_ID"           \
    --env-type        "$ENV_TYPE"         \
    --seed            "$SEED"             \
    --total-timesteps "$TOTAL_TIMESTEPS"  \
    --n-envs          1                   \
    --vlm-model       "$VLM_MODEL"        \
    --cache-dir       "$CACHE_DIR"        \
    --exp-name        "test_vlm_${VLM_MODEL}"

echo "End: $(date)"
