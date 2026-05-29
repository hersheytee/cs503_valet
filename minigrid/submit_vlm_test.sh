#!/bin/bash
#SBATCH --job-name=vlm_test
#SBATCH --output=logs/slurm_vlm_test_%j.out
#SBATCH --error=logs/slurm_vlm_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

# ── Paramètres ────────────────────────────────────────────────────────────────
VLM_MODEL="smolvlm"                              # plus petit modèle (2.2B, 4.5 GB)
CACHE_DIR="/scratch/izar/tjouven/vlm_models"     # adapte si besoin
ENV_ID="MiniGrid-DoorKey-8x8-v0"
ENV_TYPE="doorkey"
SEED=1
TOTAL_TIMESTEPS=2000                             # juste assez pour vérifier que ça tourne

set -e

echo "=== VLM test | model=$VLM_MODEL | node=$SLURMD_NODENAME ==="
echo "Start: $(date)"

source ~/.bashrc
conda activate nanofm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR
mkdir -p logs figures checkpoints

# Vérifier que le modèle est bien téléchargé
python -c "
from download_models import MODELS, is_downloaded
import sys
m = MODELS['$VLM_MODEL']
if not is_downloaded(m['repo_id'], '$CACHE_DIR'):
    print('ERROR: model not downloaded. Run download step first.')
    sys.exit(1)
print(f'Model {m[\"name\"]} found in cache.')
"

python -u train.py \
    --env-id          "$ENV_ID"          \
    --env-type        "$ENV_TYPE"        \
    --seed            "$SEED"            \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-envs          1                  \
    --reward-shaping                     \
    --vlm-model       "$VLM_MODEL"       \
    --cache-dir       "$CACHE_DIR"       \
    --exp-name        "vlm_test_${VLM_MODEL}"

echo "End: $(date)"
