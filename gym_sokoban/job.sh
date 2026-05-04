#!/bin/bash
#SBATCH --job-name=soko_${EXP_NAME}_s${SEED}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

# ── Paramètres (passés via sbatch --export) ───────────────────────────────────
# EXP_NAME, SEED, EXTRA_ARGS

set -e

echo "=== Job $SLURM_JOB_ID — exp=$EXP_NAME seed=$SEED ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# ── Environnement ─────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate nanofm

# ── Dépendances manquantes ────────────────────────────────────────────────────
pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR

mkdir -p logs figures checkpoints

# ── Lancement ─────────────────────────────────────────────────────────────────
python train.py \
    --env-id "Sokoban-small-v0" \
    --seed "$SEED" \
    --exp-name "$EXP_NAME" \
    --total-timesteps 500000 \
    --save-model \
    $EXTRA_ARGS

echo "End: $(date)"