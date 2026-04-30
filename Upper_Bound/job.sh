#!/bin/bash
#SBATCH --job-name=ub_${ENV_TYPE}_s${SEED}
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
# ENV_ID, ENV_TYPE, SEED

set -e

echo "=== Job $SLURM_JOB_ID — env=$ENV_ID seed=$SEED ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# ── Environnement ─────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate nanofm

# ── Dépendances manquantes ────────────────────────────────────────────────────
pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR

mkdir -p logs figures

# ── Lancement ─────────────────────────────────────────────────────────────────
python train.py \
    --env-id    "$ENV_ID"   \
    --env-type  "$ENV_TYPE" \
    --seed      "$SEED"     \
    --total-timesteps 500000 \
    $EXTRA_ARGS

echo "End: $(date)"
