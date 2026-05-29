#!/bin/bash
#SBATCH --job-name=eval_stats
#SBATCH --output=logs/slurm_eval_stats_%j.out
#SBATCH --error=logs/slurm_eval_stats_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

set -e

source ~/.bashrc
conda activate nanofm

cd $SLURM_SUBMIT_DIR
mkdir -p figures logs

CKPT_ORACLE="checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt"
CKPT_BASE="checkpoints/best__baseline_16_partial__MiniGrid-DoorKey-16x16-v0.pt"
ENV="MiniGrid-Fetch-16x16-N3-v0"
CSV="figures/fetch_comparison.csv"
OUT="figures/fetch_comparison.png"
N=100
SEED=200

# Reset CSV
rm -f $CSV

echo "=== 1/3 Oracle 0.01 — argmax ==="
python eval_transfer_stats.py \
    --checkpoint "$CKPT_ORACLE" --env-id "$ENV" --env-type fetch \
    --partial-obs --n-episodes $N --seed $SEED \
    --label "Oracle 0.01 (argmax)" \
    --csv-out $CSV --out $OUT

echo "=== 2/3 Oracle 0.01 — stochastic ==="
python eval_transfer_stats.py \
    --checkpoint "$CKPT_ORACLE" --env-id "$ENV" --env-type fetch \
    --partial-obs --stochastic --n-episodes $N --seed $SEED \
    --label "Oracle 0.01 (stoch)" \
    --csv-out $CSV --out $OUT

echo "=== 3/3 Baseline PPO ==="
python eval_transfer_stats.py \
    --checkpoint "$CKPT_BASE" --env-id "$ENV" --env-type fetch \
    --partial-obs --no-oracle --n-episodes $N --seed $SEED \
    --label "Baseline PPO" \
    --csv-out $CSV --out $OUT

echo "Done. Plot saved to $OUT"
