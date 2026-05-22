#!/bin/bash
#SBATCH --job-name=noise_sweep
#SBATCH --output=logs/slurm_noise_%j.out
#SBATCH --error=logs/slurm_noise_%j.err
#SBATCH --time=35:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503

# ── Paramètres ────────────────────────────────────────────────────────────────

ENV_ID="MiniGrid-DoorKey-8x8-v0"
ENV_TYPE="doorkey"

# Niveaux de bruit à comparer
# 1.0 = oracle parfait (BFS pur)
# 0.75 / 0.50 / 0.25 = oracle bruité
# 0.0  = oracle complètement aléatoire (pire cas)
NOISE_LEVELS=(0.75 0.5 0.25)

SEEDS=(4 5 6 7 8)
ORACLE_COSTS=(0.01 0.02 0.03)
TOTAL_TIMESTEPS=500000
EXTRA=""

# ── Setup ─────────────────────────────────────────────────────────────────────

set -e

N_NOISE=${#NOISE_LEVELS[@]}
N_SEEDS=${#SEEDS[@]}
N_COSTS=${#ORACLE_COSTS[@]}
N_RUNS=$((N_NOISE * N_SEEDS * N_COSTS))

echo "================================================================"
echo " Job $SLURM_JOB_ID — Noisy Oracle sweep"
echo "   env         : $ENV_ID"
echo "   noise levels: ${NOISE_LEVELS[*]}"
echo "   oracle costs: ${ORACLE_COSTS[*]}"
echo "   seeds        : ${SEEDS[*]}"
echo "   total runs   : $N_RUNS   timesteps/run: $TOTAL_TIMESTEPS"
echo "   node         : $SLURMD_NODENAME"
echo "   start        : $(date)"
echo "================================================================"

source ~/.bashrc
conda activate bench_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install -q -r requirements.txt

cd $SLURM_SUBMIT_DIR
mkdir -p logs figures checkpoints

RUN=0
FAILED=0

# ── Sweep noise × cost × seeds ───────────────────────────────────────────────

for NOISE in "${NOISE_LEVELS[@]}"; do
    NOISE_TAG="${NOISE/./}"          # 0.75 → 075
    NOISE_PCT=$(python -c "print(int(float('$NOISE') * 100))")   # 0.75 → 75

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Noise level : $NOISE  (${NOISE_PCT}% correct)"
    echo "════════════════════════════════════════════════════════════"

    for ORACLE_COST in "${ORACLE_COSTS[@]}"; do
        COST_TAG="${ORACLE_COST/./}"   # 0.1 → 01

        for SEED in "${SEEDS[@]}"; do
            RUN=$((RUN + 1))
            EXP_NAME="noisy_oracle_noise${NOISE_TAG}_cost${COST_TAG}_seed${SEED}"

            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo " Run $RUN / $N_RUNS — noise=$NOISE  cost=$ORACLE_COST  seed=$SEED   $(date)"
            echo " exp: $EXP_NAME"
            echo "────────────────────────────────────────────────────────────"

            python train.py \
                --env-id          "$ENV_ID"          \
                --env-type        "$ENV_TYPE"        \
                --seed            "$SEED"            \
                --total-timesteps "$TOTAL_TIMESTEPS" \
                --n-envs          8                  \
                --oracle-cost     "$ORACLE_COST"     \
                --oracle-noise    "$NOISE"           \
                --exp-name        "$EXP_NAME"        \
                $EXTRA \
            && echo "    OK run $RUN" \
            || { echo "    FAILED run $RUN"; FAILED=$((FAILED + 1)); }
        done
    done
done

# ── Résumé ────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo " Sweep complete : $((N_RUNS - FAILED)) / $N_RUNS runs succeeded"
if [ "$FAILED" -gt 0 ]; then
    echo " WARNING: $FAILED run(s) failed — check logs above"
fi
echo " End: $(date)"
echo "================================================================"
