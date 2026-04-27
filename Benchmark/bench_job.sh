#!/bin/bash
#SBATCH --job-name=vlm_benchmark
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/bultez/cs503_project/cs503_project/Benchmark/logs/benchmark.out
#SBATCH --error=/home/bultez/cs503_project/cs503_project/Benchmark/logs/benchmark.err

# ── Environment ───────────────────────────────────────────────────────────────
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Start time    : $(date)"
echo "GPU(s)        : $CUDA_VISIBLE_DEVICES"

# Activate conda environment
source /home/bultez/miniconda3/etc/profile.d/conda.sh
conda activate bench_env

# HuggingFace cache → scratch (models already downloaded here)
export HF_HOME=/scratch/izar/bultez/vlm_models
export HUGGINGFACE_HUB_CACHE=/scratch/izar/bultez/vlm_models

# Avoid HF trying to reach the internet on compute nodes
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Project paths
PROJECT_DIR=/home/bultez/cs503_project/cs503_project/Benchmark
DATASET=$PROJECT_DIR/dataset/dataset.json
RESULTS=$PROJECT_DIR/results
UTILS=$PROJECT_DIR

# Create output dirs if they don't exist
mkdir -p $PROJECT_DIR/logs
mkdir -p $RESULTS

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Starting VLM benchmark"
echo "=========================================="

cd $UTILS

python run_benchmark.py \
    --dataset   $DATASET \
    --results   $RESULTS \
    --cache_dir $HF_HOME \
    --models    qwen3b qwen7b internvl smolvlm \
    --views     global\
    # --max_samples 5\
    # --debug 

echo ""
echo "=========================================="
echo "  Benchmark complete : $(date)"
echo "=========================================="