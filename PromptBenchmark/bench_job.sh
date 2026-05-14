#!/bin/bash
#SBATCH --job-name=prompt_bench
#SBATCH --time=04:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/lafont/cs503_project/cs503_project/PromptBenchmark/logs/benchmark.out
#SBATCH --error=/home/lafont/cs503_project/cs503_project/PromptBenchmark/logs/benchmark.err

echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Start time    : $(date)"

source /home/lafont/miniconda3/etc/profile.d/conda.sh
conda activate bench_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_HOME=/scratch/izar/lafont/vlm_models
export HUGGINGFACE_HUB_CACHE=/scratch/izar/lafont/vlm_models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PROJECT_DIR=/home/lafont/cs503_project/cs503_project/PromptBenchmark
DATASET=$PROJECT_DIR/trajectory_dataset/trajectory_dataset.json
RESULTS=$PROJECT_DIR/results

mkdir -p $PROJECT_DIR/logs
mkdir -p $RESULTS

cd $PROJECT_DIR

python run_benchmark.py \
    --dataset   $DATASET \
    --results   $RESULTS \
    --cache_dir $HF_HOME \
    --models    internvl3 \
    --views     global \
    #--max_samples 5 