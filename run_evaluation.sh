#!/bin/bash
#SBATCH --job-name=scgpt_eval
#SBATCH --account=gendood-preclinomics
#SBATCH --qos=bbgpu
#SBATCH --output=logs/scgpt_eval_%j.out
#SBATCH --error=logs/scgpt_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yxd504@student.bham.ac.uk

# ============================================
# scGPT Evaluation Script - Test on Unseen Data
# ============================================

set -e

echo "============================================"
echo "scGPT Model Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "============================================"

# Setup
SCGPT_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT"
cd "$SCGPT_DIR"
mkdir -p logs

# Load modules
module purge
module load bluebear
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate virtual environment
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_gpu"
source "$VENV_DIR/bin/activate"

# Set environment
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo ""
echo "Python: $(python --version)"
echo ""

# Run evaluation
python evaluate_scgpt.py

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "End Time: $(date)"
echo "============================================"

deactivate

