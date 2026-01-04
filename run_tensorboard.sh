#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --output=logs/tensorboard_%j.out
#SBATCH --error=logs/tensorboard_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# ============================================
# TensorBoard Server for BlueBEAR
# ============================================

set -e

echo "============================================"
echo "TensorBoard Server Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================"

mkdir -p logs

# Load modules
module purge
module load bluebear
module load bear-apps/2022b
module load Python/3.10.8-GCCcore-12.2.0

# Activate virtual environment
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_tf-${BB_CPU}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR"
    echo "Please run training job first to create it."
    exit 1
fi

source "$VENV_DIR/bin/activate"
pip install --quiet tensorboard

# Get node info
NODE_HOSTNAME=$(hostname)
PORT=6006

echo ""
echo "============================================"
echo "TensorBoard is running!"
echo ""
echo "To access from your local machine:"
echo ""
echo "1. Open a NEW terminal on your Mac and run:"
echo "   ssh -L ${PORT}:${NODE_HOSTNAME}:${PORT} yxd504@bluebear.bham.ac.uk"
echo ""
echo "2. Then open in your browser:"
echo "   http://localhost:${PORT}"
echo ""
echo "============================================"
echo ""

# Start TensorBoard
tensorboard --logdir=/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/save/logs \
            --host=0.0.0.0 \
            --port=${PORT}

deactivate
