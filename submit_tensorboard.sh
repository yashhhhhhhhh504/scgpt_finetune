#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --account=gendood-preclinomics
#SBATCH --qos=gendood
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/tensorboard_%j.out
#SBATCH --error=logs/tensorboard_%j.err

# ============================================
# TensorBoard Server for BlueBEAR
# University of Birmingham HPC
# ============================================

echo "============================================"
echo "TensorBoard Server Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================"

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load bluebear
module load Python/3.10.8-GCCcore-12.2.0

# Activate virtual environment
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_tf"
source "$VENV_DIR/bin/activate"

# Install tensorboard if needed
pip install --quiet tensorboard

# Get the node hostname
NODE_HOSTNAME=$(hostname)
PORT=6006

echo ""
echo "============================================"
echo "TensorBoard is starting..."
echo ""
echo "To access TensorBoard, create an SSH tunnel from your local machine:"
echo ""
echo "  ssh -L ${PORT}:${NODE_HOSTNAME}:${PORT} YOUR_USERNAME@bluebear.bham.ac.uk"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:${PORT}"
echo ""
echo "============================================"
echo ""

# Start TensorBoard
tensorboard --logdir=/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT/save/logs \
            --host=0.0.0.0 \
            --port=${PORT} \
            --bind_all

deactivate

