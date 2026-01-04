#!/bin/bash
# ============================================
# One-time setup script for scGPT on BlueBEAR
# Run this ONCE before submitting jobs
# ============================================

echo "Setting up scGPT environment on BlueBEAR..."

# Load modules
module purge
module load bluebear
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Set paths
PROJECT_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning"
VENV_DIR="${PROJECT_DIR}/venv_scgpt_tf"
SCGPT_DIR="${PROJECT_DIR}/scGPT"

# Create directories
mkdir -p "${PROJECT_DIR}"
mkdir -p "${SCGPT_DIR}/logs"
mkdir -p "${SCGPT_DIR}/save"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with GPU support
echo "Installing TensorFlow..."
pip install tensorflow[and-cuda]

# Install other requirements
echo "Installing requirements..."
cd "$SCGPT_DIR"
pip install -r requirements.txt

# Install additional packages
pip install umap-learn matplotlib seaborn

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Submit training job: sbatch submit_training.sh"
echo "2. Monitor job: squeue -u \$USER"
echo "3. View logs: tail -f logs/scgpt_train_*.out"
echo "============================================"

deactivate


