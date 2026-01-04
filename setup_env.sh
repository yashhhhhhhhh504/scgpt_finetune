#!/bin/bash
# ============================================
# One-time Environment Setup for BlueBEAR
# Run this ONCE before submitting jobs
# Usage: ./setup_env.sh
# ============================================

set -e

echo "============================================"
echo "Setting up scGPT Environment on BlueBEAR"
echo "============================================"

# Load modules
module purge
module load bluebear
module load bear-apps/2022b
module load Python/3.10.8-GCCcore-12.2.0

# Paths
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_tf-${BB_CPU}"
SCGPT_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT"

# Create directories
echo "Creating directories..."
mkdir -p "${SCGPT_DIR}/logs"
mkdir -p "${SCGPT_DIR}/save"
mkdir -p "/scratch/${USER}/pip"

# Create virtual environment
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv --system-site-packages "$VENV_DIR"

# Activate
source "$VENV_DIR/bin/activate"

# Set pip cache
export PIP_CACHE_DIR="/scratch/${USER}/pip"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages
echo "Installing TensorFlow and dependencies..."
pip install tensorflow
pip install scanpy anndata scipy numpy pandas
pip install matplotlib seaborn tqdm scikit-learn
pip install umap-learn leidenalg numba

echo ""
echo "============================================"
echo "Installed packages:"
pip list | grep -E "tensorflow|scanpy|anndata|numpy|pandas"
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Submit GPU training: sbatch run_scgpt.sh"
echo "  2. Or CPU training:     sbatch run_scgpt_cpu.sh"
echo "  3. Monitor job:         squeue -u \$USER"
echo "  4. View logs:           tail -f logs/scgpt_train_*.out"
echo "============================================"

deactivate
