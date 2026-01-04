#!/bin/bash
#SBATCH --job-name=scgpt_train_cpu
#SBATCH --output=logs/scgpt_train_%j.out
#SBATCH --error=logs/scgpt_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yxd504@student.bham.ac.uk

# ============================================
# scGPT Training Script for BlueBEAR (CPU only)
# Use this if GPU queue (bbgpu) is busy
# ============================================

set -e

echo "============================================"
echo "scGPT Training Job Started (CPU Mode)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128GB"
echo "Start Time: $(date)"
echo "============================================"

# Create directories
mkdir -p logs
mkdir -p save

# Load required modules
module purge
module load bluebear
module load bear-apps/2022b
module load Python/3.10.8-GCCcore-12.2.0

# Set up virtual environment path (node-specific)
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_tf-${BB_CPU}"
SCGPT_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Set pip cache to scratch
export PIP_CACHE_DIR="/scratch/${USER}/pip"
mkdir -p "$PIP_CACHE_DIR"

# Install requirements
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow-cpu
pip install --quiet scanpy anndata scipy numpy pandas
pip install --quiet matplotlib seaborn tqdm scikit-learn
pip install --quiet umap-learn leidenalg

# Navigate to scGPT directory
cd "$SCGPT_DIR"

# Set TensorFlow to use all CPUs
export TF_NUM_INTEROP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "Python: $(python --version)"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
echo ""

# Run training
echo "============================================"
echo "Starting scGPT Training (CPU)..."
echo "============================================"
python trainscgpt.py

echo ""
echo "============================================"
echo "Training completed!"
echo "End Time: $(date)"
echo "============================================"

deactivate
