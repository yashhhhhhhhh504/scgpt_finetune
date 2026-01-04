#!/bin/bash
#SBATCH --job-name=scgpt_train
#SBATCH --account=gendood-preclinomics
#SBATCH --qos=gendood
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/scgpt_train_%j.out
#SBATCH --error=logs/scgpt_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yxd504@student.bham.ac.uk

# ============================================
# scGPT Training Script for BlueBEAR
# University of Birmingham HPC
# ============================================

echo "============================================"
echo "scGPT Training Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust versions as available on BlueBEAR)
module purge
module load bluebear
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Set up virtual environment (node-specific to handle different CPU architectures)
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_tf-${BB_CPU}"
SCGPT_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Set pip cache to scratch (avoid quota issues)
export PIP_CACHE_DIR="/scratch/${USER}/pip"
mkdir -p "$PIP_CACHE_DIR"

# Install/update requirements
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow[and-cuda]
pip install --quiet scanpy anndata scipy numpy pandas
pip install --quiet matplotlib seaborn tqdm scikit-learn
pip install --quiet umap-learn leidenalg

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navigate to scGPT directory
cd "$SCGPT_DIR"

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run training
echo "Starting scGPT training..."
python trainscgpt.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "Training failed with error code $?"
    echo "End Time: $(date)"
    echo "============================================"
    exit 1
fi

# Deactivate virtual environment
deactivate

