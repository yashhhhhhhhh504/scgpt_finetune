#!/bin/bash
#SBATCH --job-name=scgpt_gpu
#SBATCH --account=gendood-preclinomics
#SBATCH --qos=bbgpu
#SBATCH --output=logs/scgpt_gpu_%j.out
#SBATCH --error=logs/scgpt_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yxd504@student.bham.ac.uk

# ============================================
# scGPT Training Script for BlueBEAR (GPU)
# University of Birmingham HPC
# Using bear-apps/2023a for CUDA 12.1.1
# ============================================

set -e

echo "============================================"
echo "scGPT Training Job Started (GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "============================================"

# Create directories
SCGPT_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/model_1A/scGPT"
cd "$SCGPT_DIR"
mkdir -p logs
mkdir -p save

# Load required modules (bear-apps/2023a/live for CUDA 12.1.1)
module purge
module load bluebear
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Set up virtual environment (GPU-specific)
VENV_DIR="/rds/projects/g/gendood-preclinomics/LLM_finetuning/venv_scgpt_gpu"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating GPU virtual environment at $VENV_DIR..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Set pip cache to scratch (avoid quota issues)
export PIP_CACHE_DIR="/scratch/${USER}/pip"
mkdir -p "$PIP_CACHE_DIR"

# Install requirements
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow[and-cuda]
pip install --quiet scanpy anndata scipy numpy pandas
pip install --quiet matplotlib seaborn tqdm scikit-learn
pip install --quiet umap-learn leidenalg

# Set TensorFlow/CUDA environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$EBROOTCUDA

echo ""
echo "Python: $(python --version)"
echo "CUDA: $EBROOTCUDA"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
echo ""
echo "GPU Information:"
nvidia-smi
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow GPUs: {gpus}')"
echo ""

# Run training
echo "============================================"
echo "Starting scGPT Training (GPU)..."
echo "============================================"
python trainscgpt.py

echo ""
echo "============================================"
echo "Training completed!"
echo "End Time: $(date)"
echo "============================================"

deactivate
