#!/bin/bash
#SBATCH --job-name=PhysicsNeMo_Test
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load required modules
ml load mpi

# Set environment variables
export OMP_NUM_THREADS=12
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate Python environment
source ~/python/bin/activate

# Install PhysicsNeMo if not already installed
python -c "import physicsnemo" 2>/dev/null || {
    echo "Installing PhysicsNeMo..."
    pip install nvidia-physicsnemo
    pip install -r requirements.txt
}

echo "Test job start: $(date)"
echo "Running PhysicsNeMo PINN testing on 1 GPU"

# Run PhysicsNeMo testing
time python physicsnemo_test.py

echo "Test job end: $(date)"