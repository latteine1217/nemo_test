#!/bin/bash
#SBATCH --job-name=PhysicsNeMo_PINN
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=108G
#SBATCH --gres=gpu:2

# Load required modules
ml load mpi

# Set environment variables for optimal performance
export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

# PhysicsNeMo specific environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate Python environment
source ~/python/bin/activate

# Install PhysicsNeMo if not already installed
python -c "import physicsnemo" 2>/dev/null || {
    echo "Installing PhysicsNeMo..."
    pip install nvidia-physicsnemo
    pip install -r requirements.txt
}

# Create necessary directories
mkdir -p checkpoints
mkdir -p outputs

# Set distributed training environment
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=2
export NPROC_PER_NODE=2

echo "Job start: $(date)"
echo "Running PhysicsNeMo PINN training with 2 GPUs"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"

# Run PhysicsNeMo training with torchrun
time torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        physicsnemo_train.py

echo "Job end: $(date)"