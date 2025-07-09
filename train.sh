#!/bin/bash
#SBATCH --job-name=PINNs_train
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=108G
#SBATCH --gres=gpu:2

ml load mpi

export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

source ~/python/bin/activate

echo "Job start: $(date)"
time torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        NSFnet/ev-NSFnet/train.py
echo "Job end: $(date)"
