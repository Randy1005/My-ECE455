#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=matmul_naive.output

module load nvidia/cuda
nvcc matmul_naive.cu -o matmul_naive
./matmul_naive