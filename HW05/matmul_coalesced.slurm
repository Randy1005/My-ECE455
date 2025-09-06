#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=matmul_coalesced.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda
nvcc matmul_coalesced.cu -o matmul_coalesced
./matmul_coalesced
