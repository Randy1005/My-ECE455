#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=reduce_warp_cg.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda

# Compile with modern C++ standard (required for CUDA 13+ cooperative_groups)
nvcc -O3 -std=c++17 reduce_warp_cg.cu -o reduce_warp_cg
./reduce_warp_cg
