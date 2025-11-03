#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR

# Ensure a clean environment and use the Gen8-compatible compiler
module purge
module load gcc/suse/14

# Show compiler version (for confirmation in output file)
gcc --version

# Compile Taskflow example
g++ condition_task.cpp \
    -std=c++17 -O2 -pthread \
    -I/srv/home/eang2/ECE455lab/lab8/taskflow/taskflow \
    -o condition_task

# Run the compiled binary
./condition_task