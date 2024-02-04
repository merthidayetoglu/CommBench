#!/bin/bash
#SBATCH -A m4301
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

date

module -t list

export MPICH_OFI_NIC_VERBOSE=2
export MPICH_ENV_DISPLAY=1

export SLURM_CPU_BIND="cores"
gpupernode=4
numnode=4

srun -N 2 --ntasks-per-node=4 -C gpu -c 32 --gpus-per-task=1  --gpu-bind=none python3 test.py



date
