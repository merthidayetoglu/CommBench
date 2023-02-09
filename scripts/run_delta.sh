#!/bin/bash

#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbkf-delta-gpu
#SBATCH --time=00:30:00
### GPU options ###
#SBATCH --gpus-per-node=4

date

scontrol show job ${SLURM_JOBID}

module -t list

#for count in 1 10 100 1000 10000 100000 1000000 10000000
#for count in 1 4 16 64 256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456
for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
do

srun Alltoall $count 30 4

done

date
