#!/bin/bash

#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=2
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

warmup=5
numiter=10

for library in 1
do
for direction in 1 2
do
for pattern in 1 2 3
do
for p in 8
do
for g in 4
do
for k in 1 4
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 10000000
do
  srun ./CommBench $library $pattern $direction $count $warmup $numiter $p $g $k
done
done
done
done
done
done
done

date
