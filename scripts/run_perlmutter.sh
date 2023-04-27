#!/bin/bash
#SBATCH -A m4301
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

date

module -t list

export MPICH_OFI_NIC_VERBOSE=2
export MPICH_ENV_DISPLAY=1

export SLURM_CPU_BIND="cores"

warmup=5
numiter=10

for library in 1 2 #0:IPC 1:MPI 2:NCCL
do
for direction in 1 2 #1:uni-directional 2:bi-directional 3:omni-directional
do
for pattern in 1 2 3 #0:P2P 1:RAIL 2:DENSE 3:FAN
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
  srun -N 2 --ntasks-per-node=4 -C gpu -c 32 --gpus-per-task=1  --gpu-bind=none ./CommBench $library $pattern $direction $count $warmup $numiter $p $g $k
done
done
done
done
done
done
done

date
