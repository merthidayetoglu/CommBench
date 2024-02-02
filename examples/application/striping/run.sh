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

gpupernode=4
numnode=4
numstripe=4
numrhs=3
srun -N 4 --ntasks-per-node=4 -C gpu -c 32 --gpus-per-task=1  --gpu-bind=none ./CommBench $gpupernode $numnode $numstripe examples/application/16x16_pattern.txt $numrhs
exit

warmup=5
numiter=10

for library in 1 2 3
do
for direction in 1
do
for pattern in 0
do
for p in 16
do
for g in 1
do
for k in 1
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 1 #134217728
do
  srun -N 4 --ntasks-per-node=4 -C gpu -c 32 --gpus-per-task=1  --gpu-bind=none ./CommBench $library $pattern $direction $count $warmup $numiter $p $g $k
done
done
done
done
done
done
done

date
