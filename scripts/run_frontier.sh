#!/bin/bash

#SBATCH -A CHM137
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest

module -t list

date

export OMP_NUM_THREADS=7

# export MPICH_OFI_NIC_VERBOSE=2
# export MPICH_ENV_DISPLAY=1
# export NCCL_DEBUG=INFO
# export HSA_ENABLE_SDMA=0


# to make RCCL work
export LD_LIBRARY_PATH=/ccs/home/merth/HiCCL/CommBench/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
export NCCL_NET_GDR_LEVEL=3

warmup=5
numiter=10

for library in 2
do
for pattern in 1
do
for direction in 0
do
for n in 2
do
for g in 8
do
for k in 8
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
count=$((2 ** 24))
  srun -c7 ./CommBench $library $pattern $direction $count $warmup $numiter $n $g $k
done
done
done
done
done
done

date
