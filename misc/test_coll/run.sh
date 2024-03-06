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

#export MPICH_OFI_NIC_VERBOSE=2
# export MPICH_ENV_DISPLAY=1
# export NCCL_DEBUG=INFO

#export HSA_ENABLE_SDMA=0

library=1
count=10000000

export LD_LIBRARY_PATH=/ccs/home/merth/HiCCL/CommBench/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
export NCCL_NET_GDR_LEVEL=3

for pattern in 3 4 6 7 8
do
  srun -c7 ./CommBench $library $pattern $count 5 10
done


date
