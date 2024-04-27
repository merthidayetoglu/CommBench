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

library=2
# 0: IPC
# 1: MPI
# 2: XCCL
for pattern in 1 2 3 4 5 6 7 8
# 1: Gather
# 2: Scatter
# 3: Broadcast
# 4: Reduce
# 5: All-to-all
# 6: All-gather
# 7: Reduce-scatter
# 8: All-reduce
do
for size in 0
do
  count=$((2**size))$
  srun -c7 ./CommBench $library $pattern $count $warmup $numiter
done
done

date
