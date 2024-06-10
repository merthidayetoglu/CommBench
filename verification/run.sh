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

#export MPICH_OFI_NIC_VERBOSE=2
# export MPICH_ENV_DISPLAY=1

export GASNET_OFI_DEVICE_TYPE=Node
export GASNET_OFI_DEVICE_0=cxi0
export GASNET_OFI_DEVICE_1=cxi1
export GASNET_OFI_DEVICE_2=cxi2
export GASNET_OFI_DEVICE_3=cxi3
# export GASNET_OFI_LIST_DEVICES=1
# export GASNET_SPAWN_VERBOSE=1

export SLURM_CPU_BIND="cores"

# export GASNET_BACKTRACE=1

warmup=5
numiter=10

for library in 2
# 1: MPI
# 2: XCCL
# 3: IPC (PUT)
# 4: IPC (GET)
# 5: GEX (PUT)
# 6: GEX (GET)
do
for pattern in 1 2 3 5 6
# 1: Gather
# 2: Scatter
# 3: Broadcast
# 4: Reduce
# 5: All-to-all
# 6: All-gather
# 7: Reduce-scatter
# 8: All-reduce
do
for size in 22
do
  count=$((2**size))
  srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=4 -C gpu -c 32 --gpus-per-task=1  --gpu-bind=none ./CommBench $library $pattern $count $warmup $numiter
done
done
done

date
