#!/bin/bash

# Begin LSF Directives
#BSUB -P CHM137
#BSUB -W 00:40
##BSUB -q debug
#BSUB -nnodes 4
#BSUB -alloc_flags gpudefault

module -t list

date

number_of_nodes=$(cat $LSB_DJOB_HOSTFILE | uniq | head -n -1 | wc -l)

# JSRUN OPTIONS (CONFIGURE ME!)
number_of_resource_sets=24
resource_sets_per_node=6
physical_cores_per_resource_set=7
gpus_per_resource_set=1
mpi_ranks_per_resource_set=1
phyiscal_cores_per_mpi_rank=7

export OMP_NUM_THREADS=7

export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"

for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
do
jsrun --smpiargs="-gpu" \
          -n ${number_of_resource_sets}               \
          -r ${resource_sets_per_node}                \
          -c ${physical_cores_per_resource_set}       \
          -g ${gpus_per_resource_set}                 \
          -a ${mpi_ranks_per_resource_set}            \
          -bpacked:${physical_cores_per_resource_set} \
           js_task_info ./Alltoall $count 30 6
done

date
