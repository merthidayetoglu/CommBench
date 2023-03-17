#!/bin/bash

gpu_per_node=6
nstack_per_gpu=1

gpu=$((MPI_LOCALRANKID%gpu_per_node))

stack=$((MPI_LOCALRANKID/gpu_per_node%nstack_per_gpu))

export ZE_AFFINITY_MASK=$gpu.$stack

echo MPI_LOCALRANKID = $MPI_LOCALRANKID  ZE_AFFINITY_MASK = $ZE_AFFINITY_MASK
exec $@
