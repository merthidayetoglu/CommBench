#!/bin/bash

#PBS -l select=2:system=sunspot,place=scatter
#PBS -A CSC249ADCD01_CNDA
#PBS -l walltime=01:00:00
#PBS -N 2nodes_gpu
#PBS -k doe

export TZ='/usr/share/zoneinfo/US/Central'
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
#unset OMP_PLACES

date

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=12            # Number of MPI ranks per node
NDEPTH=8            # Number of hardware threads per rank, spacing between MPI ranks on a node
# NTHREADS=$OMP_NUM_THREADS # Number of OMP threads per rank, given to OMP_NUM_THREADS

export MPICH_GPU_SUPPORT_ENABLED=1

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_NODES=${NNODES}  TOTAL_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}  THREADS_PER_RANK=${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"

export MPIR_CVAR_CH4_OFI_ENABLE_GPU_PIPELINE=1
PROC_LIST='list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99'

warmup=5
numiter=20

for library in 1
do
for pattern in 1
do
for direction in 0
do
for n in 2
do
for g in 12
do
# for size in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
for size in 20
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for k in 8
do
  count=$((2**size))
  mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} --cpu-bind=$PROC_LIST gpu_tile_compact.sh ./cxi_assign_rr.sh ./CommBench $library $pattern $direction $count $warmup $numiter $n $g $k
done
done
done
done
done
done
done

date
