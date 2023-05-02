#!/bin/bash

#PBS -l select=2:system=sunspot,place=scatter
#PBS -A CSC249ADCD01_CNDA
#PBS -l walltime=00:30:00
#PBS -N 2nodes
#PBS -k doe

export TZ='/usr/share/zoneinfo/US/Central'
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8
unset OMP_PLACES

cd ~/CommBench
date

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=12            # Number of MPI ranks per node
NDEPTH=16            # Number of hardware threads per rank, spacing between MPI ranks on a node
NTHREADS=$OMP_NUM_THREADS # Number of OMP threads per rank, given to OMP_NUM_THREADS

export MPICH_GPU_SUPPORT_ENABLED=1

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_NODES=${NNODES}  TOTAL_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}  THREADS_PER_RANK=${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"

warmup=5
numiter=10

for library in 1
do
for pattern in 1 2 3
do
for direction in 1 2
do
for p in 24
do
for g in 12
do
for k in 1 12
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 10000000
do
  mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth gpu_tile_compact.sh ./CommBench $library $pattern $direction $count $warmup $numiter $p $g $k
done
done
done
done
done
done
done

date
