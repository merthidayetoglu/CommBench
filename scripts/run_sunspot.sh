#!/bin/bash

#PBS -l select=32:system=sunspot,place=scatter
#PBS -A MyProjectAllocationName
#PBS -l walltime=01:00:00
#PBS -N 32NodeRunExample
#PBS -k doe

export TZ='/usr/share/zoneinfo/US/Central'
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8
unset OMP_PLACES

cd .

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

for g in 1
do
for k in 1
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 268435456
do
  mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth ~/CommBench/gpu_tile_compact.sh ./Alltoall 1 $count 10 20 $g $k
  #mpiexec --pmi pmix -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -envall ~/CommBench/wrapper_spread.sh ./Alltoall 1 $count 10 20 $g $k
  #mpiexec --pmi pmix -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -envall ~/CommBench/gpu_tile_compact.sh ./Alltoall 1 $count 10 20 $g $k
  #mpiexec --pmi pmix -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth ./Alltoall 1 $count 10 20 $g $k
  #mpiexec --pmi pmix -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth ./Alltoall 1 $count 10 20 $g $k
  #mpiexec --pmi pmix -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth ~/CommBench/wrapper_spread.sh ./Alltoall 1 $count 10 20 $g $k
  #mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -envall gpu_tile_compact.sh ./Alltoall 1 $count 10 20 $g $k
  #mpiexec -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth ./Alltoall 1 $count 10 20 $g $k
done
done
done

