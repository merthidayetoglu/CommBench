#!/bin/bash -l

#PBS -N AFFINITY
#PBS -l select=2:ncpus=256
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A GRACE

date

module -t list

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=4 # Number of MPI ranks to spawn per node
NDEPTH=16 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=16 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

warmup=5
numiter=10

for library in 1
do
for direction in 1 2
do
for pattern in 1 2 3
do
for p in 8
do
for g in 4
do
for k in 1 4
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 10000000
do
  mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth -env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh ./CommBench $library $pattern $direction $count $warmup $numiter $p $g $k
done
done
done
done
done
done
done

date
