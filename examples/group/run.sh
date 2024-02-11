#!/bin/bash

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
