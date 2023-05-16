#!/bin/bash -l

#COBALT -A GRACE -n 2 -t 00:20:00 -q full-node --attrs="filesystems=home,theta-fs0" --mode script

date

module -t list

warmup=5
numiter=10

for library in 1
do
for direction in 1 2
do
for pattern in 1 2 3
do
for p in 16
do
for g in 8
do
for k in 1 8
do
#for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
for count in 10000000
do
  mpirun --display-map --display-allocation -hostfile ${COBALT_NODEFILE} -n $p -N 8 ./CommBench $library $pattern $direction $count $warmup $numiter $p $g $k
done
done
done
done
done
done
done

date
