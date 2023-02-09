#!/bin/bash -l

#COBALT -A GRACE -n 2 -t 00:20:00 -q full-node --attrs="filesystems=home,theta-fs0" --mode script

date

module -t list

for count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456
do
  mpirun --display-map --display-allocation -hostfile ${COBALT_NODEFILE} -n 16 -N 8 Alltoall $count 30 8
done

date
