#!/bin/bash -l

#COBALT -A GRACE -n 2 -t 00:20:00 -q full-node --attrs="filesystems=home,theta-fs0" --mode script

date

module -t list

for count in 10000000
do
  mpirun -hostfile ${COBALT_NODEFILE} -N 8 ./CommBench $count
done

date
