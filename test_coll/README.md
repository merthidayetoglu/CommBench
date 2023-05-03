### Standard Collective Tests

We made benchmarking standard collective communication function easy with an extension to CommBench. The main purpose of this test is providing a baseline for the group-to-group (*Rail*, *Dense*, and *Fan*) patterns that are defined by CommBench.

The collective tests is designed so that one can use the ```main.cpp``` herein with minimal changes to the CommBench's Make files and run scripts in the ```CommBench/scripts``` folder. This test requires a total of five command-line parameters as
```cpp
mpirun ./CommBench library pattern count warmup numiter
```
where
1. library: 1 for MPI, 2 for NCCL
2. pattern:
  - 1 for Gather, as in MPI_Gather
  - 2 for Scatter, as in MPI_Scatter
  - 3 for Reduce, as in MPI_Reduce and ncclReduce
  - 4 for Broadcast, as in MPI_Bcast and ncclBcast
  - 5 for All-to-all, as in MPI_Alltoall
  - 6 for All-reduce, as in MPI_Allreduce and ncclAllReduce
  - 7 for All-gather, as in MPI_Allgather and ncclAllGather
  - 8 for Reduce-scatter, as in MPI_Reduce_scattern and ncclReduceScatter
3. count: number of 4-byte elements
4. warmup: number of warmup rounds
5. numiter: number of measurement rounds
6. groupsize (optional): optional parameter to report B/W per group.

Notice that NCCL implements only five collective functions, whereas MPI does implement many more collective functions. Nevertheless, We only consider eight MPI and five NCCL functions to cover the most important ones.

#### Relation to Group-to-Group Patterns
