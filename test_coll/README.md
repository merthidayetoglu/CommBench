### Standard Collective Tests

We made benchmarking standard collective communication function easy with an extension to CommBench. The main purpose of this test is providing a baseline for the group-to-group (*Rail*, *Dense*, and *Fan*) patterns that are defined by CommBench.

The collective tests is designed so that one can use the ```main.cpp``` herein with minimal changes to the CommBench's Make files and run scripts in the ```CommBench/scripts``` folder. This test requires a total of five command-line parameters as
```cpp
mpirun ./CommBench library pattern count warmup numiter
```
where
1. library: 1 for MPI, 2 for NCCL
2. pattern:
  - 1 for *Gather*, as in `MPI_Gather`
  - 2 for *Scatter*, as in `MPI_Scatter`
  - 3 for *Reduce*, as in `MPI_Reduce` and `ncclReduce`
  - 4 for *Broadcast*, as in `MPI_Bcast` and `ncclBcast`
  - 5 for *All-to-all*, as in `MPI_Alltoall`
  - 6 for *All-reduce*, as in `MPI_Allreduce` and `ncclAllReduce`
  - 7 for *All-gather*, as in `MPI_Allgather` and `ncclAllGather`
  - 8 for *Reduce-scatter*, as in `MPI_Reduce_scatter` and `ncclReduceScatter`
3. count: number of 4-byte elements
4. warmup: number of warmup rounds
5. numiter: number of measurement rounds

Notice that NCCL implements only five collective functions, whereas MPI does implement many more collective functions. Nevertheless, We only consider eight MPI and five NCCL functions to cover the most important ones.

### Relation to Group-to-Group Patterns

As an example we consider *Scatter* collective as in `MPI_Scatter` and `ncclScatter`.

Example unidirectional Fan (16, 8, 1) pattern implemented with MPI using CommBench.
```cpp
CommBench::Bench<double> bench(MPI_COMM_WORLD, CommBench::Library::MPI);

for(int j = 0; j < 8; j++)
  bench.add(sendbuf, j * count, recvbuf, 0, count, 0, 8 + j); // 0 -> 8 + j
  
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
bench.start();
bench.wait();
MPI_Barrier(MPI_COMM_WORLD);
time = MPI_Wtime() - time;
```

Equivalent MPI_Scatter collective function.
```cpp
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
MPI_Scatter(sendbuf, count, MPI_DOUBLE, recvbuf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
time = MPI_Wtime() - time;
```
