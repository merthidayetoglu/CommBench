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

Notice that NCCL implements only five collective functions. Moreover, MPI has many more collective functions. We only consider eight MPI and five NCCL functions to cover the most important ones.

### Relation to Group-to-Group Patterns

#### Example 1

As an example, we consider *All-to-all* collective as in `MPI_Alltoall`:
```cpp
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
MPI_Alltoall(sendbuf, count, MPI_DOUBLE, recvbuf, count, MPI_DOUBLE, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
time = MPI_Wtime() - time;
```
On two nodes of frontier, where there are eight GPUs per node, we can isolate the communication across nodes with bidirectional Dense (16, 8, 8) pattern which discards the intra-node communications. We can implemented this pattern easily with MPI using CommBench.
```cpp
CommBench::Bench<double> bench(MPI_COMM_WORLD, CommBench::Library::MPI);

// Bidirectional Dense (16, 8, 8) pattern
for(int i = 0; i < 8; i++)
  for(int j = 0; j < 8; j++) {
    bench.add(sendbuf, i * count, recvbuf, j * count, count, i, 8 + j); // i -> 8 + j
    bench.add(sendbuf, j * count, recvbuf, i * count, count, 8 + j, i); // 8 + j -> i
  }
  
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
bench.start();
bench.wait();
MPI_Barrier(MPI_COMM_WORLD);
time = MPI_Wtime() - time;
```


#### Example 2

As a second example, we consider *Scatter* collective as in `ncclScatter`. In Frontier, NCCL will run RCCL.
```cpp
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
ncclScatter(sendbuf, recvbuf, count, ncclFloat64, 0, comm_nccl, 0);
cudaStreamSynchronize(0);
MPI_Barrier(MPI_COMM_WORLD);
time = MPI_Wtime() - time;
```

We can isolate the communication across nodes with unidirectional Fan (16, 8, 1) pattern. We can implemented this pattern easily with NCCL using CommBench.

```cpp
CommBench::Bench<double> bench(MPI_COMM_WORLD, CommBench::Library::NCCL);

// Unidirectional Fan (16, 8, 1) pattern
for(int j = 0; j < 8; j++)
  bench.add(sendbuf, j * count, recvbuf, 0, count, 0, 8 + j); // 0 -> 8 + j
  
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
bench.start();
bench.wait();
MPI_Barrier(MPI_COMM_WORLD);
time = MPI_Wtime() - time;
```
