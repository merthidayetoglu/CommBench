### CollBench

This folder includes an extension for benchmarking standard collective functions with CPU-only MPI, GPU-aware MPI, and NCCL. To compile and run, one can use the Make files and run scripts in the ```CommBench/scripts``` folder. To run CommBench, one needs to set a total of five command line parameters as
```cpp
mpirun ./CommBench library pattern count warmup numiter
```
where
1. library: 0 for IPC, 1 for MPI, 2 for NCCL
2. pattern:
  - 1 for Gather
  - 2 for Scatter
  - 3 for Reduce
  - 4 for Broadcast
  - 5 for All-to-all
  - 6 for All-reduce
  - 7 for All-gather
  - 8 for Reduce-scatter
3. count: number of 4-byte elements
4. warmup: number of warmup rounds
5. numiter: number of measurement rounds
6. groupsize: optional parameter to report B/W per group.
