### Point-to-Point (P2P) Tests

This extension measures bandwidth and latency between two GPUs. For one-dimensional scan, the sender GPU is set constant and the receiver GPU is "scanned" across various levels in the hierarchy. For two-dimensional scan, both sender and receiver GPUs are scanned through a *for* loop in the run script.

The P2P tests is designed so that one can use the ```main.cpp``` herein with minimal changes to the CommBench's Make files and run scripts in the ```CommBench/scripts``` folder. This test requires a total of seven command-line parameters as
```cpp
mpirun ./CommBench library direction count warmup numiter sender recver
```
where
1. library: 1 for MPI, 2 for NCCL
2. direction: 1 for unidirectional, 2 for bidirectional
3. count: number of 4-byte elements
4. warmup: number of warmup rounds
5. numiter: number of measurement rounds
6. sender: sending processor rank
7. recver: receiving processor rank
