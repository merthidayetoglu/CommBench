# CommBench

Preconfigured Make files and run scripts are located in the ```/scripts``` folder. The systems we test include Delta, Summit, Perlmutter, ThetaGPU, Frontier, and Sunspot.

CommBench takes in nine command line parameters. This tool runs like
```cpp
mpirun ./CommBench library pattern direction count warmup numiter p g k
```
1. ```library```: 0 for IPC, 1 for MPI, 2 for NCCL
2. ```pattern```: 1 for Rail, 2 for Dense, 3 for Fan
3. ```direction```: 1 for unidirectional, 2 for bidirectional, 3 for
omnidirectional (see Figure 5)
4. ```count```: number of 4-byte elements
5. ```warmup```: number of warmup rounds
6. ```numiter```: number of measurement rounds
7. ```p```: number of GPUs
8. ```g```: group size
9. ```k```: subgroup size
