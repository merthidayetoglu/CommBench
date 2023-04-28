# CommBench

CommBench takes in nine command line parameters. To run the executable:
```cpp
mpirun ./CommBench library pattern direction count warmup numiter p g k
```
where
1. ```library```: 0 for IPC, 1 for MPI, 2 for NCCL
2. ```pattern```: 1 for Rail, 2 for Dense, 3 for Fan
3. ```direction```: 1 for unidirectional, 2 for bidirectional, 3 for
omnidirectional
4. ```count```: number of 4-byte elements
5. ```warmup```: number of warmup rounds
6. ```numiter```: number of measurement rounds
7. ```p```: number of GPUs
8. ```g```: group size
9. ```k```: subgroup size

When ```pattern``` is set to 0, CommBench performs point-to-point (P2P) scan and ```g``` and ```k``` are insignificant.

The best practice for using CommBench is to prepare a run script that sweeps over the desired parameters and directs the output into a file. Then, the user can ```grep``` the desired output for further analysis and plotting. Each system requires a special care considering the modules, environment variables, and configuration parameters. To compile and run CommBench out-of-the-box on six systems, we include Make files and run scripts in our repository as displayed below.

| System        | Make File        | Run Script    |
| ------------- | -------------    | ------------- |
| Delta         | Makefile_delta   | run_delta.sh  |
| Summit        | Makefile_summit  | run_summit.sh |
