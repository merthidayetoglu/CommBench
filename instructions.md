# Overview of Usage

This branch provides the same `CommBench` executable with the standard usage flags, but also adds the `--source` and `--dest` flags that can be used for P2P, gather, and broadcast patterns. Gather only supports `--dest` and broadcast only supports `--source`.

In addition, we provide four new targets. `bench_p2p`, `bench_gather`, `bench_broadcast` and `bench_alltoall` for these respective patterns. These provide support for "mass benchmarking tests", testing all combinations of the available processors. The usage for these is `bench_<pattern> <output_folder> <libs>` where you can provide a space separated list of libraries (out of `ipc_get`, `ipc_put`, `mpi`, and `xccl`) to test.

This will then dump the output for each of these benchmarks into a lot of files in the directories `output_folder/<lib>` for each specified library. These can be parsed and visualized later. These DO NOT create the directories themselves, so make sure that the directory `output_folder/<lib>` exists for every library you want to test. Otherwise the benchmarks will silently go on 

# Building CommBench for El Dorado Benchmarking

In this branch, most of the "debug" output is disabled by default and when enabled is directed to `stderr`. To enable this output, configure CMake with the `-DPRINT_DEBUG=ON` flag, but by default it is off so that only the necessary output is shown.

To build, run the following in the CommBench directory:
```sh
cmake -S . -B build -DUSE_HIP=ON -DUSE_XCCL=ON -DXCCL_PATH=/opt/rocm-6.3.1 -DUSE_GTL=ON -DGTL_PATH=/opt/cray/pe/mpich/8.1.32/ofi/amd/6.0 -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE -DMPI_C_COMPILER=/opt/cray/pe/mpich/8.1.32/ofi/amd/6.0/bin/mpicc -DMPI_CXX_COMPILER=/opt/cray/pe/mpich/8.1.32/ofi/amd/6.0/bin/mpicxx
```
and if you want to use CommBench's `IPC_kernel` for the `ipc_put` and `ipc_get` communications, add the flag `-DUSE_BLIT_KERNEL=ON`. It may be helpful to have two separate build folders, one with and one without this flag. Adjust `-B <build_folder>` accordingly.

Make sure to have the `PrgEnv-amd/8.6.0`, `craype-accel-amd-gfx942`, `cmake/3.24.2` and `rocm/6.3.1` modules loaded. To build for `rocm/6.4.1`, load `amd/6.4.1` and `rocm/6.4.1`.

# Running Benchmarks

Make sure to have to have `MPICH_GPU_SUPPORT_ENABLED=1` set. Experiment with the `HSA_ENABLE_SDMA=0` environment variable as well.

Set up the output folders before running the benchmark. The scripts expect this file structure:
```
pattern
├── CPX
│   ├── lib1
│   ├── lib2
│   └── lib3
├── SPX
│   ├── lib1
│   ├── lib2
│   └── lib3
└── TPX
    ├── lib1
    ├── lib2
    └── lib3
```
You can technically have different libraries specified for the partitions, but as mentioned before, make sure the folders are created inside the partition.

Allocate an instance with
```sh
flux alloc -N1 --setattr=gpumode=<modality> --conf=resource.rediscover=true --time-limit=<time>
```

To run the benchmark, use the command with 4 for SPX, 12 for TPX, and 24 for CPX.
```sh
flux run -N1 -n<proc> -g1 -x -o mpibind=off ./build/bench_<pattern> <output_folder> <libs>
```

For example, you might specify the output folder to be `pattern/SPX` and you want to test the `ipc_get` pattern, then you would need to make sure the directory `pattern/SPX/ipc_get` exists. Then the command is 

```sh
flux run -N1 -n4 -g1 -x -o mpibind=off ./build/bench_<pattern> pattern/SPX ipc_get
```

Once this is done, you run the script `parse.py`. At the bottom of the script file, you can adjust the folder to `pattern` and then change the list to the modalities that you have. This script takes the text files and stores them in a pandas dataframe for visualization later.

Then the `visualize_df.py` script generates the graphs from this dataframe which would be stored in `pattern/SPX_data.pkl`. Adjust the function calls at the bottom for plotting different patterns with input folder corresponding to `pattern` where it can find `pattern/<modality>_data.pkl` and it will output to your desired folder. These scripts will create the folders as needed.