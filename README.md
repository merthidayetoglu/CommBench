## CommBench

CommBench is a portable benchmarking tool for HPC networks involving multi-GPU nodes. The tool integrates MPI, NCCL, and IPC capabilities, provides an API for users to compose desired communication patterns, takes measurements, and offers ports for benchmarking on Nvidia, AMD, and Intel GPUs.

For questions and support, please send an email to merth@stanford.edu

## API

CommBench is a runtime tool for implementing collective communications. It offers a C++ API for programming the desired communication patterns and run them asynchronously. CommBench is implemented with MPI and offers a global API, where functions must be hit by all processes in ``MPI_COMM_WORLD`` as if the program is sequential.

#### Communicator

The benchmarking pattern is registered into a persistent communicator. The data type must be provided at compile time with the template parameter ``T``. Communication library for the implementation must be specified at this stage because the communiator builds specific data structures accordingly. Current options are: ``CommBench::MPI``, ``CommBench::NCCL``, and ``CommBench::IPC``. 

```cpp
template <typename T>
CommBench::Comm<T> Comm(CommBench::Library);
```

#### Pattern Composition

CommBench relies on point-to-point communications. The API offers a single function ``add`` for registering point-to-point communications that can be used as the building block for the desired pattern. The function requires the pointers to the send and recieve buffers as well as the offset to the data. For IPC implementation, the pointers must point to the head of the buffer, as returned by virtual memory allocation. The rest of the arguments are the number of elements (their type is templatized) and the MPI ranks of the sender and reciever processes in the global communicator (i.e., ``MPI_COMM_WORLD``). For GPU-aware MPI and NCCL, we assume each process runs a GPU.

```cpp
template <typename T>
void CommBench::Comm<T>::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
```

For seeing the benchmarking pattern as a sparse communication matrix, one can call the ``report()`` function.
```cpp
template <typename T>
void CommBench::Comm<T>::report();
```

#### Synchronization

Synchronization across the GPUs is made by ``start()`` and ``wait()`` functions. The former launches the registered communications all at once using nonblocking API of the chosen library. The latter blocks the program until the communication buffers are safe to be reused. Among all GPUs, only those who are involved in the communications are effected. Others move on executing the program. For example, in a point-to-point communication, the sending process returns from the ``wait()`` function when the send buffer is safe to be reused. Likewise, the recieving process returns from the ``wait()`` function when the recieve buffer is safe to be reused. The GPUs that are not involved return from both ``start()`` and ``wait()`` functions immediately.

```cpp
template <typename T>
void CommBench::Comm<T>::start();

template <typename T>
void CommBench::Comm<T>::wait();
```

The communication time can be measured with minimal overhead using the synchronization functions as below.

```cpp
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
comm.start();
comm.wait();
time = MPI_Wtime() - time;
MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
```

#### Measurement

It is tedious to take accurate measurements, mainly because it has to be repeated several times to find the peak performance. We provide a measurement functions that executes the communications multiple times and reports the statistics.

```cpp
template <typename T>
void Comm<T>::measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime)
```
For "warming up", communications are executed ``warmup`` times. Then the measurement is taken over ``numiter`` times, where the latency in each round is recorded for calculating the statistics.

#### Example

There are multiple examples in the ``main.cpp`` that implement the proposed group-to-group patterns (see below). Each benchmarking pattern is parameterized and expressed in a few lines of code. Custom (e.g., application dependent) benchmarks can be configured similarly.

## Group-to-Group Benchmarking

To describe the performance behavior of communications across groups of processors, we define group-to-group patterns (i.e., Rail, Fan, and Dense). For gradually varying the number of devices involved in communication with various ways, we parameterize these patterns with configuration control variables $(p, g, k)$. To run CommBench, one needs to set a total of ten command line parameters as
```cpp
mpirun ./CommBench library pattern direction count window warmup numiter p g k
```
where
1. library: 0 for IPC, 1 for MPI, and 2 for NCCL
2. pattern: 1 for Rail, 2 for Fan, and 3 for Dense
3. direction: 1 for outbound, 2 for inbound, 3 for bidirectional and 4 for omnidirectional
4. count: number of 4-byte elements per message
5. window: number of messages per round
6. warmup: number of warmup rounds
7. numiter: number of measurement rounds
8. $p$: number of GPUs
9. $g$: group size
10. $k$: subgroup size

The best practice for using CommBench is to prepare a run script that sweeps over the desired parameters and directs the output into a file. Then, the user can "grep" the desired output for further analysis and plotting. Each system requires a special care considering the modules, environment variables, and configuration parameters. To compile and run CommBench out-of-the-box on six systems, we include Make files and run scripts in the `/script` as displayed below.

|Facility | System | Make File | Run Script |
| :--- | :---| :--- | :--- |
| NCSA | Delta | `Makefile_delta` | `run_delta.sh`  |
| OLCF | Summit | `Makefile_summit` | `run_summit.sh` |
| NERSC | Perlmutter | `Makefile_perlmutter` | `run_perlmutter.sh` |
| ALCF | ThetaGPU | `Makefile_thetagpu` | `run_thetagpu.sh` |
| OLCF | Frontier | `Makefile_frontier` | `run_frontier.sh` |
| ALCF | Sunspot | `Makefile_sunspot` | `run_sunspot.sh` |

There is an extension of CommBench for standard collectives in the ```CommBench/test_coll``` folder.

CommBench is the second (and final) iteration of the communication benchmarking tool. See https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester for the previous version.

#### Configuration Parameters

We propose a few pre-implemented benchmarking patterns that isolates data movement acoss groups. There are three parameters, $p$, $g$, and $k$ to configure the benchmark, where $p$ is the total number of processors, $g$ is the group size, and $k$ is the subgroup size where the communication is initiated in the originating group.

![Comm Patterns](https://github.com/merthidayetoglu/CommBench/blob/master/figures/comm_patterns.png)

To define benchmark configuration without ambiguity, one needs to specify $(p, g, k)$, pattern, and the direction of data movement.

#### Direction of Data Movement

For convenience, we define unidirectional, bidirectional, and omnidirectional data movement for the group communication patterns. The terminals of unidirectional and bidirectional data movement are located in the originating group. Omnidirectional data movement happens across all groups.

![Scaling Patterns](https://github.com/merthidayetoglu/CommBench/blob/master/figures/scaling_patterns.png)


For questions and support, please send an email to merth@stanford.edu
