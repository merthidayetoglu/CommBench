## CommBench

CommBench is a portable benchmarking tool for HPC networks involving multi-GPU nodes. The tool integrates MPI, NCCL, and IPC capabilities, provides an API for users to compose desired communication patterns, takes measurements, and offers ports for benchmarking on Nvidia, AMD, and Intel GPUs.

For questions and support, please send an email to merth@stanford.edu

## API

CommBench is a runtime tool for implementing custom microbenchmarks. It offers a C++ API for programming a desired pattern using a composition of point-to-point communications. When programming a microbenchmark, CommBench's API functions must be hit by all processes as if the program is sequential.

#### Communicator

The benchmarking pattern is registered into a persistent communicator. The data type must be provided at compile time with the template parameter ``T``. Communication library for the implementation must be specified at this stage because the communiator builds specific data structures accordingly. Current options are: ``MPI``, ``NCCL``, and ``IPC``. 

```cpp
template <typename T>
CommBench::Comm<T> Comm(CommBench::Library);
```

#### Pattern Composition

CommBench relies on point-to-point communications. The API offers a single function ``add`` for registering point-to-point communications that can be used as the building block for the desired pattern.

For quick tests, ``add_lazy`` allocates communication buffers buffers internally.

```cpp
void CommBench::Comm<T>::add_lazy(size_t count, int sendid, int recvid);
```

The rigorous function requires the pointers to the send and recieve buffers as well as the offset to the data. For reliability, the pointers must point to the head of the buffer, as returned by virtual memory allocation. The rest of the arguments are the number of elements (their type is templatized) and the MPI ranks of the sender and reciever processes in the global communicator (i.e., ``MPI_COMM_WORLD``).

```cpp
void CommBench::Comm<T>::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
```

For seeing the benchmarking pattern as a sparse communication matrix, one can call the ``report()`` function.
```cpp
void CommBench::Comm<T>::report();
```

#### Synchronization

Synchronization across the GPUs is made by ``start()`` and ``wait()`` functions. The former launches the registered communications all at once using nonblocking API of the chosen library. The latter blocks the program until the communication buffers are safe to be reused.

Among all GPUs, only those who are involved in the communications are effected by the ``wait()`` call. Others move on executing the program. For example, in a point-to-point communication, the sending process returns from ``wait()`` when the send buffer is safe to be reused. Likewise, the recieving process returns from ``wait()`` when the recieve buffer is safe to be reused. The GPUs that are not involved return from both ``start()`` and ``wait()`` functions immediately. In sparse communications, each GPU can be sender and receiver, and the ``wait()`` function blocks a process until all associated communications are completed.

```cpp
void CommBench::Comm<T>::start();
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
void CommBench::Comm<T>::measure(int warmup, int numiter);
```
For "warming up", communications are executed ``warmup`` times. Then the measurement is taken over ``numiter`` times, where the latency in each round is recorded for calculating the statistics.

## Multi-Step Benchmarks

For benchmarking multiple steps of communication patterns where each step depends on the previous, CommBench provides the following function:
```cpp
void CommBench::measure(std::vector<Comm<T>> sequence, int warmup, int numiter, size_t count);
```
In this case, the sequence of communications are given in a vector, e.g., ``sequence = {comm_1, comm_2, comm_3}``. CommBench internally figures out the data dependencies across steps and executes them asynchronously while preserving the dependencies across point-to-point functions.

![Striping](examples/striping/images/striping_figure.pdf)

As an example, the below shows striping of point-to-point communications across nodes. The asynchronous execution of this pattern finds opportunites to overlap communications within and across nodes using all GPUs, and utilizes the overall hierarchical network (intra-node, extra-node) efficiently towards measuring the peak bandwidth across nodes. See [examples/striping](https://github.com/merthidayetoglu/CommBench/tree/master/examples/striping) for an implementation with CommBench. The measurement will report the end-to-end latency ($t$) and throughput ($d/t$), where $d$ is the data movement across nodes and calculated based on ``count`` and the size of data type ``T``.

For questions and support, please send an email to merth@stanford.edu
