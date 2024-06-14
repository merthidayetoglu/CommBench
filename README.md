## CommBench

CommBench is a portable micro-benchmarking software for high-performance computing (HPC) networks. The tool integrates: MPI, NCCL, RCCL, and OneCCL libraries; CUDA, HIP, and Level Zero IPC capabilities; and recently GASNet-EX RMA functions. CommBench provides a library- and machine-agnostic API for users to compose desired communication patterns, takes measurements, and offers ports for benchmarking on Nvidia, AMD, and Intel GPUs.

Since this is an ongoing research project, the API specification evolves according to new needs.

For questions and support, please send an email to merth@stanford.edu

## API

CommBench has a higher-level interface for implementing custom micro-benchmarks in a system-agnostic way. C++ API is used to program a desired pattern by composition of point-to-point communication primitives. A more straightforward Python scripting interface is available in [pyComm](https://github.com/merthidayetoglu/CommBench/tree/master/pyComm). With either API, CommBench's API functions must be hit by all processes when programming a microbenchmark, where each process is bound to a single GPU. See [mapping](#rank-assignment) of parallel processes to the GPUs in the physical system.


#### Inclusion

CommBench is programmed into a single header file ``comm.h`` that is included in applications as the following example. The GPU port is specified by one of ``PORT_CUDA``, ``PORT_HIP``, or ``PORT_OneAPI`` for Nvidia, AMD, and Intel systems, respectively. When the GPU port is not specified, CommBench runs on CPUs.

```cpp
#define PORT_CUDA
#include "commbench.h"

using namespace CommBench;

int main() {

  char *sendbuf;
  char *recvbuf;
  size_t numbytes = 1e9;

  // initialize communicator with a chosen implementation library
  Comm test<char>(IPC);
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  // compose microbenchmarking pattern
  test.add(sendbuf, recvbuf, numbytes, 0, 1);

  // take measurement with 5 warmup and 10 measurement iterations
  test.measure(5, 10);

  free(sendbuf);
  free(recvbuf);

  return 0;
}
```

The above code allocates data on GPUs, and then measures IPC bandwidth across two GPUs within the same node with a message of 1 GB. The direction of data movement is from GPU 0 to GPU 1. An explanation of the CommBench functions is provided below.


#### Communicator

The benchmarking pattern is registered into a persistent communicator. The data type must be provided at compile time with the template parameter ``T``. Communication library for the implementation must be specified at this stage because the communicator builds specific data structures accordingly. Current options are ``MPI``, ``XCCL``, and ``IPC``. The choice ``XCCL`` is to enable vendor-provided collective communication library, such as NCCL for the CUDA port, RCCL for the HIP port, or OneCCL for the OneAPI port. The choice ``IPC`` is enables one-sided put protocol by default. For enabling get protocol, we included the ``IPC_get`` option.

```cpp
template <typename T>
CommBench::Comm<T> Comm(CommBench::Library);
```

#### Pattern Composition

CommBench relies on point-to-point communications as communication unit to build collective communication patterns. The API offers a single function ``add`` for registering point-to-point communications that can be used as the building block for the desired pattern.

The simple registration function is defined below, where ``count`` is the number of elements (of type ``T``) to be transfered from ``sendid`` to ``recvid``.
```cpp
void CommBench::Comm<T>::add(T *sendbuf, T *recvbuf, size_t count, int sendid, int recvid);
```

A more rigorous registration function requires the pointers to the send & receive buffers and the offset to the data. This interface is included for reliability in IPC communications, where the registered pointers must point to the head of the buffer as returned by virtual memory allocation. The rest of the arguments are the number of elements (their type is templatized) and the MPI ranks of the sender and receiver processes in the global communicator (i.e., ``MPI_COMM_WORLD``). See [rank assignment](#rank-assignment) for the affinity of the MPI processes in the physical system.

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

#### Measurement

The communication time can be measured with minimal overhead using the synchronization functions as below.

```cpp
MPI_Barrier(MPI_COMM_WORLD);
double time = MPI_Wtime();
comm.start();
comm.wait();
time = MPI_Wtime() - time;
MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
```

It is tedious to take accurate measurements, mainly because it has to be repeated several times to find the peak performance. We provide a measurement functions that executes the communications multiple times and reports the statistics.
```cpp
void CommBench::Comm<T>::measure(int warmup, int numiter);
```
For "warming up", communications are executed ``warmup`` times. Then the measurement is taken over ``numiter`` times, where the latency in each round is recorded for calculating the statistics.

## Rank Assignment
CommBench is implemented with a single-process-per-GPU paradigm. For example, on a partition with two-nodes with four GPUs per node, there are eight MPI processes assigned as:
| MPI Rank    | Node    | Device |
| ----------- | ------- | ----- |
| MPI rank 0  | node 0  | GPU 0 |
| MPI rank 1  | node 0  | GPU 1 |
| MPI rank 2  | node 0  | GPU 2 |
| MPI rank 3  | node 0  | GPU 3 |
| MPI rank 4  | node 1  | GPU 0 |
| MPI rank 5  | node 1  | GPU 1 |
| MPI rank 6  | node 1  | GPU 2 |
| MPI rank 7  | node 1  | GPU 3 |

Systems have different ways for scheduling the jobs for this assignment, which are described in their user guide, e.g. [Perlmutter](https://docs.nersc.gov/systems/perlmutter/running-jobs/#4-nodes-16-tasks-16-gpus-all-gpus-visible-to-all-tasks). In CommBench, the GPU selection is made in ``util.h`` for a specific port. For CUDA and HIP ports, all devices in the node must be seen by an MPI process. The device for each process is selected as ``myid % numdevice``, where ``myid`` is the MPI rank and ``numdevice`` is the number of GPUs seen by a process.

On the other hand, the SYCL port uses Level Zero backend and the device selection is made by setting ``ZE_SET_DEVICE`` to a single single device. This requires IPC implementation to make systems calls that work only on Aurora, which is an Intel system. Therefore IPC with the SYCL port does not work on Sunspot, which is a testbed of Aurora with an older OS. Run scripts for Aurora, and other systems are provided in the [/scripts](https://github.com/merthidayetoglu/CommBench/tree/master/scripts) folder.

MPI rank 0 and rank 4 can chosen for measuring bandwidth across nodes. When there are multiple NICs, measuring with one process per node results in utilization of a single NIC. The following micro-benchmark utilizes all NICs by striping point-to-point data across nodes.

## Multi-Step Patterns

For benchmarking multiple steps of communication patterns where each step depends on the previous, CommBench provides the following function:
```cpp
void CommBench::measure_async(std::vector<Comm<T>> sequence, int warmup, int numiter, size_t count);
```
In this case, the sequence of communications are given in a vector, e.g., ``sequence = {comm_1, comm_2, comm_3}``. CommBench internally figures out the data dependencies across steps and executes them asynchronously while preserving the dependencies across point-to-point functions.

![Striping](examples/striping/images/striping_figure.png)

As an example, the above shows striping of point-to-point communications across nodes. The asynchronous execution of this pattern finds opportunites to overlap communications within and across nodes using all GPUs, and utilizes the overall hierarchical network (intra-node, extra-node) efficiently towards measuring the peak bandwidth across nodes. See [examples/striping](https://github.com/merthidayetoglu/CommBench/tree/master/examples/striping) for an implementation with CommBench. The measurement will report the end-to-end latency ($t$) and throughput ($d/t$), where $d$ is the data movement across nodes and calculated based on ``count`` and the size of data type ``T``.

## Remarks

For questions and support, please send an email to merth@stanford.edu

Presentation: [CommBench: ICS 2024](https://merthidayetoglu.github.io/samples/CommBench_ICS24.pdf)

Paper: [CommBench: Micro-Benchmarking Hierarchical Networks with Multi-GPU, Multi-NIC Nodes](https://merthidayetoglu.github.io/samples/ics24-1.pdf)

Citation:
```
@inproceedings{hidayetoglu2024commbench,
  title={CommBench: Micro-Benchmarking Hierarchical Networks with Multi-GPU, Multi-NIC Nodes},
  author={Hidayetoglu, Mert and De Gonzalo, Simon Garcia and Slaughter, Elliott and Li, Yu and Zimmer, Christopher and Bicer, Tekin and Ren, Bin and Gropp, William and Hwu, Wen-Mei and Aiken, Alex},
  booktitle={Proceedings of the 38th ACM International Conference on Supercomputing},
  pages={426--436},
  year={2024}
}
```

Collaborators: Simon Garcia de Gonzalo (Sandia), Elliot Slaughter (SLAC), Chris Zimmer (ORNL), Bin Ren (W&M), Tekin Bicer (ANL), Wen-mei Hwu (Nvidia), Bill Gropp (Illinois), Alex Aiken (Stanford).
