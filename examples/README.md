## Striping Example

### High-level Goal

This example uses multi-rail striping algorithm to perform point-to-point GPU communication between 2 GPUs on different node. Each node consists of 4 GPUs, which is logically binding to 4 NICs respectively. A direct P2P GPU communication would only utilize 1 NIC's bandwidth while there are 4 NICs available. We first partition data into 4 chunks and transfer data chunks to GPUs at the transmitter node, then send data through each GPUs and their corresponding NICs, and finally assemble data at the receiver node.

### Code Explanation

We declare three communicator to achieve the goal described above:
```cpp
Comm<int> partition(IPC);
Comm<int> translate(NCCL);
Comm<int> assemble(IPC);
```
``IPC`` is the communication mechanism we used for intra-node communication. ``NCCL`` is used for inter-node communication. The choice of communication method is optimized for this example on Perlmutter. ``MPI`` is also available for both intra-node and inter-node communication. The optimization is shown in later section.

We use ``void CommBench::Comm::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid)`` to register data transfer events for each communicator. As we explained in High-level Goal, ``partition`` and ``assemble`` is responsible for intra-node communication at the transmitter and receiver node respectively, and ``translate`` is responsible for inter-node communication using 4 NICs' bandwidth.

After registering events, we use ``void CommBench::measure(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count)`` to run the events back-to-back and measuring the latency (performance) at the same time.

### Optimization 

![Latency](https://github.com/merthidayetoglu/CommBench/blob/master/examples/images/latency.png)

![Speedup](https://githhub.com/merthidayetoglu/CommBench/blob/master/examples/images/speedup.png)

The above pictures shows the optimization result. We have tried different combinations of communication libraries among three communicator (IPC only works for intra-node communication), and we found ``IPC`` + ``NCCL`` has the best performance when data size is large. When data size is small, ``direct MPI`` has the best performance.



