#ifndef COMM_H
#define COMM_H

#include <vector>
#include <cstdio>
#include <cstdlib>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef PORT_CUDA
#include <cuda_runtime.h>
#elif defined PORT_HIP
#include <hip/hip_runtime.h>
#elif defined PORT_SYCL
#include <level_zero/ze_api.h>
#endif

#ifdef CAP_ONECCL
#include <ccl.hpp>
#endif

#ifdef CAP_GASNET
#include <gasnet.h>
#endif

enum library {dummy, MPI, NCCL, IPC, IPC_get, GEX, GEX_get, numlib};


template <typename T>
class Comm {
public:
    // IMPLEMENTATION LIBRARY
    const library lib;

    // STATS
    int benchid;
    int numcomm = 0;

    // REGISTRY
    int numsend;
    int numrecv;
    std::vector<T*> sendbuf;
    std::vector<T*> recvbuf;
    std::vector<int> sendproc;
    std::vector<int> recvproc;
    std::vector<size_t> sendcount;
    std::vector<size_t> recvcount;
    std::vector<size_t> sendoffset;
    std::vector<size_t> recvoffset;
    // REMOTE BUFFER
    std::vector<T*> remotebuf;
    std::vector<size_t> remoteoffset;

    // SYNCHRONIZATION
    std::vector<int> ack_sender;
    std::vector<int> ack_recver;
    void block_sender();
    void block_recver();

    // MPI
#ifdef USE_MPI
    std::vector<MPI_Request> sendrequest;
    std::vector<MPI_Request> recvrequest;
#endif

    // NCCL
#if defined CAP_NCCL && defined PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined CAP_NCCL && defined PORT_HIP
    hipStream_t stream_nccl;
#elif defined CAP_ONECCL
    ccl::stream *stream_ccl;
#endif

    // IPC
#ifdef PORT_CUDA
    std::vector<cudaStream_t> stream_ipc;
#elif defined PORT_HIP
    std::vector<hipStream_t> stream_ipc;
#elif defined PORT_ONEAPI
    std::vector<sycl::queue> q_ipc;
#endif

    // IPC ZE
#ifdef IPC_ze
    std::vector<ze_command_list_handle_t> command_list;
    std::vector<ze_command_queue_handle_t> command_queue;
    bool command_list_closed = false;
    int ordinal2index = 0;
#endif

    // GASNET
#ifdef CAP_GASNET
    std::vector<int> my_ep;
    std::vector<int> remote_ep;
    int find_ep(void *buffer);
    std::vector<gex_Event_t> gex_event;
    std::vector<int> remote_sendind;
    std::vector<int> remote_recvind;
    bool send_ready();
    bool recv_ready();
    gex_AM_Index_t am_notify_sender_index;
    gex_AM_Index_t am_notify_recver_index;
    static void am_notify_sender(gex_Token_t token, gex_AM_Arg_t send, gex_AM_Arg_t bench);
    static void am_notify_recver(gex_Token_t token, gex_AM_Arg_t recv, gex_AM_Arg_t bench);
#endif

    Comm(library lib);
    void free();
    void init();

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void add(T *sendbuf, T *recvbuf, size_t count, int sendid, int recvid);
    void add_lazy(size_t count, int sendid, int recvid);
    void pyadd(pyalloc<T> sendbuf, size_t sendoffset, pyalloc<T> recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void start();
    void wait();

    void measure(int warmup, int numiter);
    void measure(int warmup, int numiter, size_t count);
    std::vector<size_t> getMatrix();
    void report();

    void allocate(T *&buffer, size_t n);
    void allocate(T *&buffer, size_t n, int i);
};

#include "comm_impl.h"

#endif // COMM_H

