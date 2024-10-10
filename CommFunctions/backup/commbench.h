/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

// Header guards
#ifndef COMMBENCH_H
#define COMMBENCH_H

//MPI OR GASNET
#ifdef USE_GASNET
#define CAP_GASNET
#else
#define USE_MPI
#endif

// Includes
#include <stdio.h>
#include <string.h>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include <sys/syscall.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef PORT_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PORT_HIP
#include <hip_runtime.h>
#endif
#ifdef PORT_ONEAPI
#include <sycl.hpp>
#endif
#ifdef CAP_GASNET
#include <gasnetex.h>
#include <gasnet_mk.h>
#endif

namespace CommBench {

// Global variables and types
enum library {dummy, MPI, NCCL, IPC, IPC_get, GEX, GEX_get, numlib};

extern int printid;
extern int numbench;
extern std::vector<void*> benchlist;
extern int mydevice;
extern int myid;
extern int numproc;

#ifdef USE_MPI
extern MPI_Comm comm_mpi;
#endif

#ifdef PORT_ONEAPI
extern sycl::queue q;
#endif

#ifdef CAP_NCCL
extern ncclComm_t comm_nccl;
#endif

#ifdef CAP_ONECCL
extern ccl::communicator* comm_ccl;
#endif

#ifdef CAP_GASNET
extern gex_Client_t myclient;
extern gex_EP_t ep_primordial;
extern gex_TM_t myteam;
extern gex_MK_t memkind;
extern std::vector<void*> myep_ptr;
extern std::vector<gex_EP_t> myep;
#endif

// Function declarations
void print_data(size_t data);
void print_lib(library lib);
void barrier();

template <typename T>
void allocate(T *&buffer, size_t n);

template <typename T>
void allocateHost(T *&buffer, size_t n);

template <typename T>
void memcpyD2H(T *host, T *device, size_t n);

template <typename T>
void memcpyH2D(T *device, T *host, size_t n);

template <typename T>
void free(T *buffer);

template <typename T>
void freeHost(T *buffer);

template <typename T>
void send(T *sendbuf, int recvid);

template <typename T>
void recv(T *recvbuf, int sendid);

template <typename T>
void pair(T *sendbuf, T *recvbuf, int sendid, int recvid);

template <typename T>
void broadcast(T *sendbuf, T *recvbuf, int root);

template <typename T>
void broadcast(T *sendbuf);

template <typename T>
void allgather(T *sendval, T *recvbuf);

template <typename T>
void allreduce_sum(T *sendbuf, T *recvbuf);

template <typename T>
void allreduce_sum(T *sendbuf);

template <typename T>
void allreduce_max(T *sendbuf, T *recvbuf);

template <typename T>
void allreduce_max(T *sendbuf);

char allreduce_land(char logic);

template <typename T>
void measure_async(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count);

template <typename T>
void measure_concur(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count);

template <typename T>
void measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime, Comm<T> &comm);

void set_device(int device);
void setup_gpu();

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

void init();
void finalize();
void print_stats(std::vector<double> times, size_t data);

} // namespace CommBench

#endif // COMMBENCH_H

