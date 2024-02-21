/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMBENCH_H
#define COMMBENCH_H

#include <mpi.h>

// GPU PORTS
// For NVIDIA: #define PORT_CUDA
// For AMD: #define PORT_HIP
// For SYCL: #define PORT_SYCL

#if defined PORT_CUDA || defined PORT_HIP
#define CAP_NCCL
#endif
#ifdef PORT_SYCL
// #define CAP_ONECCL
#define CAP_ZE
#endif

// DEPENDENCIES
#ifdef PORT_CUDA
#ifdef CAP_NCCL
#include <nccl.h>
#else
#include <cuda.h>
#endif
#elif defined PORT_HIP
#ifdef CAP_NCCL
#include <rccl.h>
#else
#include <hip_runtime.h>
#endif
#elif defined PORT_SYCL
#ifdef CAP_ONECCL
#include <oneapi/ccl.hpp>
#else
#include <sycl.hpp>
#endif
#ifdef CAP_ZE
#include <ze_api.h>
#endif
#endif

// CPP LIBRARIES
#include <stdio.h> // for printf
#include <string.h> // for memcpy
#include <algorithm> // for std::sort
#include <vector> // for std::vector
#include <unistd.h> // for fd
#include <sys/syscall.h> // for syscall

namespace CommBench
{
  static int printid = 0;

  enum library {dummy, MPI, XCCL, IPC, IPC_get, numlib};

  static MPI_Comm comm_mpi;
  static int myid;
  static int numproc;
#ifdef CAP_NCCL
  static ncclComm_t comm_nccl;
#endif
#ifdef CAP_ONECCL
  static ccl::communicator *comm_ccl;
#endif
#ifdef PORT_SYCL
  static sycl::queue q(sycl::gpu_selector_v);
#endif

  static int numbench = 0;
  static bool init_mpi_comm = false;
  static bool init_nccl_comm = false;
  static bool init_ccl_comm = false;

  static void print_data(size_t data) {
    if (data < 1e3)
      printf("%d bytes", (int)data);
    else if (data < 1e6)
      printf("%.4f KB", data / 1e3);
    else if (data < 1e9)
      printf("%.4f MB", data / 1e6);
    else if (data < 1e12)
      printf("%.4f GB", data / 1e9);
    else
      printf("%.4f TB", data / 1e12);
  }
  static void print_lib(library lib) {
    switch(lib) {
      case dummy : printf("dummy"); break;
      case IPC     : printf("PUT"); break;
      case IPC_get : printf("GET"); break;
      case MPI     : printf("MPI"); break;
      case XCCL    : printf("XCCL"); break;
      case numlib  : printf("numlib"); break;
    }
  }

  // MEMORY MANAGEMENT
  template <typename T>
  void allocate(T *&buffer,size_t n);
  template <typename T>
  void allocateHost(T *&buffer, size_t n);
  template <typename T>
  void memcpyD2H(T *device, T *host, size_t n);
  template <typename T>
  void memcpyH2D(T *host, T *device, size_t n);
  template <typename T>
  void free(T *buffer);
  template <typename T>
  void freeHost(T *buffer);

  // MEASUREMENT
  template <typename C>
  static void measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime, C comm);

  template <typename T>
  struct pyalloc {
    T* ptr;
    pyalloc(size_t n) {
      allocate(ptr, n);
    }
    void pyfree() {
      free(ptr);
    }
  };

  template <typename T>
  class Comm {

    public :

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

    // MEMORY
    std::vector<T*> buffer_list;
    std::vector<size_t> buffer_count;

    // MPI
    std::vector<MPI_Request> sendrequest;
    std::vector<MPI_Request> recvrequest;

    // XCCL
#if defined CAP_NCCL && defined PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined CAP_NCCL && defined PORT_HIP
    hipStream_t stream_nccl;
#elif defined CAP_ONECCL
    ccl::stream *stream_ccl;
#endif

    // IPC
    std::vector<T*> remotebuf;
    std::vector<size_t> remoteoffset;
    std::vector<int> ack_sender;
    std::vector<int> ack_recver;
#ifdef PORT_CUDA
    std::vector<cudaStream_t> stream_ipc;
#elif defined PORT_HIP
    std::vector<hipStream_t> stream_ipc;
#elif defined PORT_SYCL
    std::vector<sycl::queue> q_ipc;
#endif

    Comm(library lib);
    void free();

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void add(T *sendbuf, T *recvbuf, size_t count, int sendid, int recvid);
    void add_lazy(size_t count, int sendid, int recvid);
    void pyadd(pyalloc<T> sendbuf, size_t sendoffset, pyalloc<T> recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void start();
    void wait();

    void measure(int warmup, int numiter);
    void measure(int warmup, int numiter, size_t data);
    void getMatrix(std::vector<size_t> &matrix);
    void report();

    void allocate(T *&buffer, size_t n);
    void allocate(T *&buffer, size_t n, int i);
#include "util.h"
  };

  // THIS IS TO INITIALIZE COMMBENCH
  static Comm<char> init(dummy);

  template <typename T>
  Comm<T>::Comm(library lib) : lib(lib) {

    benchid = numbench;
    numbench++;

    int init_mpi;
    MPI_Initialized(&init_mpi);

    if(!init_mpi_comm) {
      if(!init_mpi) {
        MPI_Init(NULL, NULL);
      }
      MPI_Comm_dup(MPI_COMM_WORLD, &comm_mpi); // CREATE SEPARATE COMMUNICATOR EXPLICITLY
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);
      setup_gpu();
      init_mpi_comm = true;
      if(myid == printid) {
        if(!init_mpi)
          printf("#################### MPI IS INITIALIZED, it is user's responsibility to finalize.\n");
        printf("******************** MPI COMMUNICATOR IS CREATED\n");
      }
    }

    numsend = 0;
    numrecv = 0;

    if(myid == printid) {
      printf("printid: %d Create Bench %d with %d processors\n", printid, benchid, numproc);
      printf("  Port: ");
#ifdef PORT_CUDA
      printf("CUDA ");
#elif defined PORT_HIP
      printf("HIP, ");
#elif defined PORT_SYCL
      printf("SYCL, ");
#else
      printf("CPU, ");
#endif
      printf("Library: ");
      print_lib(lib);
      printf("\n");
    }
    if(lib == XCCL) {
#ifdef CAP_NCCL
      if(!init_nccl_comm) {
        ncclUniqueId id;
        if(myid == 0)
          ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm_mpi);
        ncclCommInitRank(&comm_nccl, numproc, id, myid);
        if(myid == printid)
          printf("******************** NCCL COMMUNICATOR IS CREATED\n");
        init_nccl_comm = true;
      }
#ifdef PORT_CUDA
      cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
      hipStreamCreate(&stream_nccl);
#endif
#elif defined CAP_ONECCL
      if(!init_ccl_comm) {
        /* initialize ccl */
        ccl::init();
        /* create kvs */
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type main_addr;
        if (myid == 0) {
          kvs = ccl::create_main_kvs();
          main_addr = kvs->get_address();
          MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, comm_mpi);
        }
        else {
          MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, comm_mpi);
          kvs = ccl::create_kvs(main_addr);
        }
        /* create communicator */
        auto dev = ccl::create_device(CommBench::q.get_device());
        auto ctx = ccl::create_context(CommBench::q.get_context());
        comm_ccl = new ccl::communicator(ccl::create_communicator(numproc, myid, dev, ctx, kvs));
        init_ccl_comm = true;
        if(myid == printid)
          printf("******************** ONECCL COMMUNICATOR IS CREATED\n");
        stream_ccl = new ccl::stream(ccl::create_stream(CommBench::q));
      }
#endif
    }
  }

  template <typename T>
  void Comm<T>::free() {
    for(T *ptr : buffer_list)
      CommBench::free(ptr);
    buffer_list.clear();
    buffer_count.clear();
    if(myid == printid)
      printf("memory freed.\n");
  }

  template <typename T>
  void Comm<T>::allocate(T *&buffer, size_t count) {
    for (int i = 0; i < numproc; i++)
      allocate(buffer, count, i);
  }
  template <typename T>
  void Comm<T>::allocate(T *&buffer, size_t count, int i) {
    if(myid == i) {
      MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      if(count) {
        CommBench::allocate(buffer, count);
        buffer_list.push_back(buffer);
        buffer_count.push_back(count);
        MPI_Send(&buffer, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
      }
    }
    if(myid == printid) {
      MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
      if(count) {
        T *ptr = nullptr;
        MPI_Recv(&ptr, sizeof(T*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        printf("Bench %d proc %d allocate %p count %ld (", benchid, i, ptr, count);
        print_data(count * sizeof(T));
        printf(")\n");
      }
    }
  }

  template <typename T>
  void Comm<T>::add_lazy(size_t count, int sendid, int recvid) {
    T *sendbuf;
    T *recvbuf;
    allocate(sendbuf, count, sendid);
    allocate(recvbuf, count, recvid);
    add(sendbuf, 0, recvbuf, 0, count, sendid, recvid);
  }
  template <typename T>
  void Comm<T>::add(T *sendbuf, T *recvbuf, size_t count, int sendid, int recvid) {
    add(sendbuf, 0, recvbuf, 0, count, sendid, recvid);
  }
  template <typename T>
  void Comm<T>::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
    if(count == 0) {
      if(myid == printid)
        printf("Bench %d communication (%d->%d) count = 0 (skipped)\n", benchid, sendid, recvid);
      return;
    }
    MPI_Barrier(comm_mpi); // THIS IS NECESSARY IN SOME MPI VERSIONS

    // REPORT
    if(printid > -1) {
      if(myid == sendid) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == recvid) {
        MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == printid) {
        T* sendbuf_sendid;
        T* recvbuf_recvid;
        size_t sendoffset_sendid;
        size_t recvoffset_recvid;
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&recvbuf_recvid, sizeof(T*), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&recvoffset_recvid, sizeof(size_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
        printf("Bench %d comm %d (%d->%d) sendbuf %p sendoffset %zu recvbuf %p recvoffset %zu count %zu (", benchid, numcomm, sendid, recvid, sendbuf_sendid, sendoffset_sendid, recvbuf_recvid, recvoffset_recvid, count);
        print_data(count * sizeof(T));
        printf(") ");
        print_lib(lib);
        printf("\n");
      }
    }
    numcomm++;

    // SENDER DATA STRUCTURES
    if(myid == sendid) {

      // EXTEND REGISTRY
      this->sendbuf.push_back(sendbuf);
      this->sendproc.push_back(recvid);
      this->sendcount.push_back(count);
      this->sendoffset.push_back(sendoffset);

      // SETUP CAPABILITY
      switch(lib) {
        case dummy:
          break;
        case MPI:
          sendrequest.push_back(MPI_Request());
          break;
        case XCCL:
          break;
        case IPC:
          ack_sender.push_back(int());
          remotebuf.push_back(recvbuf);
          remoteoffset.push_back(recvoffset);
          // CREATE STREAMS
#ifdef PORT_CUDA
          stream_ipc.push_back(cudaStream_t());
          cudaStreamCreate(&stream_ipc[numsend]);
#elif defined PORT_HIP
          stream_ipc.push_back(hipStream_t());
          hipStreamCreate(&stream_ipc[numsend]);
#elif defined PORT_SYCL
          q_ipc.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
          // RECIEVE REMOTE MEMORY HANDLE
          if(sendid != recvid) {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            MPI_Recv(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = cudaIpcOpenMemHandle((void**)&remotebuf[numsend], memhandle, cudaIpcMemLazyEnablePeerAccess);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            MPI_Recv(&memhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = hipIpcOpenMemHandle((void**)&remotebuf[numsend], memhandle, hipIpcMemLazyEnablePeerAccess);
#elif defined PORT_SYCL
            ze_ipc_mem_handle_t memhandle;
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
              MPI_Recv(&what_intel_should_have_done, sizeof(clone_mem_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
              int pidfd = syscall(SYS_pidfd_open,what_intel_should_have_done.pid,0);
              //      int myfd  = syscall(SYS_pidfd_getfd,pidfd,what_intel_should_have_done.fd,0);
              int myfd  = syscall(438,pidfd,what_intel_should_have_done.fd,0);
	      memcpy((void *)&memhandle,(void *)&myfd,sizeof(int));
            }
            // MPI_Recv(&memhandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
	    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_ipc[numsend].get_context());
	    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_ipc[numsend].get_device());
	    error = zeMemOpenIpcHandle(zeContext, zeDevice, memhandle, 0, (void**)&remotebuf[numsend]);
#endif
            if(error)
              printf("IpcOpenMemHandle error %d\n", error);
            MPI_Recv(&remoteoffset[numsend], sizeof(size_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
          }
          break;
        case IPC_get:
          ack_sender.push_back(int());
          // SEND REMOTE MEMORY HANDLE
          if(sendid != recvid)
          {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            error = cudaIpcGetMemHandle(&memhandle, sendbuf);
            MPI_Send(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, recvid, 0, comm_mpi);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            error = hipIpcGetMemHandle(&memhandle, sendbuf);
            MPI_Send(&memhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, recvid, 0, comm_mpi);
#elif defined PORT_SYCL
            ze_ipc_mem_handle_t memhandle;
            auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
            error = zeMemGetIpcHandle(zeContext, sendbuf, &memhandle);
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
              memcpy((void *)&what_intel_should_have_done.fd,(void *)&memhandle,sizeof(int));
              what_intel_should_have_done.pid = getpid();
              MPI_Send(&what_intel_should_have_done, sizeof(clone_mem_t), MPI_BYTE, recvid, 0, comm_mpi);
            }
            // MPI_Send(&memhandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE, sendid, 0, comm_mpi);
#endif
            if(error)
              printf("IpcGetMemHandle error %d\n", error);
            MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, recvid, 0, comm_mpi);
          }
          break;
        case numlib:
          break;
      } // switch(lib)
      numsend++;
    }

    // RECEIVER DATA STRUCTURES
    if(myid == recvid) {

      // EXTEND REGISTRY
      this->recvbuf.push_back(recvbuf);
      this->recvproc.push_back(sendid);
      this->recvcount.push_back(count);
      this->recvoffset.push_back(recvoffset);

      // SETUP LIBRARY
      switch(lib) {
        case dummy:
          break;
        case MPI:
          recvrequest.push_back(MPI_Request());
          break;
        case XCCL:
          break;
        case IPC:
          ack_recver.push_back(int());
          // SEND REMOTE MEMORY HANDLE
          if(sendid != recvid)
          {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            error = cudaIpcGetMemHandle(&memhandle, recvbuf);
            MPI_Send(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, sendid, 0, comm_mpi);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            error = hipIpcGetMemHandle(&memhandle, recvbuf);
            MPI_Send(&memhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, sendid, 0, comm_mpi);
#elif defined PORT_SYCL
            ze_ipc_mem_handle_t memhandle;
	    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	    error = zeMemGetIpcHandle(zeContext, recvbuf, &memhandle);
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
	      memcpy((void *)&what_intel_should_have_done.fd,(void *)&memhandle,sizeof(int));
	      what_intel_should_have_done.pid = getpid();
              MPI_Send(&what_intel_should_have_done, sizeof(clone_mem_t), MPI_BYTE, sendid, 0, comm_mpi);
            }
            // MPI_Send(&memhandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE, sendid, 0, comm_mpi);
#endif
            if(error)
              printf("IpcGetMemHandle error %d\n", error);
            MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, sendid, 0, comm_mpi);
          }
          break;
        case IPC_get:
          ack_recver.push_back(int());
          remotebuf.push_back(sendbuf);
          remoteoffset.push_back(sendoffset);
          // CREATE STREAMS
#ifdef PORT_CUDA
          stream_ipc.push_back(cudaStream_t());
          cudaStreamCreate(&stream_ipc[numrecv]);
#elif defined PORT_HIP
          stream_ipc.push_back(hipStream_t());
          hipStreamCreate(&stream_ipc[numrecv]);
#elif defined PORT_SYCL
          q_ipc.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
          // RECV REMOTE MEMORY HANDLE
          if(sendid != recvid) {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            MPI_Recv(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = cudaIpcOpenMemHandle((void**)&remotebuf[numrecv], memhandle, cudaIpcMemLazyEnablePeerAccess);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            MPI_Recv(&memhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = hipIpcOpenMemHandle((void**)&remotebuf[numrecv], memhandle, hipIpcMemLazyEnablePeerAccess);
#elif defined PORT_SYCL
            ze_ipc_mem_handle_t memhandle;
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
              MPI_Recv(&what_intel_should_have_done, sizeof(clone_mem_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
              int pidfd = syscall(SYS_pidfd_open,what_intel_should_have_done.pid,0);
              //      int myfd  = syscall(SYS_pidfd_getfd,pidfd,what_intel_should_have_done.fd,0);
              int myfd  = syscall(438,pidfd,what_intel_should_have_done.fd,0);
              memcpy((void *)&memhandle,(void *)&myfd,sizeof(int));
            }
            // MPI_Recv(&memhandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
            auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_ipc[numrecv].get_context());
            auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_ipc[numrecv].get_device());
            error = zeMemOpenIpcHandle(zeContext, zeDevice, memhandle, 0, (void**)&remotebuf[numrecv]);
#endif
            if(error)
              printf("IpcOpenMemHandle error %d\n", error);
            MPI_Recv(&remoteoffset[numrecv], sizeof(size_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
          }
          break;
        case numlib:
          break;
      } // switch(lib)
      numrecv++;
    }
  }

  template <typename T>
  void Comm<T>::measure(int warmup, int numiter) {
    long count_total = 0;
    for(int send = 0; send < numsend; send++)
       count_total += sendcount[send];
    MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_LONG, MPI_SUM, comm_mpi);
    measure(warmup, numiter, count_total);
  }
  template <typename T>
  void Comm<T>::measure(int warmup, int numiter, size_t count) {
    this->report();
    double minTime;
    double medTime;
    double maxTime;
    double avgTime;
    CommBench::measure(warmup, numiter, minTime, medTime, maxTime, avgTime, *this);
    if(myid == printid) {
      size_t data = count * sizeof(T);
      printf("data: "); print_data(data); printf("\n");
      printf("minTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e12, data / minTime / 1e9);
      printf("medTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e12, data / medTime / 1e9);
      printf("maxTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e12, data / maxTime / 1e9);
      printf("avgTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e12, data / avgTime / 1e9);
      printf("\n");
    }
  };

  template <typename T>
  void Comm<T>::report() {

    std::vector<size_t> matrix;
    getMatrix(matrix);

    if(myid == printid) {
      printf("\nCommBench %d: ", benchid);
      print_lib(lib);
      printf(" communication matrix (reciever x sender): %d\n", numcomm);
      for(int recver = 0; recver < numproc; recver++) {
        for(int sender = 0; sender < numproc; sender++) {
          size_t count = matrix[sender * numproc + recver];
          if(count)
            // printf("%ld ", count);
            printf("1 ");
          else
            printf(". ");
        }
        printf("\n");
      }
    }
    long sendTotal = 0;
    long recvTotal = 0;
    for(int send = 0; send < numsend; send++)
       sendTotal += sendcount[send];
    for(int recv = 0; recv < numrecv; recv++)
       recvTotal += recvcount[recv];

    MPI_Allreduce(MPI_IN_PLACE, &sendTotal, 1, MPI_LONG, MPI_SUM, comm_mpi);
    MPI_Allreduce(MPI_IN_PLACE, &recvTotal, 1, MPI_LONG, MPI_SUM, comm_mpi);

    int total_buff = buffer_list.size();
    std::vector<int> total_buffs(numproc);
    MPI_Allgather(&total_buff, 1, MPI_INT, total_buffs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    size_t total_count = 0;
    for(size_t count : buffer_count)
      total_count += count;
    std::vector<size_t> total_counts(numproc);
    MPI_Allgather(&total_count, sizeof(size_t), MPI_BYTE, total_counts.data(), sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);
    if(myid == printid) {
      for(int p = 0; p < numproc; p++) {
        printf("proc %d: %d pieces count %ld ", p, total_buffs[p], total_counts[p]);
        print_data(total_counts[p] * sizeof(T));
        printf("\n");
      }
    }

    if(myid == printid) {
      printf("send footprint: %ld ", sendTotal);
      print_data(sendTotal * sizeof(T));
      printf("\n");
      printf("recv footprint: %ld ", recvTotal);
      print_data(recvTotal * sizeof(T));
      printf("\n");
      printf("\n");
    }
  }
  template <typename T>
  void Comm<T>::getMatrix(std::vector<size_t> &matrix) {

    std::vector<size_t> sendcount_temp(numproc, 0);
    std::vector<size_t> recvcount_temp(numproc, 0);
    for (int send = 0; send < numsend; send++)
      sendcount_temp[sendproc[send]] += sendcount[send];
    for (int recv = 0; recv < numrecv; recv++)
      recvcount_temp[recvproc[recv]] += recvcount[recv];
    std::vector<size_t> sendmatrix(numproc * numproc);
    std::vector<size_t> recvmatrix(numproc * numproc);
    MPI_Allgather(sendcount_temp.data(), numproc * sizeof(size_t), MPI_BYTE, sendmatrix.data(), numproc * sizeof(size_t), MPI_BYTE, comm_mpi);
    MPI_Allgather(recvcount_temp.data(), numproc * sizeof(size_t), MPI_BYTE, recvmatrix.data(), numproc * sizeof(size_t), MPI_BYTE, comm_mpi);

    for (int sender = 0; sender < numproc; sender++)
      for (int recver = 0; recver < numproc; recver++)
        matrix.push_back(sendmatrix[sender * numproc + recver]);

    /* if(myid == printid) {
      char filename[2048];
      sprintf(filename, "matrix_%d.txt", benchid);
      FILE *matfile = fopen(filename, "w");
      for(int recver = 0; recver < numproc; recver++) {
        for(int sender = 0; sender < numproc; sender++)
          fprintf(matfile, "%ld ", matrix[sender * numproc + recver]);
        fprintf(matfile, "\n");
      }
      fclose(matfile);
    }*/
  }

  template <typename T>
  void Comm<T>::start() {
    switch(lib) {
      case MPI:
        for (int send = 0; send < numsend; send++)
          MPI_Isend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm_mpi, &sendrequest[send]);
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm_mpi, &recvrequest[recv]);
        break;
      case XCCL:
#ifdef CAP_NCCL
        ncclGroupStart();
        for(int send = 0; send < numsend; send++)
          ncclSend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), ncclInt8, sendproc[send], comm_nccl, stream_nccl);
        for(int recv = 0; recv < numrecv; recv++)
          ncclRecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), ncclInt8, recvproc[recv], comm_nccl, stream_nccl);
        ncclGroupEnd();
#elif defined CAP_ONECCL
        for (int i = 0; i < numsend; i++)
          ccl::send<T>(sendbuf[i] + sendoffset[i], sendcount[i], sendproc[i], *comm_ccl, *stream_ccl);
        for (int i = 0; i < numrecv; i++)
          ccl::recv<T>(recvbuf[i] + recvoffset[i], recvcount[i], recvproc[i], *comm_ccl, *stream_ccl);
#endif
        break;
      case IPC:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_SYCL
          q_ipc[send].memcpy(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
#endif
        }
        break;
      case IPC_get:
        for(int send = 0; send < numsend; send++)
          MPI_Send(&ack_sender[send], 1, MPI_INT, sendproc[send], 0, comm_mpi);
        for(int recv = 0; recv < numrecv; recv++) {
          MPI_Recv(&ack_recver[recv], 1, MPI_INT, recvproc[recv], 0, comm_mpi, MPI_STATUS_IGNORE);
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[recv]);
#elif defined PORT_HIP
          hipMemcpyAsync(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[recv]);
#elif defined PORT_SYCL
          q_ipc[recv].memcpy(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T));
#endif
        }
        break;
      default:
        break;
    }
  }

  template <typename T>
  void Comm<T>::wait() {
    switch(lib) {
      case MPI:
        MPI_Waitall(numsend, sendrequest.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(numrecv, recvrequest.data(), MPI_STATUSES_IGNORE);
        break;
      case XCCL:
#if defined CAP_NCCL && defined PORT_CUDA
        cudaStreamSynchronize(stream_nccl);
#elif defined CAP_NCCL && defined PORT_HIP
        hipStreamSynchronize(stream_nccl);
#elif defined CAP_ONECCL
        q.wait();
#endif
        break;
      case IPC:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[send]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_ipc[send]);
#elif defined PORT_SYCL
	  q_ipc[send].wait();
#endif
          MPI_Send(&ack_sender[send], 1, MPI_INT, sendproc[send], 0, comm_mpi);
        }
        for(int recv = 0; recv < numrecv; recv++)
          MPI_Recv(&ack_recver[recv], 1, MPI_INT, recvproc[recv], 0, comm_mpi, MPI_STATUS_IGNORE);
        break;
      case IPC_get:
        for(int recv = 0; recv < numrecv; recv++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[recv]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_ipc[recv]);
#elif defined PORT_SYCL
          q_ipc[recv].wait();
#endif
        }
        break;
      default:
        break;
    }
  }

  static void print_stats(std::vector<double> times, size_t data) {

    std::sort(times.begin(), times.end(),  [](const double & a, const double & b) -> bool {return a < b;});

    int numiter = times.size();

    if(myid == printid) {
      printf("%d measurement iterations (sorted):\n", numiter);
      for(int iter = 0; iter < numiter; iter++) {
        printf("time: %.4e", times[iter] * 1e6);
        if(iter == 0)
          printf(" -> min\n");
        else if(iter == numiter / 2)
          printf(" -> median\n");
        else if(iter == numiter - 1)
          printf(" -> max\n");
        else
          printf("\n");
      }
      printf("\n");
    }
    double minTime = times[0];
    double medTime = times[numiter / 2];
    double maxTime = times[numiter - 1];
    double avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;
    if(myid == printid) {
      printf("data: "); print_data(data); printf("\n");
      printf("minTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e12, data / minTime / 1e9);
      printf("medTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e12, data / medTime / 1e9);
      printf("maxTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e12, data / maxTime / 1e9);
      printf("avgTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e12, data / avgTime / 1e9);
      printf("\n");
    }
  }

  template <typename T>
  static void measure_async(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count) {
    std::vector<double> t;
    for(int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(comm_mpi);
      double time = MPI_Wtime();
      for (auto &i : commlist) {
        i.start();
        i.wait();
      }
      time = MPI_Wtime() - time;
      MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      if(iter >= 0)
        t.push_back(time);
    }
    print_stats(t, count * sizeof(T));
  }

  template <typename T>
  static void measure_concur(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count) {
    std::vector<double> t;
    for(int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(comm_mpi);
      double time = MPI_Wtime();
      for (auto &i : commlist) {
        i.start();
      }
      for (auto &i : commlist) {      
        i.wait();
      }
      time = MPI_Wtime() - time;
      MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      if(iter >= 0)
        t.push_back(time);
    }
    print_stats(t, count * sizeof(T));
  }

  template <typename T>
  static void measure_MPI_Alltoallv(std::vector<std::vector<int>> pattern, int warmup, int numiter) {

    std::vector<int> sendcount;
    std::vector<int> recvcount;
    for(int i = 0; i < numproc; i++) {
      sendcount.push_back(pattern[myid][i]);
      recvcount.push_back(pattern[i][myid]);
    }
    std::vector<int> senddispl(numproc + 1, 0);
    std::vector<int> recvdispl(numproc + 1, 0);
    for(int i = 1; i < numproc + 1; i++) {
      senddispl[i] = senddispl[i-1] + sendcount[i-1];
      recvdispl[i] = recvdispl[i-1] + recvcount[i-1];

    }

    //for(int i = 0; i < numproc; i++)
    //  printf("myid %d i: %d sendcount %d senddispl %d recvcount %d recvdispl %d\n", myid, i, sendcount[i], senddispl[i], recvcount[i], recvdispl[i]);

    T *sendbuf;
    T *recvbuf;
    allocate(sendbuf, senddispl[numproc]);
    allocate(recvbuf, recvdispl[numproc]);
    for(int p = 0; p < numproc; p++) {
      sendcount[p] *= sizeof(T);
      recvcount[p] *= sizeof(T);
      senddispl[p] *= sizeof(T);
      recvdispl[p] *= sizeof(T);
    }

    std::vector<double> t;
    for(int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(comm_mpi);
      double time = MPI_Wtime();
      MPI_Alltoallv(sendbuf, &sendcount[0], &senddispl[0], MPI_BYTE, recvbuf, &recvcount[0], &recvdispl[0], MPI_BYTE, comm_mpi);
      time = MPI_Wtime() - time;
      MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      if(iter >= 0)
        t.push_back(time);
    }

    free(sendbuf);
    free(recvbuf);
    int data;
    MPI_Allreduce(&senddispl[numproc], &data, 1, MPI_INT, MPI_SUM, comm_mpi);
    print_stats(t, data * sizeof(T));
  }

  template <typename C>
  static void measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime, C comm) {

    double times[numiter];
    double starts[numiter];

    if(myid == printid)
      printf("%d warmup iterations (in order):\n", warmup);
    for (int iter = -warmup; iter < numiter; iter++) {
      for(int send = 0; send < comm.numsend; send++) {
#if defined PORT_CUDA
        // cudaMemset(sendbuf[send], -1, sendcount[send] * sizeof(T));
#elif defined PORT_HIP
        // hipMemset(sendbuf[send], -1, sendcount[send] * sizeof(T));
#elif defined PORT_SYCL
        // q->memset(sendbuf[send], -1, sendcount[send] * sizeof(T)).wait();
#else
        // memset(comm.sendbuf[send], -1, comm.sendcount[send] * sizeof(T)); // NECESSARY FOR CPU TO PREVENT CACHING
#endif
      }
      MPI_Barrier(comm_mpi);
      double time = MPI_Wtime();
      comm.start();
      double start = MPI_Wtime() - time;
      comm.wait();
      time = MPI_Wtime() - time;
      MPI_Allreduce(MPI_IN_PLACE, &start, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      if(iter < 0) {
        if(myid == printid)
          printf("startup %.2e warmup: %.2e\n", start * 1e6, time * 1e6);
      }
      else {
        starts[iter] = start;
        times[iter] = time;
      }
    }
    std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});
    std::sort(starts, starts + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

    if(myid == printid) {
      printf("%d measurement iterations (sorted):\n", numiter);
      for(int iter = 0; iter < numiter; iter++) {
        printf("start: %.4e time: %.4e", starts[iter] * 1e6, times[iter] * 1e6);
        if(iter == 0)
          printf(" -> min\n");
        else if(iter == numiter / 2)
          printf(" -> median\n");
        else if(iter == numiter - 1)
          printf(" -> max\n");
        else
          printf("\n");
      }
      printf("\n");
    }
    minTime = times[0];
    medTime = times[numiter / 2];
    maxTime = times[numiter - 1];
    avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;
  }

  // MEMORY MANAGEMENT
  size_t memory = 0;
  void report_memory() {
    std::vector<size_t> memory_all(numproc);
    MPI_Allgather(&memory, sizeof(size_t), MPI_BYTE, memory_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
    if(myid == printid) {
      size_t memory_total = 0;
      printf("\n");
      printf("CommBench memory report:\n");
      for(int i = 0; i < numproc; i++) {
        printf("proc: %d memory ", i);
        print_data(memory_all[i]);
        printf("\n");
        memory_total += memory_all[i];
      }
      printf("total memory: ");
      print_data(memory_total);
      printf("\n");
      printf("\n");
    }
  }

  template <typename T>
  void allocate(T *&buffer, size_t n) {
#ifdef PORT_CUDA
    cudaMalloc(&buffer, n * sizeof(T));
#elif defined PORT_HIP
    hipMalloc(&buffer, n * sizeof(T));
#elif defined PORT_SYCL
    buffer = sycl::malloc_device<T>(n, CommBench::q);
#else
    allocateHost(buffer, n);
#endif
    memory += n * sizeof(T);
  };

  template <typename T>
  void allocateHost(T *&buffer, size_t n) {
#ifdef PORT_CUDA
    cudaMallocHost(&buffer, n * sizeof(T));
#elif defined PORT_HIP
    hipHostMalloc(&buffer, n * sizeof(T));
#elif defined PORT_SYCL
    buffer = sycl::malloc_host<T>(n, CommBench::q);
#else
    buffer = new T[n];
#endif
  }

  template <typename T>
  void memcpyD2D(T *recvbuf, T *sendbuf, size_t n) {
#ifdef PORT_CUDA
    cudaMemcpy(recvbuf, sendbuf, n * sizeof(T), cudaMemcpyDeviceToDevice);
#elif defined PORT_HIP
    hipMemcpy(recvbuf, sendbuf, n * sizeof(T), hipMemcpyDeviceToDevice);
#elif defined PORT_SYCL
    CommBench::q.memcpy(recvbuf, sendbuf, n * sizeof(T)).wait();
#else
    memcpy(recvbuf, sendbuf, n * sizeof(T));
#endif
  }

  template <typename T>
  void memcpyH2D(T *device, T *host, size_t n) {
#ifdef PORT_CUDA
    cudaMemcpy(device, host, n * sizeof(T), cudaMemcpyHostToDevice);
#elif defined PORT_HIP  
    hipMemcpy(device, host, n * sizeof(T), hipMemcpyHostToDevice);
#elif defined PORT_SYCL
    CommBench::q.memcpy(device, host, n * sizeof(T)).wait();
#else
    memcpy(device, host, n * sizeof(T));
#endif
  }

  template <typename T>
  void memcpyD2H(T *host, T *device, size_t n) {
#ifdef PORT_CUDA
    cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost);
#elif defined PORT_HIP
    hipMemcpy(host, device, n * sizeof(T), hipMemcpyDeviceToHost);
#elif defined PORT_SYCL
    CommBench::q.memcpy(host, device, n * sizeof(T)).wait();
#else
    memcpy(host, device, n * sizeof(T));
#endif
  }

  template <typename T>
  void free(T *buffer) {
#ifdef PORT_CUDA
    cudaFree(buffer);
#elif defined PORT_HIP
    hipFree(buffer);
#elif defined PORT_SYCL
    sycl::free(buffer, CommBench::q);
#else
    freeHost(buffer);
#endif
  }

  template <typename T>
  void freeHost(T *buffer) {
#ifdef PORT_CUDA
    cudaFreeHost(buffer);
#elif defined PORT_HIP
    hipHostFree(buffer);
#elif defined PORT_SYCL
    sycl::free(buffer, CommBench::q);
#else
    delete[] buffer;
#endif
  }

} // namespace CommBench
#endif
