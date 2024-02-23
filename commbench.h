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

  // class Comm<T>
#include "comm.h"

  // THIS IS TO INITIALIZE COMMBENCH
  static Comm<char> init(dummy);

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
