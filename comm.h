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

#ifdef PORT_CUDA
#include <nccl.h>
#elif defined PORT_HIP
#include <rccl.h>
#elif defined PORT_SYCL
#include <sycl.hpp>
#include <ze_api.h>
#endif

#include <stdio.h> // for printf
#include <string.h> // for memcpy
#include <algorithm> // for std::sort
#include <vector>

#if defined(PORT_CUDA) || defined(PORT_HIP)
#define CAP_NCCL
#endif

namespace CommBench
{
  static int printid = -1;

  enum library {null, MPI, NCCL, IPC, STAGE, numlib};

  static MPI_Comm comm_mpi;
#ifdef CAP_NCCL
  static ncclComm_t comm_nccl;
#endif
#ifdef PORT_SYCL
  static sycl::queue q(sycl::gpu_selector_v);
#endif
  static bool initialized = false;

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
      case null : printf("NULL"); break;
      case IPC  : printf("IPC"); break;
      case MPI  : printf("MPI"); break;
      case NCCL : printf("NCCL"); break;
      case STAGE  : printf("STAGE"); break;
      case numlib : printf("NUMLIB"); break;
    }
  }

  template <typename T>
  class Comm {

    public :

    const library lib;

    // MPI
    MPI_Request *sendrequest;
    MPI_Request *recvrequest;

    // NCCL
#ifdef PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined PORT_HIP
    hipStream_t stream_nccl;
#endif

    // IPC
    T **recvbuf_ipc;
    size_t *recvoffset_ipc;
    bool *ack_sender;
    bool *ack_recver;
#ifdef PORT_CUDA
    cudaStream_t *stream_ipc;
#elif defined PORT_HIP
    hipStream_t *stream_ipc;
#elif defined PORT_SYCL
    sycl::queue *q_ipc;
#endif

    // STAGE
#ifdef PORT_CUDA
    std::vector<cudaStream_t> stream_stage;
#elif defined PORT_HIP
    std::vector<hipStream_t> stream_stage;
#elif defined PORT_SYCL
    std::vector<sycl::queue> q_stage;
#endif

    int numsend;
    int numrecv;
    T **sendbuf;
    T **recvbuf;
    int *sendproc;
    int *recvproc;
    size_t *sendcount;
    size_t *recvcount;
    size_t *sendoffset;
    size_t *recvoffset;

    Comm(library lib);

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void start();
    void wait();

    void measure(int warmup, int numiter, double &minTime, double &medTime, double &avgTime, double &maxTime);
    void measure(int warmup, int numiter);
    void measure(int warmup, int numiter, size_t data);
    void measure_count(int warmup, int numiter, size_t data);
    void report();
  };

  template <typename T>
  Comm<T>::Comm(library lib) : lib(lib) {

    if(!initialized)
      MPI_Comm_dup(MPI_COMM_WORLD, &comm_mpi); // CREATE SEPARATE COMMUNICATOR EXPLICITLY

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    if(!initialized)
      if(myid == printid)
        printf("******************** MPI COMMUNICATOR IS CREATED\n");

    numsend = 0;
    numrecv = 0;

    if(myid == printid) {
      printf("Create Comm with %d processors\n", numproc);
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
    if(lib == NCCL) {
      if(!initialized) {
#ifdef CAP_NCCL
        ncclUniqueId id;
        if(myid == 0)
          ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm_mpi);
        ncclCommInitRank(&comm_nccl, numproc, id, myid);
        if(myid == printid)
          printf("******************** NCCL COMMUNICATOR IS CREATED\n");
#endif
      }
#ifdef PORT_CUDA
      cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
      hipStreamCreate(&stream_nccl);
#endif
    }
    initialized = true;
  }

  template <typename T>
  void Comm<T>::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    int sendid_temp = sendid;
    int recvid_temp = recvid;

    // REPORT
    if(printid > -1 && printid < numproc) {
      int sendid_temp = (sendid == -1 ? recvid : sendid);
      int recvid_temp = (recvid == -1 ? sendid : recvid);
      if(myid == sendid_temp) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, printid, 0, MPI_COMM_WORLD);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, printid, 0, MPI_COMM_WORLD);
      }
      if(myid == recvid_temp) {
        MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, printid, 0, MPI_COMM_WORLD);
        MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, printid, 0, MPI_COMM_WORLD);
      }
      if(myid == printid) {
        T* sendbuf_sendid;
        T* recvbuf_recvid;
        size_t sendoffset_sendid;
        size_t recvoffset_recvid;
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid_temp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid_temp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&recvbuf_recvid, sizeof(T*), MPI_BYTE, recvid_temp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&recvoffset_recvid, sizeof(size_t), MPI_BYTE, recvid_temp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("add (%d -> %d) sendbuf %p sendoffset %zu recvbuf %p recvoffset %zu count %zu ( ", sendid, recvid, sendbuf_sendid, sendoffset_sendid, recvbuf_recvid, recvoffset_recvid, count);
        print_data(count * sizeof(T));
        printf(" ) ");
        print_lib(lib);
        printf("\n");
      }
    }

    if(myid == sendid || (sendid == -1 && myid == recvid)) {
      // ALLOCATE NEW BUFFER
      T **sendbuf_temp = new T*[numsend + 1];
      int *sendproc_temp = new int[numsend + 1];
      size_t *sendcount_temp = new size_t[numsend + 1];
      size_t *sendoffset_temp = new size_t[numsend + 1];
      // COPY OLD BUFFER
      memcpy(sendbuf_temp, this->sendbuf, numsend * sizeof(T*));
      memcpy(sendproc_temp, this->sendproc, numsend * sizeof(int));
      memcpy(sendcount_temp, this->sendcount, numsend * sizeof(size_t));
      memcpy(sendoffset_temp, this->sendoffset, numsend * sizeof(size_t));
      // DELETE OLD BUFFER
      if(numsend) {
        delete[] this->sendbuf;
        delete[] this->sendproc;
        delete[] this->sendcount;
        delete[] this->sendoffset;
      }
      this->sendbuf = sendbuf_temp;
      this->sendproc = sendproc_temp;
      this->sendcount = sendcount_temp;
      this->sendoffset = sendoffset_temp;
      // EXTEND + 1
      this->sendbuf[numsend] = sendbuf;
      this->sendproc[numsend] = recvid;
      this->sendcount[numsend] = count;
      this->sendoffset[numsend] = sendoffset;

      // SETUP CAPABILITY
      switch(lib) {
        case null:
          break;
        case MPI:
          if(numsend) delete[] sendrequest;
          sendrequest = new MPI_Request[numsend + 1];
          break;
        case NCCL:
          break;
        case IPC:
          if(numsend) {
            delete[] sendrequest;
            delete[] ack_sender;
          }
          sendrequest = new MPI_Request[numsend + 1];
          ack_sender = new bool[numsend + 1];
          // RECV REMOTE MEMORY HANDLE
	  {
            T **recvbuf_ipc = new T*[numsend + 1];
            size_t *recvoffset_ipc = new size_t[numsend + 1];
            if(numsend) {
              memcpy(recvbuf_ipc, this->recvbuf_ipc, numsend * sizeof(T*));
              memcpy(recvoffset_ipc, this->recvoffset_ipc, numsend * sizeof(size_t));
              delete[] this->recvbuf_ipc;
              delete[] this->recvoffset_ipc;
            }
            this->recvbuf_ipc = recvbuf_ipc;
            this->recvoffset_ipc = recvoffset_ipc;
          }
          if(sendid == recvid) {
            recvbuf_ipc[numsend] = recvbuf;
            recvoffset_ipc[numsend] = recvoffset;
          }
          else {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            MPI_Recv(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = cudaIpcOpenMemHandle((void**) recvbuf_ipc + numsend, memhandle, cudaIpcMemLazyEnablePeerAccess);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            MPI_Recv(&memhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = hipIpcOpenMemHandle((void**) recvbuf_ipc + numsend, memhandle, hipIpcMemLazyEnablePeerAccess);
#elif defined PORT_SYCL
            // MPI_Recv(recvbuf_ipc + numsend, sizeof(T*), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
            error = 0;
#endif
            if(error) {
              printf("IpcOpenMemHandle error %d\n", error);
              return;
            }
            MPI_Recv(recvoffset_ipc + numsend, sizeof(size_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
          }
          {
#ifdef PORT_CUDA
            cudaStream_t *stream_ipc = new cudaStream_t[numsend + 1];
            cudaStreamCreate(stream_ipc + numsend);
            if(numsend) {
              memcpy(stream_ipc, this->stream_ipc, numsend * sizeof(cudaStream_t));
              delete[] this->stream_ipc;
            }
            this->stream_ipc = stream_ipc;
#elif defined PORT_HIP
            hipStream_t *stream_ipc = new hipStream_t[numsend + 1];
            hipStreamCreate(stream_ipc + numsend);
            if(numsend) {
              memcpy(stream_ipc, this->stream_ipc, numsend * sizeof(hipStream_t));
              delete[] this->stream_ipc;
            }
            this->stream_ipc = stream_ipc;
#endif
          }
          break;
        case STAGE:
#ifdef PORT_CUDA
          stream_stage.push_back(cudaStream_t());
          cudaStreamCreate(&stream_stage[numsend]);
#elif defined PORT_HIP
          stream_stage.push_back(hipStream_t());
          hipStreamCreate(&stream_stage[numsend]);
#elif defined PORT_SYCL
          q_stage.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
          break;
        case numlib:
          break;
      } // switch(lib)
      numsend++;
    }

    if(myid == recvid || recvid == -1 && myid == sendid) {
      // ALLOCATE NEW BUFFER
      T **recvbuf_temp = new T*[numrecv + 1];
      int *recvproc_temp = new int[numrecv + 1];
      size_t *recvcount_temp = new size_t[numrecv + 1];
      size_t *recvoffset_temp = new size_t[numrecv + 1];
      // COPY OLD BUFFER
      memcpy(recvbuf_temp, this->recvbuf, numrecv * sizeof(T*));
      memcpy(recvproc_temp, this->recvproc, numrecv * sizeof(int));
      memcpy(recvcount_temp, this->recvcount, numrecv * sizeof(size_t));
      memcpy(recvoffset_temp, this->recvoffset, numrecv * sizeof(size_t));
      // DELETE OLD BUFFER
      if(numrecv) {
        delete[] this->recvbuf;
        delete[] this->recvproc;
        delete[] this->recvcount;
        delete[] this->recvoffset;
      }
      this->recvbuf = recvbuf_temp;
      this->recvproc = recvproc_temp;
      this->recvcount = recvcount_temp;
      this->recvoffset = recvoffset_temp;
      // EXTEND + 1
      this->recvbuf[numrecv] = recvbuf;
      this->recvproc[numrecv] = sendid;
      this->recvcount[numrecv] = count;
      this->recvoffset[numrecv] = recvoffset;
      // SETUP LIBRARY
      switch(lib) {
        case null:
          break;
        case MPI:
          if(numrecv) delete[] recvrequest;
          recvrequest = new MPI_Request[numrecv + 1];
          break;
        case NCCL:
          break;
        case IPC:
          if(numrecv) {
            delete[] recvrequest;
            delete[] ack_recver;
          }
          recvrequest = new MPI_Request[numrecv + 1];
          ack_recver = new bool[numrecv + 1];
          // SEND REMOTE MEMORY HANDLE
          if(sendid != recvid)
          {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t myhandle;
            error = cudaIpcGetMemHandle(&myhandle, recvbuf);
            MPI_Send(&myhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, sendid, 0, comm_mpi);
#elif defined PORT_HIP
            hipIpcMemHandle_t myhandle;
            error = hipIpcGetMemHandle(&myhandle, recvbuf);
            MPI_Send(&myhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, sendid, 0, comm_mpi);
#elif defined PORT_SYCL
            // MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, sendid, 0, comm_mpi);
            error = 0;
#endif
            if(error) {
              printf("IpcGetMemHandle error %d\n", error);
              return;
            }
            MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, sendid, 0, comm_mpi);
          }
          break;
        case STAGE:
          break;
        case numlib:
          break;
      } // switch(lib)
      numrecv++;
    }
  }

  template <typename T>
  void Comm<T>::measure(int warmup, int numiter) {
    measure(warmup, numiter, 0);
  }

  template <typename T>
  void Comm<T>::measure(int warmup, int numiter, size_t count) {
    if(count == 0) {
      double count_total;
      for(int send = 0; send < numsend; send++)
         count_total += sendcount[send];
      MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_DOUBLE, MPI_SUM, comm_mpi);
      measure_count(warmup, numiter, count_total);
    }
    else
      measure_count(warmup, numiter, count);
  }

  template <typename T>
  void Comm<T>::measure_count(int warmup, int numiter, size_t count) {

    int myid;
    MPI_Comm_rank(comm_mpi, &myid);

    this->report();

    double minTime;
    double medTime;
    double maxTime;
    double avgTime;
    this->measure(warmup, numiter, minTime, medTime, maxTime, avgTime);

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
  void Comm<T>::measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime) {

    double times[numiter];
    double starts[numiter];
    int myid;
    MPI_Comm_rank(comm_mpi, &myid);

    if(myid == printid)
      printf("%d warmup iterations (in order):\n", warmup);
    for (int iter = -warmup; iter < numiter; iter++) {
      for(int send = 0; send < numsend; send++) {
#if defined PORT_CUDA
        // cudaMemset(sendbuf[send], -1, sendcount[send] * sizeof(T));
#elif defined PORT_HIP
        // hipMemset(sendbuf[send], -1, sendcount[send] * sizeof(T));
#elif defined PORT_SYCL
	// q->memset(sendbuf[send], -1, sendcount[send] * sizeof(T)).wait();
#else
        memset(sendbuf[send], -1, sendcount[send] * sizeof(T)); // NECESSARY FOR CPU TO PREVENT CACHING
#endif
      }
      MPI_Barrier(comm_mpi);
      double time = MPI_Wtime();
      this->start();
      double start = MPI_Wtime() - time;
      this->wait();
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

  template <typename T>
  void Comm<T>::report() {
    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    int sendmatrix[numproc][numproc];
    int recvmatrix[numproc][numproc];
    memset(sendmatrix, 0, numproc * numproc * sizeof(int));
    memset(recvmatrix, 0, numproc * numproc * sizeof(int));
    for(int send = 0; send < numsend; send++)
      sendmatrix[abs(sendproc[send])][myid]++;
    for(int recv = 0; recv < numrecv; recv++)
      recvmatrix[myid][abs(recvproc[recv])]++;
    MPI_Allreduce(MPI_IN_PLACE, sendmatrix, numproc * numproc, MPI_INT, MPI_SUM, comm_mpi);
    MPI_Allreduce(MPI_IN_PLACE, recvmatrix, numproc * numproc, MPI_INT, MPI_SUM, comm_mpi);

    if(myid == printid) {
      printf("\n");
      print_lib(lib);
      printf(" communication matrix\n");
      for(int recv = 0; recv < numproc; recv++) {
        for(int send = 0; send < numproc; send++)
          if(sendmatrix[recv][send])
            printf("%d ", sendmatrix[recv][send]);
          else
            printf(". ");
        printf("\n");
      }
    }

    double sendTotal = 0;
    double recvTotal = 0;
    for(int send = 0; send < numsend; send++)
       sendTotal += sendcount[send] * sizeof(T);
    for(int recv = 0; recv < numrecv; recv++)
       recvTotal += recvcount[recv] * sizeof(T);

    MPI_Allreduce(MPI_IN_PLACE, &sendTotal, 1, MPI_DOUBLE, MPI_SUM, comm_mpi);
    MPI_Allreduce(MPI_IN_PLACE, &recvTotal, 1, MPI_DOUBLE, MPI_SUM, comm_mpi);

    if(myid == printid) {
      printf("send footprint: ");
      print_data(sendTotal);
      printf("\n");
      printf("recv footprint: ");
      print_data(recvTotal);
      printf("\n");
      printf("\n");
    }
  }

  template <typename T>
  void Comm<T>::start() {
    switch(lib) {
      case MPI:
        for (int send = 0; send < numsend; send++)
          MPI_Isend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm_mpi, sendrequest + send);
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm_mpi, recvrequest + recv);
        break;
      case NCCL:
#ifdef CAP_NCCL
        ncclGroupStart();
        for(int send = 0; send < numsend; send++)
          ncclSend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), ncclInt8, sendproc[send], comm_nccl, stream_nccl);
        for(int recv = 0; recv < numrecv; recv++)
          ncclRecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), ncclInt8, recvproc[recv], comm_nccl, stream_nccl);
        ncclGroupEnd();
#endif
        break;
      case IPC:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_SYCL
          // L0 IPC INITIATE
	  // SELF COMMUNICATION
          q.memcpy(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
#endif
        }
        break;
      case STAGE:
        for(int send = 0; send < numsend; send++)
          if(sendproc[send] == -1)
#ifdef PORT_CUDA
            cudaMemcpyAsync(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyHostToDevice, stream_stage[send]);
          else
            cudaMemcpyAsync(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToHost, stream_stage[send]);
#elif defined PORT_HIP
            hipMemcpyAsync(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyHostToDevice, stream_stage[send]);
          else
            hipMemcpyAsync(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToHost, stream_stage[send]);
#elif defined PORT_SYCL
            q_stage[send].memcpy(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
	  else
            q_stage[send].memcpy(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
#endif
        break;
      default:
        break;
    }
  }

  template <typename T>
  void Comm<T>::wait() {
    switch(lib) {
      case MPI:
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        break;
      case NCCL:
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_nccl);
#elif defined PORT_HIP
        hipStreamSynchronize(stream_nccl);
#endif
        break;
      case IPC:
        for(int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(ack_recver + recv, 1, MPI_C_BOOL, recvproc[recv], 0, comm_mpi, recvrequest + recv);
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[send]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_ipc[send]);
#elif defined PORT_SYCL
          // L0 IPC SYNCHRONIZE
	  // SELF COMMUNICATION
	  q.wait();
#endif
          MPI_Isend(ack_sender + send, 1, MPI_C_BOOL, sendproc[send], 0, comm_mpi, sendrequest + send);
        }
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        break;
      case STAGE:
        for(int recv = 0; recv < numrecv; recv++)
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_stage[recv]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_stage[recv]);
#elif defined PORT_SYCL
          q_stage[recv].wait();
#endif
        break;
      default:
        break;
    }
  }
} // namespace CommBench

#endif
