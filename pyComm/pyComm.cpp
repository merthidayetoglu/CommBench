#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//  #include "../comm.h"

namespace py = pybind11;

#include <mpi.h>
#include <stdio.h> // for printf
#include <string.h> // for memcpy
#include <algorithm> // for std::sort
#include <vector> // for std::vector

#define PORT_CUDA

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
#include <sycl.hpp>
#ifdef CAP_ZE
#include <ze_api.h>
#endif
#endif

#define PORT_CUDA

namespace CommBench {
    static int printid = 0;
    enum library {null, MPI, NCCL, IPC, STAGE, numlib};
    static MPI_Comm comm_mpi;

    void mpi_init();
    void mpi_fin();

    // MEMORY MANAGEMENT
    template <typename T>
    void allocate(T *&buffer,size_t n);
    template <typename T>
    void allocateHost(T *&buffer, size_t n);
    template <typename T>
    void free(T *buffer);
    template <typename T>
    void freeHost(T *buffer);

    static bool initialized_MPI = false;
    static bool initialized_NCCL = false;

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
        public:
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

            // MPI
            std::vector<MPI_Request> sendrequest;
            std::vector<MPI_Request> recvrequest;

            // NCCL
        #ifdef PORT_CUDA
            cudaStream_t stream_nccl;
        #elif defined PORT_HIP
            hipStream_t stream_nccl;
        #endif

            // IPC
            T **recvbuf_ipc;
            size_t *recvoffset_ipc;
            std::vector<int> ack_sender;
            std::vector<int> ack_recver;
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
            const library lib;
            Comm(library lib);
            void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
            void add_lazy(size_t count, int sendid, int recvid);
            void start();
            void wait();
            void measure(int warmup, int numiter, double &minTime, double &medTime, double &avgTime, double &maxTime);
            void measure(int warmup, int numiter);
            void measure(int warmup, int numiter, size_t data);
            void measure_count(int warmup, int numiter, size_t data);
    };
};

void CommBench::mpi_init() {
    MPI_Init(NULL, NULL);
}

void CommBench::mpi_fin() {
    MPI_Finalize();
}

template <typename T>
void Comm<T>::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    if(count == 0)
      return;

    // REPORT
    if(printid > -1) {
      int sendid_temp = (sendid == -1 ? recvid : sendid);
      int recvid_temp = (recvid == -1 ? sendid : recvid);
      if(myid == sendid_temp) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == recvid_temp) {
        MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == printid) {
        T* sendbuf_sendid;
        T* recvbuf_recvid;
        size_t sendoffset_sendid;
        size_t recvoffset_recvid;
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid_temp, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid_temp, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&recvbuf_recvid, sizeof(T*), MPI_BYTE, recvid_temp, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&recvoffset_recvid, sizeof(size_t), MPI_BYTE, recvid_temp, 0, comm_mpi, MPI_STATUS_IGNORE);
        printf("add (%d -> %d) sendbuf %p sendoffset %zu recvbuf %p recvoffset %zu count %zu ( ", 
            sendid, recvid, sendbuf_sendid, sendoffset_sendid, recvbuf_recvid, recvoffset_recvid, count);
        print_data(count * sizeof(T));
        printf(" ) ");
        print_lib(lib);
        printf("\n");
      }
    }

    // SENDER DATA STRUCTURES
    if(myid == sendid || (sendid == -1 && myid == recvid)) {

      // EXTEND REGISTRY
      this->sendbuf.push_back(sendbuf);
      this->sendproc.push_back(recvid);
      this->sendcount.push_back(count);
      this->sendoffset.push_back(sendoffset);

      // SETUP CAPABILITY
      switch(lib) {
        case null:
          break;
        case MPI:
          sendrequest.push_back(MPI_Request());
          break;
        case NCCL:
          break;
        case IPC:
          sendrequest.push_back(MPI_Request());
          ack_sender.push_back(int());
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
	    ze_ipc_mem_handle_t memhandle;
	    void *test;
            MPI_Recv(&memhandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
	    error = zeMemOpenIpcHandle(ctx, dev, memhandle, 0u, &test);
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

    // RECEIVER DATA STRUCTURES
    if(myid == recvid || recvid == -1 && myid == sendid) {

      // EXTEND REGISTRY
      this->recvbuf.push_back(recvbuf);
      this->recvproc.push_back(sendid);
      this->recvcount.push_back(count);
      this->recvoffset.push_back(recvoffset);

      // SETUP LIBRARY
      switch(lib) {
        case null:
          break;
        case MPI:
          recvrequest.push_back(MPI_Request());
          break;
        case NCCL:
          break;
        case IPC:
          recvrequest.push_back(MPI_Request());
          ack_recver.push_back(int());
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
	    void *zeBuffer = nullptr;
            ze_device_mem_alloc_desc_t deviceDesc = {};
            zeMemAllocDevice(ctx, &deviceDesc, 128, 128, dev, &zeBuffer);
            ze_ipc_mem_handle_t myhandle;
            error = zeMemGetIpcHandle(ctx, recvbuf, &myhandle);
            MPI_Send(&myhandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE, sendid, 0, comm_mpi);
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
void Comm<T>::start() {
    switch(lib) {
      case MPI:
        for (int send = 0; send < numsend; send++)
          MPI_Isend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm_mpi, &sendrequest[send]);
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm_mpi, &recvrequest[recv]);
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
          // q.memcpy(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
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
        MPI_Waitall(numsend, sendrequest.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(numrecv, recvrequest.data(), MPI_STATUSES_IGNORE);
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
          MPI_Irecv(&ack_recver[recv], 1, MPI_INT, recvproc[recv], 0, comm_mpi, &recvrequest[recv]);
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
          MPI_Isend(&ack_sender[send], 1, MPI_INT, sendproc[send], 0, comm_mpi, &sendrequest[send]);
        }
        MPI_Waitall(numrecv, recvrequest.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(numsend, sendrequest.data(), MPI_STATUSES_IGNORE);
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

template <typename T>
void CommBench::Comm<T>::add_lazy(size_t count, int sendid, int recvid) {
    T *sendbuf;
    T *recvbuf;
    allocate(sendbuf, count, sendid);
    allocate(recvbuf, count, recvid);
    add(sendbuf, 0, recvbuf, 0, count, sendid, recvid);
}

template <typename T>
void CommBench::Comm<T>::measure(int warmup, int numiter) {
    measure(warmup, numiter, 0);
}

template <typename T>
void CommBench::Comm<T>::measure(int warmup, int numiter, size_t count) {
    if(count == 0) {
      long count_total = 0;
      for(int send = 0; send < numsend; send++)
         count_total += sendcount[send];
      MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_LONG, MPI_SUM, comm_mpi);
      measure_count(warmup, numiter, count_total);
    }
    else
      measure_count(warmup, numiter, count);
}

template <typename T>
void CommBench::Comm<T>::measure_count(int warmup, int numiter, size_t count) {

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
void CommBench::Comm<T>::measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime) {

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

  // MEMORY MANAGEMENT
template <typename T>
void CommBench::allocate(T *&buffer, size_t n) {
#ifdef PORT_CUDA
    cudaMalloc(&buffer, n * sizeof(T));
#elif defined PORT_HIP
    hipMalloc(&buffer, n * sizeof(T));
#elif defined PORT_SYCL
    buffer = sycl::malloc_device<T>(n, CommBench::q);
#else
    allocateHost(buffer, n);
#endif
}

template <typename T>
void CommBench::allocateHost(T *&buffer, size_t n) {
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
void CommBench::free(T *buffer) {
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
void CommBench::freeHost(T *buffer) {
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

template <typename T>
CommBench::Comm<T>::Comm(CommBench::library lib) : lib(lib) {
    if(!initialized_MPI)
      MPI_Comm_dup(MPI_COMM_WORLD, &comm_mpi); // CREATE SEPARATE COMMUNICATOR EXPLICITLY

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    if(!initialized_MPI) {
      initialized_MPI = true;
      if(myid == printid)
        printf("******************** MPI COMMUNICATOR IS CREATED\n");
    }

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
      if(!initialized_NCCL) {
#ifdef CAP_NCCL
        ncclUniqueId id;
        if(myid == 0)
          ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm_mpi);
        ncclCommInitRank(&comm_nccl, numproc, id, myid);
        initialized_NCCL = true;
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
};


PYBIND11_MODULE(pyComm, m) {
    py::enum_<CommBench::library>(m, "library")
        .value("null", CommBench::library::null)
        .value("MPI", CommBench::library::MPI)
        .value("NCCL", CommBench::library::NCCL)
        .value("IPC", CommBench::library::IPC)
        .value("STAGE", CommBench::library::STAGE)
        .value("numlib", CommBench::library::numlib);
    py::class_<CommBench::Comm<int>>(m, "Comm")
        .def(py::init<CommBench::library>())
        .def("mpi_init", &CommBench::mpi_init)
        .def("mpi_fin", &CommBench::mpi_fin)
        .def("add_lazy", &CommBench::Comm<int>::add_lazy)
        // .def("measure", static_cast<void (CommBench::Comm::*)(int, int, double&, double&, double&, double&)>(&CommBench::Comm::measure), "measure the latency")
        .def("measure", static_cast<void (CommBench::Comm<int>::*)(int, int)>(&CommBench::Comm<int>::measure), "measure the latency")
        // .def("measure", static_cast<void (CommBench::Comm::*)(int, int, size_t)(&CommBench::Comm::measure), "measure the latency">)
        // .def("measure_count", static_cast<void (CommBench::Comm::*)(int, int, size_t)>(&CommBench::Comm::measure_count), "measure the latency");
        // .def("add", &CommBench::Comm<int>::add)
        // .def("start", &CommBench::Comm<int>::start)
        // .def("wait", &CommBench::Comm<int>::wait);
}
