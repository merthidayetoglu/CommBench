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
            void add_lazy(size_t count, int sendid, int recvid);
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
        .def("add_lazy", &CommBench::add_lazy)
        // .def("measure", static_cast<void (CommBench::Comm::*)(int, int, double&, double&, double&, double&)>(&CommBench::Comm::measure), "measure the latency")
        .def("measure", static_cast<void (CommBench::Comm<int>::*)(int, int)>(&CommBench::Comm<int>::measure), "measure the latency")
        // .def("measure", static_cast<void (CommBench::Comm::*)(int, int, size_t)(&CommBench::Comm::measure), "measure the latency">)
        // .def("measure_count", static_cast<void (CommBench::Comm::*)(int, int, size_t)>(&CommBench::Comm::measure_count), "measure the latency");
        // .def("add", &CommBench::Comm<int>::add)
        // .def("start", &CommBench::Comm<int>::start)
        // .def("wait", &CommBench::Comm<int>::wait);
}
