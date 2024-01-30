
#include "comm.h"


namespace CommBench {

  template <typename T, typename I>
  struct sparse_t {
    // public:
    T *sendbuf;
    T *recvbuf;
    size_t count;
    size_t *offset;
    I *index;
    sparse_t() {};
    sparse_t(T *sendbuf, T *recvbuf, size_t count, size_t *offset, I *index) : sendbuf(sendbuf), recvbuf(recvbuf), count(count), offset(offset), index(index) {};
  };

  template <typename T, typename I>
  sparse_t<T, I>* create_sparse(T *sendbuf, T *recvbuf, size_t count, size_t *offset, I *index, int i) {
    sparse_t<T, I> *sparse_d = nullptr;
    if(myid == i) {
      I *index_d;
      size_t *offset_d;
      if(offset == nullptr) {
        offset_d = nullptr;
        CommBench::allocate(index_d, count);
        CommBench::memcpyH2D(index_d, index, count);
      }
      else {
        CommBench::allocate(offset_d, count + 1);
        CommBench::allocate(index_d, offset[count]);
        CommBench::memcpyH2D(offset_d, offset, count + 1);
        CommBench::memcpyH2D(index_d, index, offset[count]);
      }
      sparse_t<T, I> sparse(sendbuf, recvbuf, count, offset_d, index_d);
      CommBench::allocate(sparse_d, 1);
      CommBench::memcpyH2D(sparse_d, &sparse, 1);
      MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
      MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
      MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
    }
    if(myid == printid) {
      MPI_Recv(&sendbuf, sizeof(T*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
      MPI_Recv(&recvbuf, sizeof(T*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
      MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
      printf("proc %d creates sparse operator: sendbuf %p recvbuf %p count %ld\n", i, sendbuf, recvbuf, count);
    }
    return sparse_d;
  }

#if defined PORT_CUDA || PORT_HIP
  template <typename T, typename I>
  __global__ void sparse_kernel(void *sparse_temp) {
    sparse_t<T,I> &sparse = *((sparse_t<T,I>*)sparse_temp);
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t count = sparse.count;
    if (tid < count) {
      T *sendbuf = sparse.sendbuf;
      T *recvbuf = sparse.recvbuf;
      size_t *offset = sparse.offset;
      I *index = sparse.index;
      if(offset == nullptr)
        recvbuf[tid] = sendbuf[index[tid]];
      else {
        T acc = 0;
        for (size_t i = offset[tid]; i < offset[tid + 1]; i++)
          acc += sendbuf[index[i]];
        recvbuf[tid] = acc;
      }
    }
  }
#elif defined SYCL
#else
  template <typename T, typename I>
  void sparse_kernel(void *sparse_temp) {
    sparse_t<T,I> &sparse = *((sparse_t<T,I>*)sparse_temp);
    T *sendbuf = sparse.sendbuf;
    T *recvbuf = sparse.recvbuf;
    size_t count = sparse.count;
    size_t *offset = sparse.offset;
    I *index = sparse.index;
    if(offset == nullptr) {
      #pragma omp parallel for
      for(size_t i = 0; i < count; i++)
        recvbuf[i] = sendbuf[index[i]];
    }
    else {
      #pragma omp parallel for
      for (size_t i = 0; i < count; i++) {
        T acc = 0;
        for (size_t j = offset[i]; j < offset[i + 1]; j++)
          acc += sendbuf[index[j]];
        recvbuf[i] = acc;
      }
    }
  }
#endif

  template <typename T, typename I>
  class SpComm : public Comm<T> {
    public:
    std::vector<void*> arg;
    std::vector<void (*)(void *)> func;
    std::vector<size_t> count;
    std::vector<int> precompid;
    std::vector<int> postcompid;

#ifdef PORT_CUDA
    std::vector<cudaStream_t> stream;
#elif defined PORT_HIP
    std::vector<hipStream_t> stream;
#elif defined PORT_SYCL
    std::vector<sycl::queue> queue;
#endif

    using Comm<T>::Comm;

    // MEMORY ALLOCATION FUNCTIONS
    T* allocate(size_t n, std::vector<int> vec) {
      T *buffer = nullptr;
      for (int i : vec)
        Comm<T>::allocate(buffer, n, i);
      return buffer;
    };
    T* allocate(size_t n, int i) {
      std::vector<int> v{i};
      return allocate(n, v);
    };
    T* allocate(size_t n) {
      std::vector<int> v;
      for (int i = 0; i < numproc; i++)
        v.push_back(i);
      return allocate(n, v);
    };

    // ADD COMPUTATION
    void add_comp(void func(void *), void *arg, size_t count) {
      this->count.push_back(count);
      this->arg.push_back(arg);
      this->func.push_back(func);
#ifdef PORT_CUDA
      stream.push_back(cudaStream_t());
      cudaStreamCreate(&stream[stream.size() - 1]);
#elif defined PORT_HIP
      stream.push_back(hipStream_t());
      hipStreamCreate(&stream[stream.size() - 1]);
#elif defined PORT_SYCL
      queue.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
    }
    // ADD COMPUTATION BEFORE COMMUNICATION
    void add_precomp(void func(void *), void *arg, size_t count, int i) {
      // REPORT
      if(myid == i) {
        MPI_Send(&func, sizeof(void(*)(void*)), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&arg, sizeof(void*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == printid) {
        MPI_Recv(&func, sizeof(void(*)(void*)), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&arg, sizeof(void*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        if(count)
          printf("Bench %d proc %d add pre-compute function %p arg %p count %ld\n", Comm<T>::benchid, myid, func, arg, count);
      }
      if(count == 0) return;
      if(myid == i) {
        precompid.push_back(this->count.size());
        add_comp(func, arg, count);
      }
    };
    // ADD COMPUTATION AFTER COMMUNICATION
    void add_postcomp(void func(void *), void *arg, size_t count, int i) {
      // REPORT
      if(myid == i) {
        MPI_Send(&func, sizeof(void(*)(void*)), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&arg, sizeof(void*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == printid) {
        MPI_Recv(&func, sizeof(void(*)(void*)), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&arg, sizeof(void*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        if(count)
          printf("Bench %d proc %d add post-compute function %p arg %p count %ld\n", Comm<T>::benchid, myid, func, arg, count);
      }
      if(count == 0) return;
      if(myid == i) {
        postcompid.push_back(this->count.size());
        add_comp(func, arg, count);
      }
    };
    // SPARSE GATHER
    void add_gather(T *sendbuf, T *recvbuf, size_t count, I *index, int i) {
       sparse_t<T, I> *temp = create_sparse(sendbuf, recvbuf, count, nullptr, index, i);
       add_precomp(sparse_kernel<T, I>, temp, count, i);
    }
    // SPARSE SCATTER
    void add_scatter(T *sendbuf, T *recvbuf, size_t count, I *index, int i) {
       sparse_t<T, I> *temp = create_sparse(sendbuf, recvbuf, count, nullptr, index, i);
       add_postcomp(sparse_kernel<T, I>, temp, count, i);
    }

    void start() {
      for (int i : precompid) {
#if defined PORT_CUDA || defined PORT_HIP
        const int blocksize = 256;
        func[i]<<<(count[i] + blocksize - 1) / blocksize, blocksize, 0, stream[i]>>>(arg[i]);
#elif defined PORT_SYCL
        queue.function(arg[i]);
#else
        func[i](arg[i]);
#endif
      }
      for (int i : precompid) {
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream[i]);
#elif defined PORT_HIP
        hipStreamSynchronize(stream[i]);
#elif defined PORT_SYCL
        queue[i].wait();
#else
        ;
#endif
      }
      Comm<T>::start();
    }

    void wait() {
      Comm<T>::wait();
      for (int i : postcompid) {
#if defined PORT_CUDA || defined PORT_HIP
        const int blocksize = 256;
        func[i]<<<(count[i] + blocksize - 1) / blocksize, blocksize, 0, stream[i]>>>(arg[i]);
#elif defined PORT_SYCL
        queue.function(arg[i]);
#else
        func[i](arg[i]);
#endif
      }
      for (int i : postcompid) {
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream[i]);
#elif defined PORT_HIP
        hipStreamSynchronize(stream[i]);
#elif defined PORT_SYCL
        queue[i].wait();
#else
        ;
#endif
      }
    }

    void measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime) {

      double times[numiter];
      double starts[numiter];

      if(myid == printid)
        printf("%d warmup iterations (in order):\n", warmup);
      for (int iter = -warmup; iter < numiter; iter++) {
        for(int send = 0; send < Comm<T>::numsend; send++) {
#if defined PORT_CUDA
          // cudaMemset(sendbuf[send], -1, sendcount[send] * sizeof(T));
#elif defined PORT_HIP
          // hipMemset(sendbuf[send], -1, sendcount[send] * sizeof(T));
#elif defined PORT_SYCL
          // q->memset(sendbuf[send], -1, sendcount[send] * sizeof(T)).wait();
#else
          memset(Comm<T>::sendbuf[send], -1, Comm<T>::sendcount[send] * sizeof(T)); // NECESSARY FOR CPU TO PREVENT CACHING
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
            printf("startup: %.2e warmup: %.2e\n", start * 1e6, time * 1e6);
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
    };

    void measure(int warmup, int numiter, size_t count) {
      Comm<T>::report();
      double minTime;
      double medTime;
      double maxTime;
      double avgTime;
      measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
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

    void measure(int warmup, int numiter) {
      long count_total = 0;
      for(int send = 0; send < Comm<T>::numsend; send++)
         count_total += Comm<T>::sendcount[send];
      MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_LONG, MPI_SUM, comm_mpi);
      measure(warmup, numiter, count_total);
    };

  };
}
