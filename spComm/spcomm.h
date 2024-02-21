
#include "comm.h"
#include "kernels.h"

namespace CommBench {

  template <typename T>
  class SpComm : public Comm<T> {
    public:
    std::vector<void*> arg;
    std::vector<void (*)(void *)> func;
    std::vector<size_t> count;
    std::vector<int> precompid;
    std::vector<int> postcompid;

#ifdef PORT_CUDA
    std::vector<cudaStream_t> stream_comp;
    std::vector<cudaStream_t> stream_self;
#elif defined PORT_HIP
    std::vector<hipStream_t> stream_comp;
    std::vector<hipStream_t> stream_self;
#elif defined PORT_SYCL
    std::vector<sycl::queue> queue_comp;
    std::vector<sycl::queue> queue_self;
#endif

    std::vector<T*> sendbuf_self;
    std::vector<T*> recvbuf_self;
    std::vector<size_t> count_self;

    using Comm<T>::Comm;

    // ADD COMPUTATION
    void add_comp(void func(void *), void *arg, size_t count) {
      this->count.push_back(count);
      this->arg.push_back(arg);
      this->func.push_back(func);
#ifdef PORT_CUDA
      stream_comp.push_back(cudaStream_t());
      cudaStreamCreate(&stream_comp[stream_comp.size() - 1]);
#elif defined PORT_HIP
      stream_comp.push_back(hipStream_t());
      hipStreamCreate(&stream[stream_comp.size() - 1]);
#elif defined PORT_SYCL
      queue_comp.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
    }
    // REGISTER COMPUTATION BEFORE COMMUNICATION
    template <typename S>
    void add_precomp(void func(void*), S arg, size_t count, int i) {
      // REPORT
      S *arg_d;
      if(myid == i) {
        if(count) {
          // COPY ARGUMENT TO GPU
          CommBench::allocate(arg_d, 1);
          CommBench::memcpyH2D(arg_d, &arg, 1);
        }
        // REPORT
        MPI_Send(&func, sizeof(void(*)(void*)), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&arg_d, sizeof(S*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == printid) {
        MPI_Recv(&func, sizeof(void(*)(void*)), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&arg_d, sizeof(S*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        if(count)
          printf("Bench %d proc %d add pre-compute function %p argsize %ld bytes GPU addr %p numthreads %ld\n", Comm<T>::benchid, i, func, sizeof(S), arg_d, count);
      }
      if(!count)
        return;
      // ADD COMPUTATION
      if(myid == i) {
        precompid.push_back(this->count.size());
        add_comp(func, arg_d, count);
      }
    };
    template <typename S>
    void add_precomp(void func(void*), S arg, size_t count) {
      for (int i = 0; i < numproc; i++)
        add_precomp(func, arg, count, i);
    }
    template <typename I>
    void add_precomp_gather(T *sendbuf, T *recvbuf, size_t count, I *index) {
      sparse_t<T, I> sparse(sendbuf, recvbuf, count, nullptr, index);
      add_precomp(sparse_gather<T, I>, sparse, count);
    }
    // REGISTER COMPUTATION AFTER COMMUNICATION
    template <typename S>
    void add_postcomp(void func(void *), S arg, size_t count, int i) {
      S *arg_d;
      if(myid == i) {
        if(count) {
          // SEND ARGUMENT TO GPU
          CommBench::allocate(arg_d, 1);
          CommBench::memcpyH2D(arg_d, &arg, 1);
        }
        // REPORT
        MPI_Send(&func, sizeof(void(*)(void*)), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&arg_d, sizeof(S*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      if(myid == printid) {
        MPI_Recv(&func, sizeof(void(*)(void*)), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&arg_d, sizeof(S*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        if(count)
          printf("Bench %d proc %d add post-compute function %p argsize %ld bytes GPU addr %p numthreads %ld\n", Comm<T>::benchid, i, func, sizeof(S), arg_d, count);
      }
      if(!count)
        return;
      // ADD COMPUTATION
      if(myid == i) {
        postcompid.push_back(this->count.size());
        add_comp(func, arg_d, count);
      }
    };
    template <typename S>
    void add_postcomp(void func(void*), S arg, size_t count) {
      for (int i = 0; i < numproc; i++)
        add_postcomp(func, arg, count, i);
    }
    template <typename I>
    void add_postcomp_scatter(T *sendbuf, T *recvbuf, size_t count, I *index) {
      sparse_t<T, I> sparse(sendbuf, recvbuf, count, nullptr, index);
      add_postcomp(sparse_scatter<T, I>, sparse, count);
    }

    void add(T *sendbuf, size_t sendoffset, size_t sendupper, T *recvbuf, size_t recvoffset, size_t recvupper, int sendid, int recvid) {
      size_t sendcount = sendupper - sendoffset;
      size_t recvcount = recvupper - recvoffset;
      MPI_Bcast(&sendcount, sizeof(size_t), MPI_BYTE, sendid, comm_mpi);
      MPI_Bcast(&recvcount, sizeof(size_t), MPI_BYTE, recvid, comm_mpi);
      if(sendcount != recvcount) {
        if(myid == printid)
          printf("sendid %d sendcount %ld recvid %d recvcount %ld does not match!\n", sendid, sendcount, recvid, recvcount);
      }
      else if(sendcount == 0) {
          if(myid == printid)
            printf("Bench %d communication (%d->%d) count = 0 (skipped)\n", Comm<T>::benchid, sendid, recvid);
      }
      else if(sendid == recvid) {
        if(myid == printid)
          printf("register self communication proc %d count %ld\n", sendid, sendcount);
        if(myid == sendid) {
          sendbuf_self.push_back(sendbuf + sendoffset);
          recvbuf_self.push_back(recvbuf + recvoffset);
          count_self.push_back(sendcount);
#ifdef PORT_CUDA
          stream_self.push_back(cudaStream_t());
          cudaStreamCreate(&stream_self[stream_self.size() - 1]);
#elif defined PORT_HIP
          stream_self.push_back(hipStream_t());
          hipStreamCreate(&stream_self[stream_self.size() - 1]);
#elif defined PORT_SYCL
          queue_self.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
        }
      }
      else
        Comm<T>::add(sendbuf, sendoffset, recvbuf, recvoffset, sendcount, sendid, recvid);
    }

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      add(sendbuf, sendoffset, sendoffset + count, recvbuf, recvoffset, recvoffset + count, sendid, recvid);
    }
    void add(T *sendbuf, T *recvbuf, size_t count, int sendid, int recvid) {
      add(sendbuf, 0, recvbuf, 0, count, sendid, recvid);
    }

    void start() {
      for (int i : precompid) {
#if defined PORT_CUDA || defined PORT_HIP
        const int blocksize = 256;
        func[i]<<<(count[i] + blocksize - 1) / blocksize, blocksize, 0, stream_comp[i]>>>(arg[i]);
#elif defined PORT_SYCL
        ; // start compute
#else
        func[i](arg[i]);
#endif
      }
      for (int i : precompid) {
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_comp[i]);
#elif defined PORT_HIP
        hipStreamSynchronize(stream_comp[i]);
#elif defined PORT_SYCL
        queue_comp[i].wait();
#endif
      }
      Comm<T>::start();
      for(int i = 0; i < count_self.size(); i++) {
#ifdef PORT_CUDA
        cudaMemcpyAsync(recvbuf_self[i], sendbuf_self[i], count_self[i] * sizeof(T), cudaMemcpyDeviceToDevice, stream_self[i]);
#elif PORT_HIP
        hipMemcpyAsync(recvbuf_self[i], sendbuf_self[i], count_self[i] * sizeof(T), hipMemcpyDeviceToDevice, stream_self[i]);
#elif defined PORT_SYCL
	queue_self[i].memcpy(recvbuf_self[i], sendbuf_self[i], count_self[i] * sizeof(T));
#else
	memcpy(recvbuf_self[i], sendbuf_self[i], count_self[i] * sizeof(T));
#endif
      }
    }

    void wait() {
      for(int i = 0; i < count_self.size(); i++) {
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_self[i]);
#elif PORT_HIP
        hipStreamSynchronize(stream_self[i]);
#elif defined PORT_SYCL
        queue_self[i].wait();
#endif
      }
      Comm<T>::wait();
      for (int i : postcompid) {
#if defined PORT_CUDA || defined PORT_HIP
        const int blocksize = 256;
        func[i]<<<(count[i] + blocksize - 1) / blocksize, blocksize, 0, stream_comp[i]>>>(arg[i]);
#elif defined PORT_SYCL
        ;//queue_comp[i].function(arg[i]);
#else
        func[i](arg[i]);
#endif
      }
      for (int i : postcompid) {
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_comp[i]);
#elif defined PORT_HIP
        hipStreamSynchronize(stream_comp[i]);
#elif defined PORT_SYCL
        queue_comp[i].wait();
#endif
      }
    }
    void measure(int warmup, int numiter, size_t count) {
      Comm<T>::report();
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
    void measure(int warmup, int numiter) {
      long count_total = 0;
      for(int send = 0; send < Comm<T>::numsend; send++)
         count_total += Comm<T>::sendcount[send];
      MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_LONG, MPI_SUM, comm_mpi);
      measure(warmup, numiter, count_total);
    };
  };
}
