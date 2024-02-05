

  template <typename T, typename I>
  struct sparse_t {
    T *sendbuf;
    T *recvbuf;
    size_t count;
    I *offset;
    I *index;
    void init_sparse(T *sendbuf, T *recvbuf, size_t count, I *offset, I *index, int i) {
      if (CommBench::myid == i) {
        this->sendbuf = sendbuf;
        this->recvbuf = recvbuf;
	this->count = count;
        if(offset == nullptr) {
          this->offset = nullptr;
          CommBench::allocate(this->index, count);
          CommBench::memcpyH2D(this->index, index, count);
        }
        else {
          CommBench::allocate(this->offset, count + 1);
          CommBench::memcpyH2D(this->offset, offset, count + 1);
          CommBench::allocate(this->index, offset[count]);
          CommBench::memcpyH2D(this->index, index, offset[count]);
        }
        // REPORT
        if(CommBench::printid > -1) {
          MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, CommBench::printid, 0, CommBench::comm_mpi);
          MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, CommBench::printid, 0, CommBench::comm_mpi);
          MPI_Send(&count, sizeof(I*), MPI_BYTE, CommBench::printid, 0, CommBench::comm_mpi);
        }
      }
      if(CommBench::myid == CommBench::printid) {
        MPI_Recv(&sendbuf, sizeof(T*), MPI_BYTE, i, 0, CommBench::comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&recvbuf, sizeof(T*), MPI_BYTE, i, 0, CommBench::comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&count, sizeof(I*), MPI_BYTE, i, 0, CommBench::comm_mpi, MPI_STATUS_IGNORE);
        printf("proc %d creates sparse operator: sendbuf %p recvbuf %p count %ld\n", i, sendbuf, recvbuf, count);
      }
    }
    public:
    sparse_t(T *sendbuf, T *recvbuf, size_t count, I *offset, I *index, int i) {
      init_sparse(sendbuf, recvbuf, count, offset, index, i);
    }
    sparse_t(T *sendbuf, T *recvbuf, size_t count, I *offset, I *index) {
      for(int i = 0; i < CommBench::numproc; i++)
        init_sparse(sendbuf, recvbuf, count, offset, index, i);
    }
  };

#if defined PORT_CUDA || PORT_HIP
  template <typename T, typename I>
  __global__ void sparse_gather(void *sparse_temp) {
    sparse_t<T,I> &sparse = *((sparse_t<T,I>*)sparse_temp);
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t count = sparse.count;
    if (tid < count) {
      T *sendbuf = sparse.sendbuf;
      T *recvbuf = sparse.recvbuf;
      I *offset = sparse.offset;
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
  void sparse_gather(void *sparse_temp) {
    sparse_t<T,I> &sparse = *((sparse_t<T,I>*)sparse_temp);
    T *sendbuf = sparse.sendbuf;
    T *recvbuf = sparse.recvbuf;
    size_t count = sparse.count;
    I *offset = sparse.offset;
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


#if defined PORT_CUDA || PORT_HIP
  template <typename T, typename I>
  __global__ void sparse_scatter(void *sparse_temp) {
    sparse_t<T,I> &sparse = *((sparse_t<T,I>*)sparse_temp);
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t count = sparse.count;
    if (tid < count) {
      T *sendbuf = sparse.sendbuf;
      T *recvbuf = sparse.recvbuf;
      I *offset = sparse.offset;
      I *index = sparse.index;
      if(offset == nullptr)
        recvbuf[index[tid]] = sendbuf[tid];
      else
        for (size_t i = offset[tid]; i < offset[tid + 1]; i++)
          recvbuf[index[i]] = sendbuf[tid];
    }
  }
#elif defined SYCL
#else
  template <typename T, typename I>
  void sparse_scatter(void *sparse_temp) {
    sparse_t<T,I> &sparse = *((sparse_t<T,I>*)sparse_temp);
    T *sendbuf = sparse.sendbuf;
    T *recvbuf = sparse.recvbuf;
    size_t count = sparse.count;
    size_t *offset = sparse.offset;
    I *index = sparse.index;
    if(offset == nullptr) {
      #pragma omp parallel for
      for(size_t i = 0; i < count; i++)
        recvbuf[index[i]] = sendbuf[i]
    }
    else {
      #pragma omp parallel for
      for (size_t i = 0; i < count; i++)
        for (size_t j = offset[i]; j < offset[i + 1]; j++)
          recvbuf[index[j]] += sendbuf[j];
    }
  }
#endif

