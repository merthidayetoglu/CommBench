
namespace CommBench
{
  enum capability {MPI, MPI_staged, NCCL, IPC, MEMCPY};

  template <typename T>
  class Comm {

    const capability cap;
    const MPI_Comm comm;

    // GPU-Aware MPI
    MPI_Request *sendrequest;
    MPI_Request *recvrequest;

    // CPU-Staged MPI
    T sendbuf_h;
    T recvbuf_h;
    bool *sendcomplete;
    bool *recvcomplete;
#ifdef PORT_CUDA
    cudaStream_t *sendstream;
    cudaStream_t *recvstream;
#elif defined PORT_HIP
    hipStream_t *sendstream;
    hipStream_t *recvstream;
#endif

#ifdef CAP_NCCL
    ncclComm_t comm_nccl;
#endif
#ifdef PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined PORT_HIP
    hipStream_t stream_nccl;
#endif

    // IPC
    T **recvbuf_ipc;
    size_t *recvoffset_ipc;
#ifdef PORT_CUDA
    cudaStream_t *stream_ipc;
#elif defined PORT_HIP
    hipStream_t *stream_ipc;
#endif

    // memcpy
#ifdef PORT_CUDA
    cudaStream_t *stream_memcpy;
#elif defined PORT_HIP
    hipStream_t *stream_memcpy;
#endif

    T **sendbuf;
    T **recvbuf;
    int numsend;
    int numrecv;
    int *sendproc;
    int *recvproc;
    size_t *sendcount;
    size_t *recvcount;
    size_t *sendoffset;
    size_t *recvoffset;

    public:

    Comm(const MPI_Comm &comm, capability cap) : comm(comm), cap(cap) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numproc);

      numsend = 0;
      numrecv = 0;

      if(myid == ROOT) {
        printf("Create CommBench::Comm with %d processors\n", numproc);
        printf(" Port: ");
#ifdef PORT_CUDA
        printf("CUDA ");
#elif defined PORT_HIP
        printf("HIP ");
#else
        printf("CPU ");
#endif
        printf("Capability: ");
        switch(cap) {
          case MEMCPY    : printf("memcpy\n");         break;
          case MPI       : printf("GPU-Aware MPI\n");  break;
          case MPI_staged: printf("CPU-Staged MPI\n"); break;
          case NCCL      : printf("NCCL\n");           break;
          case IPC       : printf("IPC\n");            break;
        }
      }
      if(cap == NCCL) {
#ifdef CAP_NCCL
        ncclUniqueId id;
        if(myid == 0)
          ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
        ncclCommInitRank(&comm_nccl, numproc, id, myid);
#ifdef PORT_CUDA
        cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
        hipStreamCreate(&stream_nccl);
#endif
#endif
      }
    };

    ~Comm() {
      if(numsend) {
        delete[] sendbuf;
        delete[] sendproc;
        delete[] sendcount;
        delete[] sendoffset;
      }
      if(numrecv) {
        delete[] recvbuf;
        delete[] recvproc;
        delete[] recvcount;
        delete[] recvoffset;
      }

    };

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numproc);
      if(myid == ROOT)
        printf("Add CommBench::Comm sendid %d recvid %d count %zu (%.2e MB)\n", sendid, recvid, count, count * sizeof(T) / 1.e6);
      if(myid == sendid) {
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
        switch(cap) {
          case MEMCPY:
          {
#ifdef PORT_CUDA
            cudaStream_t *stream_memcpy = new cudaStream_t[numsend + 1];
            cudaStreamCreate(stream_memcpy + numsend);
            if(numsend) {
              memcpy(stream_memcpy, this->stream_memcpy, numsend * sizeof(cudaStream_t));
              delete[] this->stream_memcpy;
            }
            this->stream_memcpy = stream_memcpy;
#elif defined PORT_HIP
            hipStream_t *stream_memcpy = new hipStream_t[numsend + 1];
            hipStreamCreate(stream_memcpy + numsend);
            if(numsend) {
              memcpy(stream_memcpy, this->stream_memcpy, numsend * sizeof(hipStream_t));
              delete[] this->stream_memcpy;
            }
            this->stream_memcpy = stream_memcpy;
#endif
            break;
          }
          case MPI:
            if(numsend) delete[] sendrequest;
            sendrequest = new MPI_Request[numsend + 1];
            break;
          case MPI_staged: break;
          case NCCL: break;
          case IPC:
            assert(sendid != recvid); // SENDER AND RECEIVER HAS TO BE DISTINCT
            {
               T **recvbuf_ipc = new T*[numsend + 1];
               size_t *recvoffset_ipc = new size_t[numsend + 1];
#ifdef PORT_CUDA
               cudaStream_t *stream_ipc = new cudaStream_t[numsend + 1];
               cudaStreamCreate(stream_ipc + numsend);
               if(numsend) {
                 memcpy(recvbuf_ipc, this->recvbuf_ipc, numsend * sizeof(T*));
                 memcpy(recvoffset_ipc, this->recvoffset_ipc, numsend * sizeof(size_t));
                 memcpy(stream_ipc, this->stream_ipc, numsend * sizeof(cudaStream_t));
                 delete[] this->stream_ipc;
                 delete[] this->recvbuf_ipc;
                 delete[] this->recvoffset_ipc;
               }
               this->stream_ipc = stream_ipc;
#elif defined PORT_HIP
               hipStream_t *stream_ipc = new hipStream_t[numsend + 1];
               hipStreamCreate(stream_ipc + numsend);
               if(numsend) {
                 memcpy(recvbuf_ipc, this->recvbuf_ipc, numsend * sizeof(T*));
                 memcpy(recvoffset_ipc, this->recvoffset_ipc, numsend * sizeof(size_t));
                 memcpy(stream_ipc, this->stream_ipc, numsend * sizeof(hipStream_t));
                 delete[] this->stream_ipc;
                 delete[] this->recvbuf_ipc;
                 delete[] this->recvoffset_ipc;
               }
               this->stream_ipc = stream_ipc;
#endif
               this->recvbuf_ipc = recvbuf_ipc;
               this->recvoffset_ipc = recvoffset_ipc;
            }
            bool duplicate;
            MPI_Recv(&duplicate, 1, MPI_C_BOOL, recvid, 0, comm, MPI_STATUS_IGNORE);
            if(duplicate) {
              int count_temp;
              MPI_Recv(&count_temp, 1, MPI_INT, recvid, 0, comm, MPI_STATUS_IGNORE);
              int count = 0;
              for(int send = 0; send < numsend; send++)
                if(sendproc[send] == recvid) {
                  if(count == count_temp) {
                    recvbuf_ipc[numsend] = recvbuf_ipc[send];
                    break;
                  }
                  else
                    count++;
                }
            }
            else {
#ifdef PORT_CUDA
              cudaIpcMemHandle_t memhandle;
              MPI_Recv(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, recvid, 0, comm, MPI_STATUS_IGNORE);
              int error = cudaIpcOpenMemHandle((void**) recvbuf_ipc + numsend, memhandle, cudaIpcMemLazyEnablePeerAccess);
              assert(error == 0); // CHECK RECEIVER POINTER HEAD
#elif defined PORT_HIP
              hipIpcMemHandle_t memhandle;
              MPI_Recv(&memhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, recvid, 0, comm, MPI_STATUS_IGNORE);
              int error = hipIpcOpenMemHandle((void**) recvbuf_ipc + numsend, memhandle, hipIpcMemLazyEnablePeerAccess);
              assert(error == 0); // CHECK RECEIVER POINTER HEAD
#endif
            }
            MPI_Recv(recvoffset_ipc + numsend, sizeof(size_t), MPI_BYTE, recvid, 0, comm, MPI_STATUS_IGNORE);
            break;
        } // switch(cap)
        numsend++;
      }
      if(myid == recvid) {
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
        // SETUP CAPABILITY
        switch(cap) {
          case MEMCPY: break;
          case MPI:
            if(numrecv) delete[] recvrequest;
            recvrequest = new MPI_Request[numrecv + 1];
            break;
          case MPI_staged: break;
          case NCCL: break;
          case IPC:
            bool duplicate = false;
            int count = 0;
            for(int recv = 0; recv < numrecv; recv++)
              if(recvproc[recv] == sendid) {
                if(this->recvbuf[recv] == recvbuf) {
                  duplicate = true;
                  break;
                }
                else
                  count++;
              }
            MPI_Send(&duplicate, 1, MPI_C_BOOL, sendid, 0, comm);
            if(duplicate)
              MPI_Send(&count, 1, MPI_INT, sendid, 0, comm);
            else {
#ifdef PORT_CUDA
              cudaIpcMemHandle_t myhandle;
              cudaIpcGetMemHandle(&myhandle, recvbuf);
              MPI_Send(&myhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, sendid, 0, comm);
#elif defined PORT_HIP
              hipIpcMemHandle_t myhandle;
              hipIpcGetMemHandle(&myhandle, recvbuf);
              MPI_Send(&myhandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, sendid, 0, comm);
#endif
            }
            MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, sendid, 0, comm);
            break;
        }
        numrecv++;
      }
    };

    void init();
    void wait();

    void report();
  };

  template <typename T>
  void Comm<T>::report() {
    int myid;
    int numproc;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &numproc);

    int sendmatrix[numproc][numproc];
    int recvmatrix[numproc][numproc];
    memset(sendmatrix, 0, numproc * numproc * sizeof(int));
    memset(recvmatrix, 0, numproc * numproc * sizeof(int));
    for(int send = 0; send < numsend; send++)
      sendmatrix[sendproc[send]][myid]++;
    for(int recv = 0; recv < numrecv; recv++)
      recvmatrix[myid][recvproc[recv]]++;

    MPI_Allreduce(MPI_IN_PLACE, recvmatrix, numproc * numproc, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, sendmatrix, numproc * numproc, MPI_INT, MPI_SUM, comm);

    if(myid == ROOT) {
      printf("recv matrix\n");
      for(int recv = 0; recv < numproc; recv++) {
        for(int send = 0; send < numproc; send++)
          printf("%d ", recvmatrix[recv][send]);
        printf("\n");
      }
      printf("send matrix\n");
      for(int recv = 0; recv < numproc; recv++) {
        for(int send = 0; send < numproc; send++)
          printf("%d ", sendmatrix[recv][send]);
        printf("\n");
      }
    }
  }

  template <typename T>
  void Comm<T>::init() {
    switch(cap) {
      case MEMCPY:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_memcpy[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(recvbuf[send] + recvoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_memcpy[send]);
#endif
        }
        break;
      case MPI:
        for (int send = 0; send < numsend; send++)
           MPI_Isend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm, sendrequest + send);
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm, recvrequest + recv);
        break;
      case MPI_staged:
        break;
      case NCCL:
#ifdef CAP_NCCL
        ncclGroupStart(); 
        for(int send = 0; send < numsend; send++)
          ncclSend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), ncclInt8, sendproc[send], comm_nccl, stream_nccl);
        for(int recv = 0; recv < numrecv; recv++)
          ncclRecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), ncclInt8, recvproc[recv], comm_nccl, stream_nccl);
        ncclGroupEnd();
        break;
#endif
      case IPC:
        int myid;
        MPI_Comm_rank(comm, &myid);
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
#endif
        }
        break;
    }
  } // Comm::init

  template <typename T>
  void Comm<T>::wait() { 
    switch(cap) {
      case MEMCPY:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_memcpy[send]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_memcpy[send]);
#endif
        }
        break;
      case MPI:
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        break;
      case MPI_staged:
        break;
      case NCCL:
#ifdef PORT_CUDA
        cudaStreamSynchronize(stream_nccl);
#elif defined PORT_HIP
        hipStreamSynchronize(stream_nccl);
#endif
        break;
      case IPC:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[send]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_ipc[send]);
#endif
        }
        MPI_Barrier(comm);
        break;
    }
  } // Comm::wait()

} // namespace CommBench
