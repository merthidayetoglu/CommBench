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

#if defined(PORT_CUDA) || defined(PORT_HIP)
#define CAP_NCCL
#endif

namespace CommBench
{
  enum library {IPC, MPI, NCCL};

  template <typename T>
  class Comm {

    const library lib;
    const MPI_Comm comm_mpi;

    // GPU-Aware MPI
    MPI_Request *sendrequest;
    MPI_Request *recvrequest;
#ifdef PORT_SYCL
    sycl::queue *q = new sycl::queue(sycl::gpu_selector_v);
#endif

    // NCCL
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
    cudaEvent_t *sendevent_ipc;
    cudaEvent_t *recvevent_ipc;
#elif defined PORT_HIP
    hipStream_t *stream_ipc;
    hipEvent_t *sendevent_ipc;
    hipEvent_t *recvevent_ipc;
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

    Comm(const MPI_Comm &comm_mpi, library lib) : comm_mpi(comm_mpi), lib(lib) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      numsend = 0;
      numrecv = 0;

      if(myid == ROOT) {
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
        switch(lib) {
          case MPI       : printf("GPU-Aware MPI\n");  break;
          case NCCL      : printf("NCCL\n");           break;
          case IPC       : printf("IPC\n");            break;
        }
      }
      if(lib == NCCL) {
#ifdef CAP_NCCL
        ncclUniqueId id;
        if(myid == 0)
          ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm_mpi);
        ncclCommInitRank(&comm_nccl, numproc, id, myid);
#ifdef PORT_CUDA
        cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
        hipStreamCreate(&stream_nccl);
#endif
#endif
      }
    }

    ~Comm() {
      if(numsend) {
        delete[] sendbuf;
        delete[] sendproc;
        delete[] sendcount;
        delete[] sendoffset;
	if(lib == MPI)
          delete[] sendrequest;
      }
      if(numrecv) {
        delete[] recvbuf;
        delete[] recvproc;
        delete[] recvcount;
        delete[] recvoffset;
	if(lib == MPI)
          delete[] recvrequest;
      }
#ifdef PORT_SYCL
      delete q;
#endif
    }

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void launch();
    void wait();
    void run() {launch(); wait();}

    void measure(int warmup, int numiter, double &minTime, double &medTime, double &avgTime, double &maxTime);
    void measure(int warmup, int numiter);
    void report();
  };

  template <typename T>
  void Comm<T>::add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);
    if(myid == ROOT) {
      printf("add (%d -> %d) sendoffset %zu recvoffset %zu count %zu (", sendid, recvid, sendoffset, recvoffset, count);
      double data = count * sizeof(T);
      if (data < 1e3)
        printf("%d bytes)\n", (int)data);
      else if (data < 1e6)
        printf("%.4f KB)\n", data / 1e3);
      else if (data < 1e9)
        printf("%.4f MB)\n", data / 1e6);
      else if (data < 1e12)
        printf("%.4f GB)\n", data / 1e9);
      else
        printf("%.4f TB)\n", data / 1e12);
    }
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
      switch(lib) {
        case MPI:
          if(numsend) delete[] sendrequest;
          sendrequest = new MPI_Request[numsend + 1];
          break;
        case NCCL: break;
        case IPC:
          if(numsend) delete[] sendrequest;
          sendrequest = new MPI_Request[numsend + 1];
          // SEND REMOTE EVENT HANDLE
          {
#ifdef PORT_CUDA
	    cudaEvent_t *sendevent_ipc = new cudaEvent_t[numsend + 1];
            if(numsend) {
              memcpy(sendevent_ipc, this->sendevent_ipc, numsend * sizeof(cudaEvent_t));
              delete[] this->sendevent_ipc;
            }
            this->sendevent_ipc = sendevent_ipc;
#elif PORT_HIP
#endif
          }
#ifdef PORT_CUDA
          if(sendid == recvid) {
            cudaEventCreate(sendevent_ipc + numsend);
	  }
          if(sendid != recvid) {
            cudaEventCreate(sendevent_ipc + numsend, cudaEventInterprocess + cudaEventDisableTiming);
            cudaIpcEventHandle_t eventhandle;
            int error = cudaIpcGetEventHandle(&eventhandle, sendevent_ipc[numsend]);
            MPI_Send(&eventhandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, recvid, 0, comm_mpi);
            if(error) {
              printf("IpcGetEventHandle error %d\n", error);
              return;
            }
          }
#elif PORT_HIP
#endif
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
          } else {
            bool duplicate;
            MPI_Recv(&duplicate, 1, MPI_C_BOOL, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
            if(duplicate) {
              int count_temp;
              MPI_Recv(&count_temp, 1, MPI_INT, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
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
              MPI_Recv(recvbuf_ipc + numsend, sizeof(T*), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
#endif
              if(error) {
                printf("IpcOpenMemHandle error %d\n", error);
                return;
              }
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
        default: break; // do nothing
      } // switch(lib)
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
      // SETUP LIBRARY
      switch(lib) {
        case MPI:
          if(numrecv) delete[] recvrequest;
          recvrequest = new MPI_Request[numrecv + 1];
          break;
        case NCCL: break;
        case IPC:
          if(numrecv) delete[] recvrequest;
          recvrequest = new MPI_Request[numrecv + 1];
          // RECIEVE REMOTE EVENT HANDLE
          {
#ifdef PORT_CUDA
            cudaEvent_t *recvevent_ipc = new cudaEvent_t[numrecv + 1];
            if(numrecv) {
              memcpy(recvevent_ipc, this->recvevent_ipc, numrecv * sizeof(cudaEvent_t));
              delete[] this->recvevent_ipc;
            }
            this->recvevent_ipc = recvevent_ipc;
#elif defined PORT_HIP
#endif
          }
#ifdef PORT_CUDA
          if(sendid == recvid) {
            recvevent_ipc[numrecv] = sendevent_ipc[numsend - 1];
          }
          if(sendid != recvid) {
            cudaIpcEventHandle_t eventhandle;
            MPI_Recv(&eventhandle, sizeof(cudaIpcEventHandle_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
            int error = cudaIpcOpenEventHandle(recvevent_ipc + numrecv, eventhandle);
            if(error) {
              printf("IpcOpenEventHandle error %d\n", error);
              return;
            }
          }
#elif defined PORT_HIP
#endif
          // SEND REMOTE MEMORY HANDLE
          if(sendid != recvid)
          {
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
            MPI_Send(&duplicate, 1, MPI_C_BOOL, sendid, 0, comm_mpi);
            if(duplicate)
              MPI_Send(&count, 1, MPI_INT, sendid, 0, comm_mpi);
            else {
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
              MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, sendid, 0, comm_mpi);
#endif
              if(error) {
                printf("IpcGetMemHandle error %d\n", error);
                return;
              }
            }
            MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, sendid, 0, comm_mpi);
          }
          break;
        default: break; // do nothing
      } // switch(lib)
      numrecv++;
    }
  }

  template <typename T>
  void Comm<T>::measure(int warmup, int numiter) {

    int myid;
    MPI_Comm_rank(comm_mpi, &myid);

    this->report();

    double minTime;
    double medTime;
    double maxTime;
    double avgTime;
    this->measure(warmup, numiter, minTime, medTime, maxTime, avgTime);

    double data = 0;
    for(int send = 0; send < numsend; send++)
       data += sendcount[send] * sizeof(T);
    MPI_Allreduce(MPI_IN_PLACE, &data, 1, MPI_DOUBLE, MPI_SUM, comm_mpi);

    if(myid == ROOT) {
      if (data < 1e3)
        printf("data: %d bytes\n", (int)data);
      else if (data < 1e6)
        printf("data: %.4f KB\n", data / 1e3);
      else if (data < 1e9)
        printf("data: %.4f MB\n", data / 1e6);
      else if (data < 1e12)
        printf("data: %.4f GB\n", data / 1e9);
      else
        printf("data: %.4f TB\n", data / 1e12);
      printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime * 1e9 / data, data / minTime / 1e9);
      printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime * 1e9 / data, data / medTime / 1e9);
      printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime * 1e9 / data, data / maxTime / 1e9);
      printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime * 1e9 / data, data / avgTime / 1e9);
      printf("\n");
    }
  };

  template <typename T>
  void Comm<T>::measure(int warmup, int numiter, double &minTime, double &medTime, double &maxTime, double &avgTime) {

    double times[numiter];
    int myid;
    MPI_Comm_rank(comm_mpi, &myid);

    if(myid == ROOT)
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
      this->launch();
      double start = MPI_Wtime() - time;
      this->wait();
      time = MPI_Wtime() - time;
      MPI_Allreduce(MPI_IN_PLACE, &start, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
      if(iter < 0) {
        if(myid == ROOT)
          printf("startup %.2e warmup: %.2e\n", start, time);
      }
      else {
        times[iter] = time;
      }
    }
    std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

    if(myid == ROOT) {
      printf("%d measurement iterations (sorted):\n", numiter);
      for(int iter = 0; iter < numiter; iter++) {
        printf("time: %.4e", times[iter]);
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
      sendmatrix[sendproc[send]][myid]++;
    for(int recv = 0; recv < numrecv; recv++)
      recvmatrix[myid][recvproc[recv]]++;
    MPI_Allreduce(MPI_IN_PLACE, sendmatrix, numproc * numproc, MPI_INT, MPI_SUM, comm_mpi);
    MPI_Allreduce(MPI_IN_PLACE, recvmatrix, numproc * numproc, MPI_INT, MPI_SUM, comm_mpi);

    if(myid == ROOT) {
      printf("\n");
      switch(lib) {
        case 0: printf("IPC "); break;
        case 1: printf("MPI "); break;
        case 2: printf("NCCL "); break;
      }
      printf("communication matrix\n");
      for(int recv = 0; recv < numproc; recv++) {
        for(int send = 0; send < numproc; send++)
          printf("%d ", sendmatrix[recv][send]);
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

    if(myid == ROOT) {
      printf("send footprint: %e bytes\n", sendTotal);
      printf("recv footprint: %e bytes\n", recvTotal);
      printf("\n");
    }
  }

  template <typename T>
  void Comm<T>::launch() {
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
        break;
#endif
      case IPC:
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_HIP
          hipMemcpyAsync(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
#elif defined PORT_SYCL
	  q->memcpy(recvbuf_ipc[send] + recvoffset_ipc[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T)).wait();
#endif
        }
        break;
    }
  } // Comm::launch

  template <typename T>
  void Comm<T>::wait() { 
    switch(lib) {
      case MPI:
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
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
          bool test = true;
          MPI_Isend(&test, 1, MPI_C_BOOL, sendproc[send], 0, comm_mpi, sendrequest + send);
        }
	for(int recv = 0; recv < numrecv; recv++) {
          bool test = false;
          MPI_Irecv(&test, 1, MPI_C_BOOL, recvproc[recv], 0, comm_mpi, recvrequest + recv);
	}
        MPI_Waitall(numsend, sendrequest, MPI_STATUSES_IGNORE);
        MPI_Waitall(numrecv, recvrequest, MPI_STATUSES_IGNORE);
        break;
    }
  } // Comm::wait()
} // namespace CommBench
