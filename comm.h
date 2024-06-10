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
    // REMOTE BUFFER
    std::vector<T*> remotebuf;
    std::vector<size_t> remoteoffset;

    // SYNCRONIZATION
    std::vector<int> ack_sender;
    std::vector<int> ack_recver;
    void block_sender();
    void block_recver();


    // MEMORY TODO: implement memory management
    // std::vector<T*> buffer_list;
    // std::vector<size_t> buffer_count;

    // MPI
#ifdef USE_MPI
    std::vector<MPI_Request> sendrequest;
    std::vector<MPI_Request> recvrequest;
#endif

    // NCCL
#if defined CAP_NCCL && defined PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined CAP_NCCL && defined PORT_HIP
    hipStream_t stream_nccl;
#elif defined CAP_ONECCL
    ccl::stream *stream_ccl;
#endif

    // IPC
#ifdef PORT_CUDA
    std::vector<cudaStream_t> stream_ipc;
#elif defined PORT_HIP
    std::vector<hipStream_t> stream_ipc;
#elif defined PORT_ONEAPI
    std::vector<sycl::queue> q_ipc;
#endif
    // IPC ZE
#ifdef IPC_ze
    std::vector<ze_command_list_handle_t> command_list;
    std::vector<ze_command_queue_handle_t> command_queue;
    bool command_list_closed = false;
    int ordinal2index = 0;
#endif

    // GASNET
#ifdef CAP_GASNET
    std::vector<gex_Event_t> gex_event;
    std::vector<int> remote_sendind;
    std::vector<int> remote_recvind;
    bool send_ready() {
      for (int i = 0; i < numsend; i++)
        if (!ack_sender[i])
          return false;
      return true;
    };
    bool recv_ready() {
      for (int i = 0; i < numrecv; i++)
        if (!ack_recver[i])
          return false;
      return true;
    };
    gex_AM_Index_t am_notify_sender_index = GEX_AM_INDEX_BASE + 2;
    gex_AM_Index_t am_notify_recver_index = GEX_AM_INDEX_BASE + 3;
    static void am_notify_sender(gex_Token_t token, gex_AM_Arg_t send, gex_AM_Arg_t bench) { ((Comm<T>*)benchlist[bench])->ack_sender[send] = 1; };
    static void am_notify_recver(gex_Token_t token, gex_AM_Arg_t recv, gex_AM_Arg_t bench) { ((Comm<T>*)benchlist[bench])->ack_recver[recv] = 1; };
#endif

    Comm(library lib);
    void free();
    void init();

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void add(T *sendbuf, T *recvbuf, size_t count, int sendid, int recvid);
    void add_lazy(size_t count, int sendid, int recvid);
    void pyadd(pyalloc<T> sendbuf, size_t sendoffset, pyalloc<T> recvbuf, size_t recvoffset, size_t count, int sendid, int recvid);
    void start();
    void wait();

    void measure(int warmup, int numiter);
    void measure(int warmup, int numiter, size_t data);
    std::vector<size_t> getMatrix();
    void report();

    void allocate(T *&buffer, size_t n);
    void allocate(T *&buffer, size_t n, int i);
  };

  template <typename T>
  Comm<T>::Comm(library lib) : lib(lib) {

    benchid = numbench;
    numbench++;
    benchlist.push_back(this);

    numsend = 0;
    numrecv = 0;

    if(myid == printid) {
      printf("printid: %d Create Bench %d with %d processors\n", printid, benchid, numproc);
      printf("  Port: ");
#ifdef PORT_CUDA
      printf("CUDA, ");
#elif defined PORT_HIP
      printf("HIP, ");
#elif defined PORT_ONEAPI
      printf("ONEAPI, ");
#else
      printf("CPU, ");
#endif
#ifdef CAP_NCCL
      printf("NCCL, ");
#elif CAP_ONECCL
      printf("ONECCL, ");
#endif
#ifdef IPC_kernel
      printf("IPC will call a kernel, \n");
#endif
      printf("Library: ");
      print_lib(lib);
      printf("\n");
    }
    if(lib == NCCL) {
#ifdef CAP_NCCL
      static bool init_nccl_comm = false;
      if(!init_nccl_comm) {
        init_nccl_comm = true;
        ncclUniqueId id;
        if(myid == 0)
          ncclGetUniqueId(&id);
        broadcast(&id);
        ncclCommInitRank(&comm_nccl, numproc, id, myid);
        if(myid == printid)
          printf("******************** NCCL COMMUNICATOR IS CREATED\n");
      }
#ifdef PORT_CUDA
      cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
      hipStreamCreate(&stream_nccl);
#endif
#elif defined CAP_ONECCL
      static bool init_ccl_comm = false;
      if(!init_ccl_comm) {
        init_ccl_comm = true;
        /* initialize ccl */
        ccl::init();
        /* create kvs */
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type main_addr;
        if (myid == 0) {
          kvs = ccl::create_main_kvs();
          main_addr = kvs->get_address();
          broadcast(&main_addr);
        }
        else {
          broadcast(&main_addr);
          kvs = ccl::create_kvs(main_addr);
        }
        /* create communicator */
        auto dev = ccl::create_device(CommBench::q.get_device());
        auto ctx = ccl::create_context(CommBench::q.get_context());
        comm_ccl = new ccl::communicator(ccl::create_communicator(numproc, myid, dev, ctx, kvs));
        if(myid == printid)
          printf("******************** ONECCL COMMUNICATOR IS CREATED\n");
        stream_ccl = new ccl::stream(ccl::create_stream(CommBench::q));
      }
#endif
    }
#ifdef CAP_GASNET
    static bool init_gasnet_ep = false;
    if(!init_gasnet_ep) {
      init_gasnet_ep = true;
      // create device endpoint
      gex_EP_Create(&myep, myclient, GEX_EP_CAPABILITY_RMA, 0);
    }
#endif
    if(lib == IPC || lib == IPC_get) {
#ifdef IPC_ze
      ze_context_handle_t hContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
      ze_device_handle_t hDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

      ze_device_properties_t device_properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
      zeDeviceGetProperties(hDevice, &device_properties);
      uint32_t numQueueGroups = 0;
      zeDeviceGetCommandQueueGroupProperties(hDevice, &numQueueGroups, nullptr);
      std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
      for (ze_command_queue_group_properties_t &prop : queueProperties)
        prop = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr};
      zeDeviceGetCommandQueueGroupProperties(hDevice, &numQueueGroups, queueProperties.data());
      int n_commands_lists = 0;
      if(myid == printid)
        printf("device descovery:\n");
      for (uint32_t i = 0; i < numQueueGroups; i++) {
        bool isCompute = false;
        bool isCopy = false;
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
          isCompute = true;
        if ((queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY))
          isCopy = true;
        if(myid == printid)
          printf("group %d isCompute %d isCopy %d\n", i, isCompute, isCopy);
        for (uint32_t j = 0; j < queueProperties[i].numQueues; j++) {
          if(myid == printid)
            printf("  queue: %d\n", j);
          n_commands_lists++;
        }
      }
      {
        int q = 0;
        for(int ord = 0; ord < numQueueGroups; ord++) {
          for(int ind = 0; ind < queueProperties[ord].numQueues; ind++) {
            ze_command_queue_desc_t cmdQueueDesc = {};
            cmdQueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            cmdQueueDesc.ordinal = ord;
            cmdQueueDesc.index = ind;

	    ze_command_list_desc_t command_list_description{};
            command_list_description.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
            command_list_description.commandQueueGroupOrdinal = ord;
            command_list_description.pNext = nullptr;

	    command_queue.push_back(ze_command_queue_handle_t());
            command_list.push_back(ze_command_list_handle_t());
	    zeCommandQueueCreate(hContext, hDevice, &cmdQueueDesc, &command_queue[q]);
	    zeCommandListCreate(hContext, hDevice, &command_list_description, &command_list[q]); 
            q++;
          }
        }
        if(myid == printid)
          printf("number of command queues: %ld\n", command_queue.size());
      }
#endif
    }
  }

  template <typename T>
  void Comm<T>::init() {
#ifdef CAP_GASNET
    gex_AM_Entry_t handlers[] = {
        {am_notify_sender_index, (gex_AM_Fn_t)am_notify_sender, GEX_FLAG_AM_REQUEST | GEX_FLAG_AM_SHORT, 0},
        {am_notify_recver_index, (gex_AM_Fn_t)am_notify_recver, GEX_FLAG_AM_REQUEST | GEX_FLAG_AM_SHORT, 0}
        // Add more handlers if needed
    };
    gex_EP_RegisterHandlers(ep_primordial, handlers, sizeof(handlers) / sizeof(gex_AM_Entry_t));
    barrier();
#endif
  }

  /*template <typename T>
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
      // MPI_Send(&count, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      send(&count, printid);
      if(count) {
        CommBench::allocate(buffer, count);
        buffer_list.push_back(buffer);
        buffer_count.push_back(count);
        // MPI_Send(&buffer, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        send(&buffer, printid);
      }
    }
    if(myid == printid) {
      // MPI_Recv(&count, sizeof(size_t), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
      recv(&count, printid);
      if(count) {
        T *ptr = nullptr;
        // MPI_Recv(&ptr, sizeof(T*), MPI_BYTE, i, 0, comm_mpi, MPI_STATUS_IGNORE);
        recv(&ptr, i);
        printf("Bench %d proc %d allocate %p count %ld (", benchid, i, ptr, count);
        print_data(count * sizeof(T));
        printf(")\n");
      }
    }
  }*/

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
    // OMIT ZERO MESSAGE SIZE
    if(count == 0) {
      if(myid == printid)
        printf("Bench %d communication (%d->%d) count = 0 (skipped)\n", benchid, sendid, recvid);
      return;
    }
    // ADJUST MESSAGE SIZE
    {
// #define COMMBENCH_MESSAGE 16777216 // 16 MB message size if desired
#ifdef COMMBENCH_MESSAGE
      size_t max = COMMBENCH_MESSAGE / sizeof(T);
#else
      size_t max = (lib == MPI ? 2e9 / sizeof(T) : count);
#endif
      while(count > max) {
        add(sendbuf, sendoffset, recvbuf, recvoffset, max, sendid, recvid);
        count = count - max;
        sendoffset += max;
        recvoffset += max;
      }
    }

    // REPORT
    if(printid > -1) {
      barrier(); // THIS IS NECESSARY FOR AURORA
      T* sendbuf_temp;
      T* recvbuf_temp;
      size_t sendoffset_temp;
      size_t recvoffset_temp;
      pair(&sendbuf, &sendbuf_temp, sendid, printid);
      pair(&recvbuf, &recvbuf_temp, recvid, printid);
      pair(&sendoffset, &sendoffset_temp, sendid, printid);
      pair(&recvoffset, &recvoffset_temp, recvid, printid);
      if(myid == printid) {
        printf("Bench %d comm %d (%d->%d) sendbuf %p sendoffset %zu recvbuf %p recvoffset %zu count %zu (", benchid, numcomm, sendid, recvid, sendbuf_temp, sendoffset_temp, recvbuf_temp, recvoffset_temp, count);
        print_data(count * sizeof(T));
        printf(") ");
        print_lib(lib);
        printf("\n");
      }
    }
    numcomm++;

#ifdef IPC_ze
    // queue selection for copy engines
    int queue = -1; // invalid queue
    if((lib == IPC) || (lib == IPC_get)) {
      if(sendid / 2 == recvid / 2) {
        // in the same device
        if(sendid == recvid)
          queue = 0; // self
        else
          queue = 1; // across tiles in the same device
      }
      else {
        // tiles across devices
        queue = 2 + (ordinal2index % 7); // roundrobin: 2, 3, 4, 5, 6, 7, 8, 2, 3, ...
        if(((lib == IPC) && (myid == sendid)) || ((lib == IPC_get) && (myid == recvid)))
          ordinal2index++;
      }
      // REPORT QUEUE
      if(printid > -1) {
        int queue_temp;
        if(lib == IPC) {
          // PUT (SENDER INITIALIZES)
          pair(&queue, queue_temp, sendid, printid);
          if(myid == printid)
            printf("selected put queue: %d\n", queue_temp);
        }
        if(lib == IPC_get) {
          // GET (RECVER INITIALIZES)
          pair(&queue, queue_temp, recvid, printid);
          if(myid == printid)
            printf("selected get queue: %d\n", queue_temp);
        }
      }
    }
#endif

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
#ifdef USE_MPI
        case MPI:
          sendrequest.push_back(MPI_Request());
          break;
#endif
        case NCCL:
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
#elif defined PORT_ONEAPI
          q_ipc.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
          // RECIEVE REMOTE MEMORY HANDLE
          if(sendid != recvid) {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            recv(&memhandle, recvid);
            error = cudaIpcOpenMemHandle((void**)&remotebuf[numsend], memhandle, cudaIpcMemLazyEnablePeerAccess);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            recv(&memhandle, recvid);
            error = hipIpcOpenMemHandle((void**)&remotebuf[numsend], memhandle, hipIpcMemLazyEnablePeerAccess);
#elif defined PORT_ONEAPI
            ze_ipc_mem_handle_t memhandle;
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
              recv(&what_intel_should_have_done, recvid);
              int pidfd = syscall(SYS_pidfd_open,what_intel_should_have_done.pid,0);
              //      int myfd  = syscall(SYS_pidfd_getfd,pidfd,what_intel_should_have_done.fd,0);
              int myfd  = syscall(438,pidfd,what_intel_should_have_done.fd,0);
	      memcpy((void *)&memhandle,(void *)&myfd,sizeof(int));
            }
            // recv(&memhandle, recvid);
	    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
	    error = zeMemOpenIpcHandle(zeContext, zeDevice, memhandle, 0, (void**)&remotebuf[numsend]);
#endif
            if(error)
              printf("IpcOpenMemHandle error %d\n", error);
            recv(&remoteoffset[numsend], recvid);
          }
#ifdef IPC_ze
          zeCommandListAppendMemoryCopy(command_list[queue], remotebuf[numsend] + remoteoffset[numsend], this->sendbuf[numsend] + this->sendoffset[numsend], this->sendcount[numsend], nullptr, 0, nullptr);
#endif
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
            send(&memhandle, recvid);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            error = hipIpcGetMemHandle(&memhandle, sendbuf);
            send(&memhandle, recvid);
#elif defined PORT_ONEAPI
            ze_ipc_mem_handle_t memhandle;
            auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
            error = zeMemGetIpcHandle(zeContext, sendbuf, &memhandle);
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
              memcpy((void *)&what_intel_should_have_done.fd,(void *)&memhandle,sizeof(int));
              what_intel_should_have_done.pid = getpid();
              send(&what_intel_should_have_done, recvid);
            }
            // send(&memhandle, sendid);
#endif
            if(error)
              printf("IpcGetMemHandle error %d\n", error);
            send(&sendoffset, recvid);
          }
          break;
#ifdef CAP_GASNET
        case GEX:
          gex_event.push_back(gex_Event_t());
          ack_sender.push_back(int(0));
          remotebuf.push_back(recvbuf);
          remoteoffset.push_back(recvoffset);
          remote_recvind.push_back(numrecv);
          if(sendid != recvid) {
            recv(&remotebuf[numsend], recvid);
            recv(&remoteoffset[numsend], recvid);
            recv(&remote_recvind[numsend], recvid);
            send(&numsend, recvid);
          }
          break;
        case GEX_get:
          ack_sender.push_back(int(0));
          remote_recvind.push_back(numrecv);
          if(sendid != recvid) {
            send(&sendbuf, recvid);
            send(&sendoffset, recvid);
            send(&numsend, recvid);
            recv(&remote_recvind[numsend], recvid);
          }
          break;
#endif
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
#ifdef USE_MPI
        case MPI:
          recvrequest.push_back(MPI_Request());
          break;
#endif
        case NCCL:
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
	    send(&memhandle, sendid);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            error = hipIpcGetMemHandle(&memhandle, recvbuf);
	    send(&memhandle, sendid);
#elif defined PORT_ONEAPI
            ze_ipc_mem_handle_t memhandle;
	    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	    error = zeMemGetIpcHandle(zeContext, recvbuf, &memhandle);
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
	      memcpy((void *)&what_intel_should_have_done.fd,(void *)&memhandle,sizeof(int));
	      what_intel_should_have_done.pid = getpid();
              send(&what_intel_should_have_done, sendid);
            }
            // send(&memhandle, sendid);
#endif
            if(error)
              printf("IpcGetMemHandle error %d\n", error);
            send(&recvoffset, sendid);
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
#elif defined PORT_ONEAPI
          q_ipc.push_back(sycl::queue(sycl::gpu_selector_v));
#endif
          // RECV REMOTE MEMORY HANDLE
          if(sendid != recvid) {
            int error = -1;
#ifdef PORT_CUDA
            cudaIpcMemHandle_t memhandle;
            recv(&memhandle, sendid);
            error = cudaIpcOpenMemHandle((void**)&remotebuf[numrecv], memhandle, cudaIpcMemLazyEnablePeerAccess);
#elif defined PORT_HIP
            hipIpcMemHandle_t memhandle;
            recv(&memhandle, sendid);
            error = hipIpcOpenMemHandle((void**)&remotebuf[numrecv], memhandle, hipIpcMemLazyEnablePeerAccess);
#elif defined PORT_ONEAPI
            ze_ipc_mem_handle_t memhandle;
            {
              typedef struct { int fd; pid_t pid ; } clone_mem_t;
              clone_mem_t what_intel_should_have_done;
              recv(&what_intel_should_have_done, sendid);
              int pidfd = syscall(SYS_pidfd_open,what_intel_should_have_done.pid,0);
              //      int myfd  = syscall(SYS_pidfd_getfd,pidfd,what_intel_should_have_done.fd,0);
              int myfd  = syscall(438,pidfd,what_intel_should_have_done.fd,0);
              memcpy((void *)&memhandle,(void *)&myfd,sizeof(int));
            }
            // recv(&memhandle, sendid);
            auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
            auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
            error = zeMemOpenIpcHandle(zeContext, zeDevice, memhandle, 0, (void**)&remotebuf[numrecv]);
#endif
            if(error)
              printf("IpcOpenMemHandle error %d\n", error);
            recv(&remoteoffset[numrecv], sendid);
          }
#ifdef IPC_ze
          zeCommandListAppendMemoryCopy(command_list[queue], this->recvbuf[numrecv] + this->recvoffset[numrecv], remotebuf[numrecv] + remoteoffset[numrecv], this->recvcount[numrecv], nullptr, 0, nullptr);
#endif
          break;
#ifdef CAP_GASNET
        case GEX:
          ack_recver.push_back(int(0));
          remote_sendind.push_back(numsend - 1);
          if(sendid != recvid) {
            send(&recvbuf, sendid);
	    send(&recvoffset, sendid);
            send(&numrecv, sendid);
            recv(&remote_sendind[numrecv], sendid);
          }
          break;
        case GEX_get:
          gex_event.push_back(gex_Event_t());
          ack_recver.push_back(int(0));
          remotebuf.push_back(sendbuf);
          remoteoffset.push_back(sendoffset);
          remote_sendind.push_back(numsend - 1);
          if(sendid != recvid) {
            recv(&remotebuf[numrecv], sendid);
            recv(&remoteoffset[numrecv], sendid);
            recv(&remote_sendind[numrecv], sendid);
            send(&numrecv, sendid);
          }
          break;
#endif
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
    allreduce_sum(&count_total);
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

    std::vector<size_t> matrix = getMatrix();

    if(myid == printid) {
      printf("\nCommBench %d: ", benchid);
      print_lib(lib);
      printf(" communication matrix (reciever x sender) nnz: %d\n", numcomm);
      for(int recver = 0; recver < numproc; recver++) {
        for(int sender = 0; sender < numproc; sender++) {
          size_t count = matrix[recver * numproc + sender];
          if(count)
            printf("%ld ", count);
            // printf("1 ");
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

    allreduce_sum(&sendTotal);
    allreduce_sum(&recvTotal);

    /*int numbuf = buffer_list.size();
    allreduce_sum(&numbuf);
    if(numbuf) {
      int total_buff = buffer_list.size();
      std::vector<int> total_buffs(numproc);
      MPI_Allgather(&total_buff, 1, MPI_INT, total_buffs.data(), 1, MPI_INT, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &total_buff, 1, MPI_INT, MPI_SUM, comm_mpi);
      long total_count = 0;
      for(size_t count : buffer_count)
        total_count += count;
      std::vector<long> total_counts(numproc);
      MPI_Allgather(&total_count, sizeof(long), MPI_BYTE, total_counts.data(), sizeof(long), MPI_BYTE, comm_mpi);
      MPI_Allreduce(MPI_IN_PLACE, &total_count, 1, MPI_LONG, MPI_SUM, comm_mpi);
      if(myid == printid) {
        for(int p = 0; p < numproc; p++) {
          printf("proc %d: %d pieces count %ld ", p, total_buffs[p], total_counts[p]);
          print_data(total_counts[p] * sizeof(T));
          printf("\n");
        }
        printf("total pieces: %d count %ld ", total_buff, total_count);
        print_data(total_count * sizeof(T));
        printf("\n");
      }
    }*/

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
  std::vector<size_t> Comm<T>::getMatrix() {

    std::vector<size_t> sendcount_temp(numproc, 0);
    std::vector<size_t> recvcount_temp(numproc, 0);
    for (int send = 0; send < numsend; send++)
      sendcount_temp[sendproc[send]]++;// += sendcount[send];
    for (int recv = 0; recv < numrecv; recv++)
      recvcount_temp[recvproc[recv]]++;// += recvcount[recv];
    std::vector<size_t> sendmatrix(numproc * numproc);
    std::vector<size_t> recvmatrix(numproc * numproc);
    for(int i = 0; i < numproc; i++) {
      allgather(&sendcount_temp[i], &sendmatrix[i * numproc]);
      allgather(&recvcount_temp[i], &recvmatrix[i * numproc]);
    }
    std::vector<size_t> matrix;
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
    return matrix;
  }

#ifdef IPC_kernel
  template <typename T>
  __global__ void copy_kernel(T *output, T *input, size_t count) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < count)
      output[tid] = input[tid];
  }
#endif

  template <typename T>
  void Comm<T>::block_sender() {
#ifdef USE_MPI
    for(int recv = 0; recv < numrecv; recv++)
      MPI_Send(&ack_recver[recv], 1, MPI_INT, recvproc[recv], 0, comm_mpi);
    for(int send = 0; send < numsend; send++)
      MPI_Recv(&ack_sender[send], 1, MPI_INT, sendproc[send], 0, comm_mpi, MPI_STATUS_IGNORE);
#endif
#ifdef USE_GASNET
    for(int recv = 0; recv < numrecv; recv++)
      gex_AM_RequestShort2(myteam, recvproc[recv], am_notify_sender_index, 0, remote_sendind[recv], benchid);
    GASNET_BLOCKUNTIL(send_ready());
    memset(ack_sender.data(), 0, numsend * sizeof(int));
#endif
  }
  template <typename T>
  void Comm<T>::block_recver() {
#ifdef USE_MPI
    for(int send = 0; send < numsend; send++)
      MPI_Send(&ack_sender[send], 1, MPI_INT, sendproc[send], 0, comm_mpi);
    for(int recv = 0; recv < numrecv; recv++)
      MPI_Recv(&ack_recver[recv], 1, MPI_INT, recvproc[recv], 0, comm_mpi, MPI_STATUS_IGNORE);
#endif
#ifdef USE_GASNET
    for(int send = 0; send < numsend; send++)
      gex_AM_RequestShort2(myteam, sendproc[send], am_notify_recver_index, 0, remote_recvind[send], benchid);
    GASNET_BLOCKUNTIL(recv_ready());
    memset(ack_recver.data(), 0, numrecv * sizeof(int));
#endif
  }

  template <typename T>
  void Comm<T>::start() {
    switch(lib) {
#ifdef USE_MPI
      case MPI:
        for (int send = 0; send < numsend; send++)
          MPI_Isend(sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), MPI_BYTE, sendproc[send], 0, comm_mpi, &sendrequest[send]);
        for (int recv = 0; recv < numrecv; recv++)
          MPI_Irecv(recvbuf[recv] + recvoffset[recv], recvcount[recv] * sizeof(T), MPI_BYTE, recvproc[recv], 0, comm_mpi, &recvrequest[recv]);
        break;
#endif
      case NCCL:
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
        block_sender();
        for(int send = 0; send < numsend; send++) {
#ifdef IPC_kernel
  #if defined(PORT_CUDA) || defined(PORT_HIP)
          copy_kernel<T><<<(sendcount[send] + 255) / 256, 256, 0, stream_ipc[send]>>>(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send]);
  #elif defined PORT_ONEAPI && !defined IPC_ze
          // q_ipc[send].memcpy(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
  #endif
#else
  #ifdef PORT_CUDA
          cudaMemcpyAsync(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[send]);
  #elif defined PORT_HIP
          hipMemcpyAsync(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[send]);
  #elif defined PORT_ONEAPI && !defined IPC_ze
          q_ipc[send].memcpy(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
  #endif
#endif
        }
#ifdef IPC_ze
        if(!command_list_closed) {
          for(int i = 0; i < command_queue.size(); i++)
            zeCommandListClose(command_list[i]);
          command_list_closed = true;
        }
        for(int i = 0; i < command_queue.size(); i++)
          zeCommandQueueExecuteCommandLists(command_queue[i], 1, &command_list[i], nullptr);
#endif
        break;
      case IPC_get:
        block_recver();
        for(int recv = 0; recv < numrecv; recv++) {
#ifdef IPC_kernel
  #if defined(PORT_CUDA) || defined(PORT_HIP)
          copy_kernel<T><<<(recvcount[recv] + 255) / 256, 256, 0, stream_ipc[recv]>>>(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv]);
  #elif defined PORT_ONEAPI && !defined IPC_ze
          // q_ipc[send].memcpy(remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T));
  #endif
#else
  #ifdef PORT_CUDA
          cudaMemcpyAsync(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T), cudaMemcpyDeviceToDevice, stream_ipc[recv]);
  #elif defined PORT_HIP
          hipMemcpyAsync(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T), hipMemcpyDeviceToDevice, stream_ipc[recv]);
  #elif defined PORT_ONEAPI && !defined IPC_ze
          q_ipc[recv].memcpy(recvbuf[recv] + recvoffset[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T));
  #endif
#endif
        }
#ifdef IPC_ze
        if(!command_list_closed) {
          for(int i = 0; i < command_queue.size(); i++)
            zeCommandListClose(command_list[i]);
          command_list_closed = true;
        }
        for(int i = 0; i < command_queue.size(); i++)
          zeCommandQueueExecuteCommandLists(command_queue[i], 1, &command_list[i], nullptr);
#endif
        break;
#ifdef CAP_GASNET
      case GEX:
        block_sender();
        for (int send = 0; send < numsend; send++)
          gex_event[send] = gex_RMA_PutNB(gex_TM_Pair(myep, 1), sendproc[send], remotebuf[send] + remoteoffset[send], sendbuf[send] + sendoffset[send], sendcount[send] * sizeof(T), GEX_EVENT_NOW, 0);
        break;
      case GEX_get:
        block_recver();
        for (int recv = 0; recv < numrecv; recv++)
          gex_event[recv] = gex_RMA_GetNB(gex_TM_Pair(myep, 1), recvbuf[recv] + recvoffset[recv], recvproc[recv], remotebuf[recv] + remoteoffset[recv], recvcount[recv] * sizeof(T), 0);
        break;
#endif
      default:
        print_lib(lib);
        printf(" option is not implemented!\n");
        break;
    }
  }

  template <typename T>
  void Comm<T>::wait() {
    switch(lib) {
#ifdef USE_MPI
      case MPI:
        MPI_Waitall(numsend, sendrequest.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(numrecv, recvrequest.data(), MPI_STATUSES_IGNORE);
        break;
#endif
      case NCCL:
#if defined CAP_NCCL && defined PORT_CUDA
        cudaStreamSynchronize(stream_nccl);
#elif defined CAP_NCCL && defined PORT_HIP
        hipStreamSynchronize(stream_nccl);
#elif defined CAP_ONECCL
        q.wait();
#endif
        break;
      case IPC:
#ifdef IPC_ze
        for(int i = 0; i < command_queue.size(); i++)
          zeCommandQueueSynchronize(command_queue[i], UINT64_MAX);
#endif
        for(int send = 0; send < numsend; send++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[send]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_ipc[send]);
#elif defined PORT_ONEAPI && !defined IPC_ze
	  q_ipc[send].wait();
#endif
        }
        block_recver();
        break;
      case IPC_get:
#ifdef IPC_ze
        for(int i = 0; i < command_queue.size(); i++)
          zeCommandQueueSynchronize(command_queue[i], UINT64_MAX);
#endif
        for(int recv = 0; recv < numrecv; recv++) {
#ifdef PORT_CUDA
          cudaStreamSynchronize(stream_ipc[recv]);
#elif defined PORT_HIP
          hipStreamSynchronize(stream_ipc[recv]);
#elif defined PORT_ONEAPI && !defined IPC_ze
          q_ipc[recv].wait();
#endif
        }
        block_sender();
        break;
#ifdef CAP_GASNET
      case GEX:
        for (int send = 0; send < numsend; send++)
          gex_Event_Wait(gex_event[send]);
        block_recver();
        break;
      case GEX_get:
        for (int recv = 0; recv < numrecv; recv++)
          gex_Event_Wait(gex_event[recv]);
        block_sender();
        break;
#endif
      default:
        print_lib(lib);
        printf(" option is not implemented!\n");
        break;
    }
  }
