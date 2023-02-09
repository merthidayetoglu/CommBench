{
  if(myid == ROOT)
    printf("TEST ALLGATHER LATENCY\n");
  Type *sendbuf_d;
  Type *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * sizeof(Type) * numproc);
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type));
  hipMalloc(&recvbuf_d, count * sizeof(Type) * numproc);
#else
  sendbuf_d = new Type[count];
  recvbuf_d = new Type[count * numproc];
#endif

  Type *sendbuf = new Type[count];
  Type *recvbuf = new Type[count * numproc];

  for(size_t i = 0; i < count; i++)
    sendbuf[i].data[0] = myid;
#ifdef PORT_CUDA
  cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), cudaMemcpyHostToDevice);
#elif defined PORT_HIP
  hipMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), hipMemcpyHostToDevice);
#else
  memcpy(sendbuf_d, sendbuf, count * sizeof(Type));
#endif

  {
    using namespace CommBench;
    
    int numlevel = 3;
    int groupsize[numlevel - 1] = {6, 3};
    
    Arch arch(numlevel, groupsize, MPI_COMM_WORLD);
    
    capability cap[numlevel + 1] = {MEMCPY, MPI, IPC, NCCL};
    
    
    Allgather<Type> test(sendbuf_d, count , recvbuf_d, arch, cap);
  
    double totalTime = 0;
    double totalData = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      //test.start();
      test.waitall();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
       totalTime += time;
       totalData += 2 * count * sizeof(Type) / 1.e9;
       if(myid == ROOT)
         printf("time: %e\n", time);
      }
    }
    if(myid == ROOT)
      printf("totalData %.2e totalTime %.2e B/W: %.2e GB/s time: %.2e s/GB --- CommBench::Allgather(3|6,3)\n", totalData, totalTime, totalData / totalTime, totalTime / totalData);
  
  }

  {
    CommBench::Allgather<Type> test(sendbuf_d, count, recvbuf_d, MPI_COMM_WORLD, CommBench::MPI);

    double totalTime = 0;
    double totalData = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      //test.start();
      test.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
       totalTime += time;
       totalData += 2 * count * sizeof(Type) / 1.e9;
       if(myid == ROOT)
         printf("time: %e\n", time);
      }
    }
    if(myid == ROOT)
      printf("totalData %.2e totalTime %.2e --- CommBench::Allgather(MPI)\n", totalData, totalTime);
  }
#ifdef CAP_NCCL
  {
    CommBench::Allgather<Type> test(sendbuf_d, count, recvbuf_d, MPI_COMM_WORLD, CommBench::NCCL);

    double totalTime = 0;
    double totalData = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      //test.start();
      test.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
       totalTime += time;
       totalData += 2 * count * sizeof(Type) / 1.e9;
       if(myid == ROOT)
         printf("time: %e\n", time);
      }
    }
    if(myid == ROOT)
      printf("totalData %.2e totalTime %.2e --- CommBench::Allgather(NCCL)\n", totalData, totalTime);
  }
#endif
  {
    double totalTime = 0;
    double totalData = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      MPI_Allgather(sendbuf_d, count * sizeof(Type), MPI_BYTE, recvbuf_d, count * sizeof(Type), MPI_BYTE, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
       totalTime += time;
       totalData += 2 * count * sizeof(Type) / 1.e9;
       if(myid == ROOT)
         printf("time: %e\n", time);
      }
    }
    if(myid == ROOT)
      printf("totalData %.2e totalTime %.2e --- MPI_Allgather\n", totalData, totalTime);
  }
#ifdef CAP_NCCL
  {
    ncclComm_t comm_nccl;
    ncclUniqueId id;
    if(myid == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm_nccl, numproc, id, myid);

    double totalTime = 0;
    double totalData = 0;
    for (int iter = -10; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      ncclAllGather(sendbuf_d, recvbuf_d, count * sizeof(Type), ncclInt8, comm_nccl, 0);
      cudaStreamSynchronize(0);
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup time: %e\n", time);
      }
      else {
       totalTime += time;
       totalData += 2 * count * sizeof(Type) / 1.e9;
       if(myid == ROOT)
         printf("time: %e\n", time);
      }
    }
    if(myid == ROOT)
      printf("totalData %.2e totalTime %.2e B/W: %.2e GB/s time: %.2e s/GB --- ncclAllGather\n", totalData, totalTime, totalData / totalTime, totalTime / totalData);
  }
#endif

#ifdef PORT_CUDA
  cudaMemcpy(recvbuf, recvbuf_d, count * sizeof(Type) * numproc, cudaMemcpyDeviceToHost);
#elif defined PORT_HIP
  hipMemcpy(recvbuf, recvbuf_d, count * sizeof(Type) * numproc, hipMemcpyDeviceToHost);
#else
  memcpy(recvbuf, recvbuf_d, count * sizeof(Type) * numproc);
#endif

  bool pass = true;
  for(int p = 0; p < numproc; p++)
    for(size_t i = 0; i < count; i++)
      if(recvbuf[p * count + i].data[0] != p)
        pass = false;
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  if(pass && myid == ROOT)
    printf("PASS!\n");
  else
    if(myid == ROOT)
      printf("ERROR!!!!\n");

  delete[] sendbuf;
  delete[] recvbuf;
#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
#else
  delete[] sendbuf_d;
  delete[] recvbuf_d;
#endif
}
