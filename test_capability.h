{
  if(myid == ROOT)
    printf("TEST ALL-TO-ALL CAPABILITIES\n");

  // Alltoall
  size_t sendcount[numproc];
  size_t recvcount[numproc];
  size_t sendoffset[numproc];
  size_t recvoffset[numproc];
  for (int p = 0; p < numproc; p++) {
    sendcount[p] = count;
    recvcount[p] = count;
    sendoffset[p] = p * count;
    recvoffset[p] = p * count;
  }

  Type *sendbuf = new Type[count * numproc];
  Type *recvbuf = new Type[count * numproc];

  for (int p = 0; p < numproc; p++)
    for (size_t i = 0; i < count; i++)
      sendbuf[p * count + i].data[0] = p;

  Type *sendbuf_d;
  Type *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * numproc * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numproc * sizeof(Type));
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * numproc * sizeof(Type));
  hipMalloc(&recvbuf_d, count * numproc * sizeof(Type));
#else
  sendbuf_d = new Type[count * numproc];
  recvbuf_d = new Type[count * numproc];
#endif

#ifdef PORT_CUDA
  cudaMemcpy(sendbuf_d, sendbuf, count * numproc * sizeof(Type), cudaMemcpyHostToDevice);
  cudaMemset(recvbuf_d, -1, count * numproc * sizeof(Type));
#elif defined PORT_HIP
  hipMemcpy(sendbuf_d, sendbuf, count * numproc * sizeof(Type), hipMemcpyHostToDevice);
  hipMemset(recvbuf_d, -1, count * numproc * sizeof(Type));
#else
  memcpy(sendbuf_d, sendbuf, count * numproc * sizeof(Type));
  memset(recvbuf_d, -1, count * numproc * sizeof(Type));
#endif

  {
    CommBench::Comm<Type> test(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, MPI_COMM_WORLD, CommBench::MPI);

    double totalData = 0;
    double totalTime = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      test.start();
      double start = MPI_Wtime() - time;
      test.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("start %.2e warmup: %.2e\n", start, time);
      }
      else {
        if(myid == ROOT)
          printf("start %.2e time: %.2e\n", start, time);
       totalTime += time;
       totalData += count * numproc * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime %.2e --- MPI\n", totalTime);

  }
  {
    CommBench::Comm<Type> test(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, MPI_COMM_WORLD, CommBench::NCCL);

    double totalData = 0;
    double totalTime = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      test.start();
      double start = MPI_Wtime() - time;
      test.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("start %.2e warmup: %.2e\n", start, time);
      }
      else {
        if(myid == ROOT)
          printf("start %.2e time: %.2e\n", start, time);
       totalTime += time;
       totalData += count * numproc * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime %.2e --- NCCL\n", totalTime);

  }
  {
    CommBench::Comm<Type> test(sendbuf_d, sendcount, sendoffset, recvbuf_d, recvcount, recvoffset, MPI_COMM_WORLD, CommBench::IPC);

    double totalData = 0;
    double totalTime = 0;
    for (int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      test.start();
      double start = MPI_Wtime() - time;
      test.wait();
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("start %.2e warmup: %.2e\n", start, time);
      }
      else {
        if(myid == ROOT)
          printf("start %.2e time: %.2e\n", start, time);
       totalTime += time;
       totalData += count * numproc * sizeof(Type) / 1.e9;
      }
    }
    if(myid == ROOT)
      printf("totalTime %.2e --- IPC\n", totalTime);

  }

#ifdef PORT_CUDA
  cudaMemcpy(recvbuf, recvbuf_d, count * numproc * sizeof(Type), cudaMemcpyDeviceToHost);
#elif defined PORT_HIP
  hipMemcpy(recvbuf, recvbuf_d, count * numproc * sizeof(Type), hipMemcpyDeviceToHost);
#else
  memcpy(recvbuf, recvbuf_d, count * numproc * sizeof(Type));
#endif

  bool pass = true;
  for(int p = 0; p < numproc; p++)
    for(size_t i = 0; i < count; i++)
      if(recvbuf[p * count + i].data[0] != myid)
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
