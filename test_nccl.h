{

  Type *sendbuf_d;
  Type *recvbuf_d;

#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type) * numproc);
  cudaMalloc(&recvbuf_d, count * sizeof(Type) * numproc);
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type) * numproc);
  hipMalloc(&recvbuf_d, count * sizeof(Type) * numproc);
#else
  sendbuf_d = new Type[count * numproc];
  recvbuf_d = new Type[count * numproc];
#endif

#ifdef CAP_NCCL
    ncclComm_t comm_nccl;
#endif
#ifdef PORT_CUDA
    cudaStream_t stream_nccl;
#elif defined PORT_HIP
    hipStream_t stream_nccl;
#endif

  Type *recvbuf = new Type[count];
  Type *sendbuf = new Type[count];

  for(size_t i = 0; i < count; i++)
    sendbuf[i].data[0] = i * 0.1;

  {

#ifdef CAP_NCCL
    ncclUniqueId id;
    if(myid == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm_nccl, numproc, id, myid);
#ifdef PORT_CUDA
    cudaStreamCreate(&stream_nccl);
#elif defined PORT_HIP
    hipStreamCreate(&stream_nccl);
#endif
#endif

    double times[numiter];
    for (int iter = -warmup; iter < numiter; iter++) {
#if defined PORT_CUDA
      //cudaMemset(sendbuf_d, 0, count * sizeof(Type) * numproc);
      cudaMemset(sendbuf_d, -1, count * sizeof(Type) * numproc);
      if(myid == 0)
        cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(Type), cudaMemcpyHostToDevice);
      cudaMemset(recvbuf_d, -1, count * sizeof(Type) * numproc);
#elif defined PORT_HIP
      hipMemset(sendbuf_d, -1, count * sizeof(Type) * numproc);
      hipMemset(recvbuf_d, -1, count * sizeof(Type) * numproc);
#else
      memset(sendbuf_d, -1, count * sizeof(Type) * numproc);
      memset(recvbuf_d, -1, count * sizeof(Type) * numproc);
#endif

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();

      // ncclResult_t result = ncclAllGathersendbuf_d, recvbuf_d, count, ncclFloat, comm_nccl, stream_nccl);
      //ncclResult_t result = ncclReduce(sendbuf_d, recvbuf_d, count, ncclFloat, ncclMax, 0, comm_nccl, stream_nccl);
      cudaEventRecord(start);
      ncclResult_t result = ncclBroadcast(sendbuf_d, sendbuf_d, count, ncclFloat, 0, comm_nccl, stream_nccl);
      cudaEventRecord(stop);

      // MPI_Allgather(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, MPI_COMM_WORLD);
      // MPI_Alltoall(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, MPI_COMM_WORLD);
      // MPI_Scatter(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
      // MPI_Bcast(sendbuf_d, count, MPI_FLOAT, 0, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);
      cudaStreamSynchronize(stream_nccl);
      time = MPI_Wtime() - time;
      cudaMemcpy(recvbuf, recvbuf_d, count * sizeof(Type), cudaMemcpyDeviceToHost);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      if(myid == ROOT)
        printf("%e milliseconds\n", milliseconds);

      if(myid == numproc-1)
        for(int i = 0; i < 10; i++)
          printf("%d\n", recvbuf[i].data[0]);
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup: %.2e %d\n", time, result);
      }
      else {
        times[iter] = time;
      }
    }

    std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

    if(myid == ROOT)
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

    double minTime, medTime, maxTime, avgTime;
    minTime = times[0];
    medTime = times[numiter / 2];
    maxTime = times[numiter - 1];
    avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;

    double data = count * sizeof(Type) / 1.e9;
    if(myid == ROOT) {
      printf("TEST_G2G_rail (%d)\n", subgroupsize);
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }

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
