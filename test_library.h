{

  int numgroup = numproc / groupsize;

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

  double totalData = 0;
  double totalTime = 0;
  double minTime = 1e9;
  double minData = 2 * count * (numgroup - 1) * sizeof(Type) / 1.e9 * groupsize * groupsize;
  for (int iter = -warmup; iter < numiter; iter++) {
#if !defined(PORT_CUDA) && !defined(PORT_HIP)
    memset(sendbuf_d, -1, count * numproc * sizeof(Type));
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    MPI_Alltoall(sendbuf_d, count * sizeof(Type), MPI_BYTE, recvbuf_d, count * sizeof(Type), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    if(iter < 0) {
      if(myid == ROOT)
        printf("warmup: %.2e\n", time);
    }
    else {
      if(time < minTime)
        minTime = time;
      if(myid == ROOT)
        printf("time: %.2e\n", time);
     totalTime += time;
     totalData += minData;
    }
  }
  if(myid == ROOT) {
    printf("minTime %.2e minData %.2e MB (%.2e s/GB) B/W %.2e GB/s --- MPI\n", minTime, minData * 1e3, minTime / minData, minData / minTime);
    printf("totalTime %.2e s totalData %.2e GB B/W %.2e (%.2e max) GB/s --- MPI\n", totalTime, totalData, totalData / totalTime, minData / minTime);
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
