{
  int numgroup = numproc / groupsize;

  Type *sendbuf_d;
  Type *recvbuf_d;

  Type *sendbuf_h;
  Type *recvbuf_h;

#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type) * numgroup);
  cudaMalloc(&recvbuf_d, count * sizeof(Type) * numgroup);
  cudaMallocHost(&sendbuf_h, count * sizeof(Type) * numgroup);
  cudaMallocHost(&recvbuf_h, count * sizeof(Type) * numgroup);
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type) * numgroup);
  hipMalloc(&recvbuf_d, count * sizeof(Type) * numgroup);
  hipHostMalloc(&sendbuf_h, count * sizeof(Type) * numgroup);
  hipHostMalloc(&recvbuf_h, count * sizeof(Type) * numgroup);
#endif

  if(myid == 0)
  {
    CommBench::Comm<Type> bench(MPI_COMM_SELF, CommBench::TEST_CAPABILITY);

    for(int sender = 0; sender < numgroup; sender++)
      for(int recver = 0; recver < numgroup; recver++)
        bench.add(sendbuf_d, recver * count, recvbuf_d, sender * count, count, 0, 0);

    double data = 2 * count * sizeof(Type) / 1.e9 * numgroup * numgroup;

    bench.report();

    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
     printf("TEST_P2P (%d)\n", subgroupsize);
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }

 
  if(myid < numgroup) 
  {
    cudaStream_t sendstream;
    cudaStream_t recvstream;
    cudaStreamCreate(&sendstream);
    cudaStreamCreate(&recvstream);
    double minTime = 1e9;
    for(int iter = -warmup; iter < numiter; iter++) {
      double time = MPI_Wtime();
#ifdef PORT_CUDA
      cudaMemcpyAsync(sendbuf_d, recvbuf_h, count * sizeof(Type) * numgroup, cudaMemcpyDeviceToHost, sendstream);
#ifdef TEST_BIDIRECTIONAL
      cudaMemcpyAsync(sendbuf_h, recvbuf_d, count * sizeof(Type) * numgroup, cudaMemcpyHostToDevice, recvstream);
      cudaStreamSynchronize(recvstream);
#endif
      cudaStreamSynchronize(sendstream);
#elif defined PORT_HIP
      hipMemcpyAsync(sendbuf_d, recvbuf_h, count * sizeof(Type) * numgroup, sendstream);
#ifdef TEST_BIDIRECTIONAL
      hipMemcpyAsync(sendbuf_h, recvbuf_d, count * sizeof(Type) * numgroup, cudaMemcpyHostToDevice, recvstream);
      hipStreamSynchronize(recvstream);
#endif
      hipStreamSynchronize(sendstream);
#endif
      time = MPI_Wtime() - time;
      if(time < minTime)
        minTime = time;
    }
    if(myid == ROOT)
#ifdef TEST_BIDIRECTIONAL
      printf("time %e bandwidth %e\n", minTime, 2 * count * sizeof(Type) * numgroup / minTime / 1e9);
#else
      printf("time %e bandwidth %e\n", minTime, count * sizeof(Type) * numgroup / minTime / 1e9);
#endif
  }

#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
  cudaFreeHost(sendbuf_h);
  cudaFreeHost(recvbuf_h);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
  hipFreeHost(sendbuf_h);
  hipFreeHost(recvbuf_h);
#endif

}
