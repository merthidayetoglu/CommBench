{

  MPI_Comm comm_rail;
  MPI_Comm_split(MPI_COMM_WORLD, myid % groupsize, myid / groupsize, &comm_rail);

  int numgroup;
  int mygroup;
  MPI_Comm_size(comm_rail, &numgroup);
  MPI_Comm_rank(comm_rail, &mygroup);

  printf("myid %d numgroup %d mygroup %d\n", myid, numgroup, mygroup);


  Type *sendbuf_d;
  Type *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numgroup * sizeof(Type));
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type));
  hipMalloc(&recvbuf_d, count * numgroup * sizeof(Type));
#else
  sendbuf_d = new Type[count];
  recvbuf_d = new Type[count * numgroup];
#endif

  {
    CommBench::Comm<Type> bench(comm_rail, CommBench::TEST_CAPABILITY);
    for(int sender = 0; sender < numgroup; sender++)
      for(int recver = 0; recver < numgroup; recver++)
        if(sender != recver)
          bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * groupsize * (numgroup - 1);
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
     printf("TEST_G2G_rail\n");
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
